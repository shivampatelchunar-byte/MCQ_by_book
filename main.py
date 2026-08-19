"""
AI MCQ Generator System — v3.0 (Vision-OCR Edition)
=====================================================

WORKFLOW
--------
1. A SCANNED (image-based) textbook PDF lives in Google Drive.
2. Every page is rendered as an image and read with Gemini Vision OCR
   (there is no real text layer to extract, so Vision is the primary path).
3. Gemini Vision also reads the PRINTED page number from the page's own
   header/footer — this is what we use for topic-mapping and for the
   "Book Page" column in the output Sheet. The PDF's own internal page
   count is NEVER used for anything user-facing; it only drives the
   read-loop and resume pointer.
4. MCQs are generated from the OCR'd text via a 6-provider fallback chain.
5. Results are appended to the user's Google Sheet, and a live "Progress"
   tab in that SAME spreadsheet is kept up to date after every page, so
   progress is visible without needing Telegram at all.
6. A Telegram bot (webhook-based, no polling) lets the user configure the
   PDF/Sheet, start/pause the worker, and — importantly — say things like
   "reset to page 45" to resume from a specific PRINTED book page. The
   system maintains a page-number map + a one-time calibration offset to
   translate a requested book page into the correct PDF page index.
"""

import os
import re
import json
import time
import random
import threading
import traceback
from datetime import datetime, timezone

import gspread
import gdown
import pymupdf as fitz
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from pymongo import MongoClient
from google.oauth2.service_account import Credentials
import google.generativeai as genai
from openai import OpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes


# =====================================================================
# 1. ENVIRONMENT VARIABLES & SECRETS
# =====================================================================
MONGO_URI = os.getenv("MONGO_URI")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GCP_SERVICE_ACCOUNT_JSON = os.getenv("GCP_SERVICE_ACCOUNT_JSON")

GEMINI_KEY_1 = os.getenv("GEMINI_API_KEY_1")
GEMINI_KEY_2 = os.getenv("GEMINI_API_KEY_2")
GROQ_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
MISTRAL_KEY = os.getenv("MISTRAL_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
CEREBRAS_KEY = os.getenv("CEREBRAS_API_KEY")
SAMBANOVA_KEY = os.getenv("SAMBANOVA_API_KEY")

REQUIRED_VARS = {
    "MONGO_URI": MONGO_URI,
    "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN,
    "GCP_SERVICE_ACCOUNT_JSON": GCP_SERVICE_ACCOUNT_JSON,
    "GEMINI_API_KEY_1": GEMINI_KEY_1,  # powers BOTH the Vision OCR and the Telegram intent agent
}
_missing = [k for k, v in REQUIRED_VARS.items() if not v]
if _missing:
    print(f"❌ FATAL: Missing required environment variables: {', '.join(_missing)}")
    print("   The app will start but core features will fail until these are set on Render.")


# =====================================================================
# 2. MONGODB — STATE / RESUME TRACKING
# =====================================================================
print("📡 Connecting to MongoDB...")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["mcq_agent_db"]
config_col = db["system_config"]      # single-document master config
page_map_col = db["page_map"]         # pdf_index -> printed book page(s) seen there
error_log_col = db["error_logs"]

DEFAULT_SYSTEM_PROMPT = (
    "Generate 5 to 10 exam-oriented MCQs from the given text. 60% direct, 40% tricky. "
    "Output strictly in English. Ensure no consecutive duplicate correct answers."
)

# Create the config doc if this is a brand-new deployment.
config_col.update_one(
    {"_id": "master_config"},
    {"$setOnInsert": {
        "sheet_url": "",
        "pdf_drive_link": "",
        "worker_status": "paused",
        "current_page": 1,             # PDF-internal page pointer (loop/resume control ONLY)
        "book_page_offset": None,      # PDF index where printed book-page "1" was found
        "last_book_page_label": "",    # e.g. "3-4" — for display only
        "total_questions_generated": 0,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    }},
    upsert=True
)
# Safe migration: add any newer fields to a pre-existing config doc without
# touching progress that's already been made.
config_col.update_one(
    {"_id": "master_config", "book_page_offset": {"$exists": False}},
    {"$set": {"book_page_offset": None}}
)
config_col.update_one(
    {"_id": "master_config", "last_book_page_label": {"$exists": False}},
    {"$set": {"last_book_page_label": ""}}
)
print("✅ Configuration ready!")


# =====================================================================
# 3. GEMINI SETUP (Vision OCR + Telegram Intent Agent)
# =====================================================================
GEMINI_MODEL_NAME = "gemini-3.6-flash"  # multimodal — handles both text and image input
vision_model = None
agent_model = None
try:
    genai.configure(api_key=GEMINI_KEY_1)
    vision_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    agent_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    print("✅ Gemini configured successfully!")
except Exception as e:
    print(f"⚠️ Gemini config warning: {e}")


def call_gemini_with_retry(model, contents, max_retries: int = 3, base_delay: float = 5.0):
    """
    Calls model.generate_content() with exponential backoff.
    Rate-limit errors (429 / quota / RESOURCE_EXHAUSTED) get a longer backoff
    than generic transient errors.
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return model.generate_content(contents)
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            is_rate_limit = any(s in msg for s in ("429", "quota", "resource_exhausted", "rate limit"))
            delay = base_delay * (3 if is_rate_limit else 1) * attempt + random.uniform(0, 2)
            print(f"⚠️ Gemini call failed (attempt {attempt}/{max_retries}): {e} — retrying in {delay:.0f}s")
            time.sleep(delay)
    raise last_err


# =====================================================================
# 4. GOOGLE SHEETS & DRIVE TOOLS
# =====================================================================
def get_gspread_client():
    """Authenticated Google Sheets client (service account)."""
    creds_dict = json.loads(GCP_SERVICE_ACCOUNT_JSON)
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)


MCQ_SHEET_HEADERS = [
    "Serial No", "Book Page", "Topic", "Question",
    "Option A", "Option B", "Option C", "Option D", "Option E",
    "Correct Answer", "Explanation", "Processed At (UTC)"
]


def get_or_create_worksheet(spreadsheet, title: str, rows: int = 100, cols: int = 12):
    """Returns the worksheet with this title, creating it if missing."""
    try:
        return spreadsheet.worksheet(title)
    except gspread.exceptions.WorksheetNotFound:
        return spreadsheet.add_worksheet(title=title, rows=rows, cols=cols)


def ensure_mcq_sheet_headers(mcq_sheet):
    """Writes the header row once, if the sheet is currently empty."""
    first_row = mcq_sheet.row_values(1)
    if not first_row:
        mcq_sheet.update("A1:L1", [MCQ_SHEET_HEADERS])


def sync_progress_sheet(spreadsheet, config: dict, total_pdf_pages: int = None):
    """
    Writes a live, human-readable status block into a 'Progress' tab of the
    SAME spreadsheet, so the user can see exactly where the system is
    without needing Telegram. This is overwritten in place (not appended).
    """
    try:
        progress_ws = get_or_create_worksheet(spreadsheet, "Progress", rows=20, cols=2)
        pdf_pointer_display = (
            f"{config.get('current_page', 1)}/{total_pdf_pages}"
            if total_pdf_pages else str(config.get("current_page", 1))
        )
        rows = [
            ["AI MCQ Generator — Live Status", ""],
            ["", ""],
            ["Worker Status", config.get("worker_status", "unknown").upper()],
            ["PDF Page Pointer (internal)", pdf_pointer_display],
            ["Last Book Page Processed", config.get("last_book_page_label", "—")],
            ["Book Page 1 Calibrated At (PDF pg)", config.get("book_page_offset") or "not yet calibrated"],
            ["Total MCQs Generated", config.get("total_questions_generated", 0)],
            ["PDF Source Link", config.get("pdf_drive_link", "—")],
            ["Last Updated (UTC)", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")],
            ["", ""],
            ["To resume from a specific printed page, tell the Telegram bot:", ""],
            ["  \"reset to page 45\"", ""],
        ]
        progress_ws.update("A1:B12", rows)
    except Exception as e:
        # Progress sync is a nice-to-have — never let it break the worker.
        print(f"⚠️ Progress sheet sync failed (non-fatal): {e}")


def download_pdf_from_drive(drive_link: str, output_path: str = "/tmp/current_book.pdf") -> str:
    """Downloads a PDF from a Google Drive share link."""
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", drive_link)
    if not match:
        raise ValueError("Invalid Google Drive link format — expected '.../d/<FILE_ID>/...'")

    file_id = match.group(1)
    download_url = f"https://drive.google.com/uc?id={file_id}"
    print(f"📥 Downloading PDF from Drive: {download_url}")
    gdown.download(download_url, output_path, quiet=False)

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("PDF download failed — output file missing or empty.")

    print(f"✅ PDF downloaded successfully: {os.path.getsize(output_path)} bytes")
    return output_path


# =====================================================================
# 5. VISION OCR — reads scanned page images + printed page numbers
# =====================================================================
def ocr_page_with_gemini_vision(doc, page_index: int) -> dict:
    """
    Renders one PDF page as a PNG and asks Gemini Vision to:
      (a) transcribe all body text, and
      (b) read the PRINTED page number(s) from the header/footer.

    This book is scanned as two-page spreads, so a single PDF page can
    show two printed book pages side by side — hence a LIST of page
    numbers is returned, not a single int.

    Returns: {"text": str, "page_numbers": list[int]}
    """
    try:
        page = doc.load_page(page_index)
        pix = page.get_pixmap(dpi=200)  # 200 DPI balances OCR accuracy vs payload size
        img_bytes = pix.tobytes("png")

        prompt = (
            "This is a scanned spread from an agriculture exam textbook. It may show "
            "one or two printed book pages side by side. Respond in EXACTLY this format:\n"
            "PAGE_NUMBERS: <comma-separated printed page number(s) found in the header/footer "
            "of the image, left-to-right, digits only, e.g. '3,4'. If no printed page number "
            "is visible, write NONE.>\n"
            "---\n"
            "<all readable body text from the page(s), exactly as written, preserving "
            "paragraph structure. Ignore scanner watermarks like 'Scanned with...'. "
            "If genuinely blank, leave this section empty.>"
        )
        response = call_gemini_with_retry(
            vision_model, [prompt, {"mime_type": "image/png", "data": img_bytes}]
        )
        raw = (response.text or "").strip()

        page_numbers, body_text = [], raw
        if raw.upper().startswith("PAGE_NUMBERS:"):
            header_line, _, rest = raw.partition("\n")
            nums_part = header_line.split(":", 1)[1].strip()
            if nums_part.upper() != "NONE":
                page_numbers = [int(n) for n in re.findall(r"\d+", nums_part)]
            body_text = rest.split("---", 1)[-1].strip() if "---" in rest else rest.strip()

        return {"text": body_text, "page_numbers": page_numbers}

    except Exception as e:
        print(f"⚠️ Vision OCR failed for PDF page {page_index + 1}: {e}")
        return {"text": "", "page_numbers": []}


def resolve_book_page_label(detected_numbers: list, fallback_pdf_page: int):
    """
    Converts OCR-detected printed page number(s) into:
      - a representative int for topic-range lookups (smallest number found)
      - a display label ('3-4' for a spread, '3' for a single page, or a
        clearly-flagged fallback if OCR couldn't read any page number).
    """
    if detected_numbers:
        numbers = sorted(set(detected_numbers))
        representative = numbers[0]
        label = f"{numbers[0]}-{numbers[-1]}" if len(numbers) > 1 else str(numbers[0])
        return representative, label
    return fallback_pdf_page, f"PDF-p{fallback_pdf_page} (unread)"


def get_topic_for_page(book_page_num: int) -> str:
    """Maps a PRINTED book page number to its syllabus topic (per the book's TOC)."""
    if 1 <= book_page_num <= 27:
        return "General Agriculture"
    elif 28 <= book_page_num <= 214:
        return "Agronomy"
    elif 215 <= book_page_num <= 318:
        return "Soil Science"
    elif 319 <= book_page_num <= 338:
        return "Agrometeorology"
    elif 339 <= book_page_num <= 407:
        return "Animal Husbandry and Dairy Science"
    elif 408 <= book_page_num <= 466:
        return "Agricultural Extension"
    elif 467 <= book_page_num <= 540:
        return "Agricultural Economics"
    elif 541 <= book_page_num <= 571:
        return "Agricultural Statistics"
    elif book_page_num >= 572:
        return "Agricultural Engineering"
    else:
        return "Preliminary / Index"


# =====================================================================
# 6. PYDANTIC SCHEMAS
# =====================================================================
class SingleMCQ(BaseModel):
    question: str = Field(description="1-2 lines concise exam-oriented question")
    option_a: str = Field(description="Option A")
    option_b: str = Field(description="Option B")
    option_c: str = Field(description="Option C")
    option_d: str = Field(description="Option D")
    option_e: str = Field(default="None of these", description="Always 'None of these'")
    correct_answer: str = Field(description="Strictly A, B, C, D, or E")
    explanation: str = Field(description="1-3 lines factual explanation in pure English")


class MCQList(BaseModel):
    mcqs: list[SingleMCQ]


# =====================================================================
# 7. MULTI-TIER AI FALLBACK ENGINE (MCQ generation from OCR'd text)
# =====================================================================
TIERS = [
    {"name": "Groq", "base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile", "key": GROQ_KEY},
    {"name": "Cerebras", "base_url": "https://api.cerebras.ai/v1", "model": "llama-3.3-70b", "key": CEREBRAS_KEY},
    {"name": "Mistral", "base_url": "https://api.mistral.ai/v1", "model": "mistral-small-latest", "key": MISTRAL_KEY},
    {"name": "GitHub-Azure", "base_url": "https://models.inference.ai.azure.com", "model": "gpt-4o", "key": GITHUB_TOKEN},
    {"name": "SambaNova", "base_url": "https://api.sambanova.ai/v1", "model": "Meta-Llama-3.1-70B-Instruct", "key": SAMBANOVA_KEY},
    {"name": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", "model": "openai/gpt-oss-20b:free", "key": OPENROUTER_KEY},
]


def generate_mcqs_with_fallback(page_text: str, custom_prompt: str) -> MCQList:
    """Generates MCQs from OCR'd page text, trying each provider tier in order."""
    full_prompt = f"{custom_prompt}\n\nPAGE TEXT:\n{page_text}\n\nReturn output strictly matching the required JSON schema."

    for tier in TIERS:
        if not tier["key"]:
            continue
        try:
            print(f"🔄 Trying {tier['name']}...")
            client = OpenAI(api_key=tier["key"], base_url=tier["base_url"])
            response = client.chat.completions.create(
                model=tier["model"],
                messages=[
                    {"role": "system", "content": "You are an Expert Agricultural Exam Setter. Output exclusively in valid JSON."},
                    {"role": "user", "content": full_prompt}
                ],
                response_format={"type": "json_object"},
                timeout=45
            )
            raw_data = response.choices[0].message.content
            parsed_data = json.loads(raw_data)

            if "mcqs" not in parsed_data and isinstance(parsed_data, list):
                parsed_data = {"mcqs": parsed_data}

            mcq_list = MCQList(**parsed_data)
            if not mcq_list.mcqs:
                raise ValueError("Provider returned zero MCQs")

            print(f"✅ {tier['name']} succeeded — {len(mcq_list.mcqs)} MCQs")
            return mcq_list

        except Exception as e:
            print(f"❌ [{tier['name']} Failed]: {e}")
            time.sleep(3)
            continue

    raise RuntimeError("Critical Failure: All AI provider tiers exhausted for this page.")


# =====================================================================
# 8. BACKGROUND WORKER — the page-by-page read → OCR → MCQ → Sheet loop
# =====================================================================
def background_worker_process():
    """
    Main autonomous loop. Design goals:
      - Never die: every page is wrapped so one bad page can't kill the thread.
      - Always uses Vision OCR (book is 100% scanned) to read each page image
        AND its printed page number.
      - Tracks progress both in MongoDB (fast, resume-safe) and in a
        'Progress' tab inside the user's own Google Sheet (visible, editable).
    """
    print("🚀 Background Worker Started!")

    while True:
        try:
            config = config_col.find_one({"_id": "master_config"})

            if config["worker_status"] != "running":
                time.sleep(10)
                continue
            if not config["sheet_url"]:
                print("⚠️ No Google Sheet configured — waiting...")
                time.sleep(10)
                continue
            if not config["pdf_drive_link"]:
                print("⚠️ No PDF Drive link configured — waiting...")
                time.sleep(10)
                continue

            # --- Ensure PDF is available locally ---
            pdf_path = "/tmp/current_book.pdf"
            if not os.path.exists(pdf_path):
                download_pdf_from_drive(config["pdf_drive_link"], pdf_path)

            doc = fitz.open(pdf_path)
            pdf_pointer = config["current_page"]        # PDF-internal loop index (1-based)
            total_pdf_pages = len(doc)

            if pdf_pointer > total_pdf_pages:
                print("✅ All pages processed!")
                config_col.update_one({"_id": "master_config"}, {"$set": {"worker_status": "completed"}})
                doc.close()
                time.sleep(60)
                continue

            print(f"📖 Processing PDF page {pdf_pointer}/{total_pdf_pages} (Vision OCR)...")

            # --- Vision OCR (primary and only extraction method — book is scanned) ---
            ocr_result = ocr_page_with_gemini_vision(doc, pdf_pointer - 1)
            page_text = ocr_result["text"]
            detected_numbers = ocr_result["page_numbers"]

            if len(page_text) > 100:
                book_page_num, book_page_label = resolve_book_page_label(detected_numbers, pdf_pointer)
                topic = get_topic_for_page(book_page_num)
                print(f"📚 Book Page: {book_page_label} | Topic: {topic} | Text length: {len(page_text)} chars")

                # One-time calibration: remember which PDF page shows printed "page 1".
                # This lets us later estimate the PDF page for any requested book page
                # (this book is scanned at 2 printed pages per PDF page).
                if config.get("book_page_offset") is None and 1 in detected_numbers:
                    config_col.update_one(
                        {"_id": "master_config"},
                        {"$set": {"book_page_offset": pdf_pointer}}
                    )
                    print(f"🎯 Calibrated: printed book page 1 = PDF page {pdf_pointer}")

                # Remember exactly which PDF page contains which printed page(s),
                # for exact (non-estimated) jumps later via "reset to page X".
                if detected_numbers:
                    page_map_col.update_one(
                        {"_id": pdf_pointer},
                        {"$set": {"book_pages": detected_numbers, "topic": topic}},
                        upsert=True
                    )

                mcq_data = generate_mcqs_with_fallback(page_text, config["system_prompt"])

                gc = get_gspread_client()
                spreadsheet = gc.open_by_url(config["sheet_url"])
                mcq_sheet = spreadsheet.sheet1
                ensure_mcq_sheet_headers(mcq_sheet)

                start_serial = config["total_questions_generated"] + 1
                now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

                rows_to_append = []
                for idx, item in enumerate(mcq_data.mcqs):
                    rows_to_append.append([
                        start_serial + idx, book_page_label, topic, item.question,
                        item.option_a, item.option_b, item.option_c, item.option_d, item.option_e,
                        item.correct_answer, item.explanation, now_utc
                    ])
                mcq_sheet.append_rows(rows_to_append)
                print(f"📊 Added {len(rows_to_append)} MCQ rows to Google Sheet")

                config_col.update_one(
                    {"_id": "master_config"},
                    {
                        "$inc": {"current_page": 1, "total_questions_generated": len(rows_to_append)},
                        "$set": {"last_book_page_label": book_page_label},
                    }
                )
                # Refresh the live status tab with the latest numbers.
                updated_config = config_col.find_one({"_id": "master_config"})
                sync_progress_sheet(spreadsheet, updated_config, total_pdf_pages)

            else:
                print(f"⏭️ PDF page {pdf_pointer} produced no usable text even after OCR — skipping.")
                config_col.update_one({"_id": "master_config"}, {"$inc": {"current_page": 1}})

            doc.close()
            time.sleep(5)  # pacing — avoids hammering AI provider rate limits

        except Exception as e:
            print(f"❌ Worker Error: {e}")
            traceback.print_exc()
            try:
                error_log_col.insert_one({
                    "timestamp": time.time(),
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
            except Exception:
                pass
            time.sleep(15)


# =====================================================================
# 9. TELEGRAM AGENT — natural-language control (webhook mode, no polling)
# =====================================================================
AGENT_PROMPT = """
You are the Autonomous AI Manager for a scanned-textbook MCQ Generation System.
Analyze the user's natural language input (Hindi/Hinglish/English) and return a
STRICT JSON object representing the action to take.

Valid actions:
1. update_sheet: User provided a Google Sheet link. Extract the full URL.
2. update_pdf: User provided a Google Drive PDF link. Extract the full URL.
3. update_prompt: User wants to change how MCQs are generated. Extract the new instructions.
4. start_worker: User wants to start/resume the system from wherever it currently is.
5. pause_worker: User wants to stop/pause the system.
6. reset_page: User wants to jump to a specific PRINTED BOOK page number (e.g.
   "page 45 se start karo", "reset to page 45"). Extract ONLY the digits as extracted_value.
7. status_report: User is asking for the current status/progress.
8. help: User is asking what commands/features are available.
9. general_chat: None of the above, just chat.

Return ONLY this JSON format:
{"action": "action_name", "extracted_value": "value if applicable, else empty string", "reply_message": "A natural, friendly response to the user confirming the action, in the same language style they used."}
"""

HELP_TEXT = (
    "🤖 *Yahan kya-kya bol sakte ho:*\n\n"
    "• Google Sheet ka link bhejo → output MCQs isi Sheet mein jayenge\n"
    "• Google Drive PDF ka link bhejo → ye book process hogi\n"
    "• \"start\" ya \"resume kro\" → worker start/resume\n"
    "• \"pause\" ya \"stop\" → worker rok do\n"
    "• \"reset to page 45\" → book ke PRINTED page 45 se dobara start\n"
    "• \"status\" → abhi kahan tak kaam hua hai\n\n"
    "Progress hamesha aapki Google Sheet ke *Progress* tab mein bhi live dikhta rehta hai."
)


def _estimate_pdf_page_for_book_page(target_book_page: int, config: dict) -> tuple:
    """
    Resolves a requested PRINTED book page to a PDF page index.
    Returns (pdf_page_index_or_None, source_description).
    """
    exact = page_map_col.find_one({"book_pages": target_book_page})
    if exact:
        return exact["_id"], "exact match from processing history"

    offset = config.get("book_page_offset")
    if offset:
        # This book is scanned at 2 printed pages per PDF page (a spread).
        estimated = max(1, offset + (target_book_page - 1) // 2)
        return estimated, f"estimated (book page 1 calibrated at PDF page {offset})"

    return None, None


async def handle_telegram_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles one incoming Telegram text message via the Gemini intent agent."""
    user_text = update.message.text
    user_name = update.effective_user.first_name
    print(f"💬 Telegram from {user_name}: {user_text}")

    if agent_model is None:
        await update.message.reply_text("❌ AI Agent not configured. Please check GEMINI_API_KEY_1 on the server.")
        return

    try:
        response = call_gemini_with_retry(
            agent_model,
            f"{AGENT_PROMPT}\n\nUser Input: {user_text}",
        )
        intent = json.loads(response.text)
        action = intent.get("action")
        value = intent.get("extracted_value", "") or ""
        reply = intent.get("reply_message", "Action acknowledged.")
        print(f"🤖 Intent: {action} | Value: {value[:50]}")

        if action == "update_sheet":
            config_col.update_one({"_id": "master_config"}, {"$set": {"sheet_url": value}})
            reply = f"✅ Google Sheet updated!\n\n{reply}"

        elif action == "update_pdf":
            config_col.update_one(
                {"_id": "master_config"},
                {"$set": {"pdf_drive_link": value, "current_page": 1, "book_page_offset": None}}
            )
            page_map_col.delete_many({})  # new book = old page map is meaningless
            if os.path.exists("/tmp/current_book.pdf"):
                os.remove("/tmp/current_book.pdf")
            reply = f"✅ PDF updated, progress reset to page 1, cache cleared!\n\n{reply}"

        elif action == "update_prompt":
            config_col.update_one({"_id": "master_config"}, {"$set": {"system_prompt": value}})
            reply = f"✅ System prompt updated!\n\n{reply}"

        elif action == "start_worker":
            config_col.update_one({"_id": "master_config"}, {"$set": {"worker_status": "running"}})
            reply = f"🚀 Worker started!\n\n{reply}"

        elif action == "pause_worker":
            config_col.update_one({"_id": "master_config"}, {"$set": {"worker_status": "paused"}})
            reply = f"⏸️ Worker paused.\n\n{reply}"

        elif action == "reset_page":
            digits = re.findall(r"\d+", value)
            if not digits:
                reply = "⚠️ Page number samajh nahi aaya. Aise likho: \"reset to page 45\""
            else:
                target_book_page = int(digits[0])
                config = config_col.find_one({"_id": "master_config"})
                new_pdf_index, source = _estimate_pdf_page_for_book_page(target_book_page, config)

                if new_pdf_index:
                    config_col.update_one(
                        {"_id": "master_config"},
                        {"$set": {"current_page": new_pdf_index, "worker_status": "running"}}
                    )
                    reply = (
                        f"🔁 Reset to book page {target_book_page} (PDF page {new_pdf_index}, {source}).\n"
                        f"Worker resumed. Agar 1-2 page ke baad mapping galat lage, mujhe bata dena."
                    )
                else:
                    reply = (
                        "⚠️ Abhi tak page-number calibration nahi hui hai (printed 'page 1' abhi tak scan "
                        "nahi hua). Pehle PDF page 1 se chalne do taaki system offset seekh le, uske baad "
                        "kisi bhi page pe jump kar sakte ho."
                    )

        elif action == "status_report":
            config = config_col.find_one({"_id": "master_config"})
            reply = (
                f"📊 *System Status*\n\n"
                f"Status: {config['worker_status'].upper()}\n"
                f"PDF Page Pointer: {config['current_page']}\n"
                f"Last Book Page: {config.get('last_book_page_label') or '—'}\n"
                f"MCQs Generated: {config['total_questions_generated']}\n"
                f"PDF: {'✅' if config['pdf_drive_link'] else '❌'}\n"
                f"Sheet: {'✅' if config['sheet_url'] else '❌'}\n\n"
                f"(Sheet ke 'Progress' tab mein bhi live status dikhta hai)"
            )

        elif action == "help":
            reply = HELP_TEXT

        await update.message.reply_text(reply)

    except Exception as e:
        print(f"❌ Telegram Error: {e}")
        traceback.print_exc()
        await update.message.reply_text(f"❌ Agent Error: {e}\nPlease try rephrasing your request.")


# Built once at import time. Runs inside FastAPI's own event loop via
# process_update() — no separate thread/event loop, no competing
# getUpdates() calls, so there's no "Conflict" error on redeploy.
telegram_app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_telegram_message))

WEBHOOK_PATH = f"/telegram-webhook/{TELEGRAM_BOT_TOKEN}"
EXTERNAL_URL = (os.getenv("WEBHOOK_URL") or os.getenv("RENDER_EXTERNAL_URL", "")).rstrip("/")


# =====================================================================
# 10. FASTAPI APP
# =====================================================================
app = FastAPI()
start_time = time.time()


@app.on_event("startup")
async def on_startup():
    await telegram_app.initialize()
    await telegram_app.start()
    if EXTERNAL_URL:
        webhook_url = f"{EXTERNAL_URL}{WEBHOOK_PATH}"
        await telegram_app.bot.set_webhook(url=webhook_url, drop_pending_updates=True)
        print(f"✅ Telegram webhook registered: {webhook_url}")
    else:
        print("⚠️ No RENDER_EXTERNAL_URL/WEBHOOK_URL found — webhook NOT set. "
              "Set WEBHOOK_URL env var manually to your public Render URL if needed.")


@app.on_event("shutdown")
async def on_shutdown():
    try:
        await telegram_app.bot.delete_webhook()
    except Exception:
        pass
    await telegram_app.stop()
    await telegram_app.shutdown()


@app.post(WEBHOOK_PATH)
async def telegram_webhook(request: Request):
    """Receives updates pushed by Telegram."""
    data = await request.json()
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return {"ok": True}


@app.get("/")
def keep_alive():
    """Health check / uptime-ping endpoint."""
    try:
        config = config_col.find_one({"_id": "master_config"})
        return {
            "service": "AI_MCQ_Agent_Active",
            "status": config["worker_status"],
            "current_page": config["current_page"],
            "total_questions_generated": config["total_questions_generated"],
            "timestamp": time.time(),
        }
    except Exception:
        return {"service": "AI_MCQ_Agent_Active", "status": "starting", "timestamp": time.time()}


@app.get("/health")
def health_check():
    """Detailed health/diagnostics check."""
    try:
        config = config_col.find_one({"_id": "master_config"})
        return {
            "status": "healthy",
            "uptime_seconds": round(time.time() - start_time),
            "mongo": "connected",
            "worker_status": config["worker_status"],
            "current_page": config["current_page"],
            "api_keys": {
                "gemini": bool(GEMINI_KEY_1),
                "groq": bool(GROQ_KEY),
                "cerebras": bool(CEREBRAS_KEY),
                "mistral": bool(MISTRAL_KEY),
                "github": bool(GITHUB_TOKEN),
                "sambanova": bool(SAMBANOVA_KEY),
                "openrouter": bool(OPENROUTER_KEY),
            },
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


# =====================================================================
# 11. ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("🚀 AI MCQ Generator System v3.0 Starting...")
    print("=" * 60)
    print(f"📊 MongoDB: {'Connected' if mongo_client else 'Failed'}")
    print(f"🤖 Telegram: {'Configured' if TELEGRAM_BOT_TOKEN else 'Missing'}")
    print(f"👁️  Gemini Vision: {'Configured' if vision_model else 'Missing'}")
    print(f"📚 AI Providers: {sum(1 for t in TIERS if t['key'])} of {len(TIERS)} configured")
    print("=" * 60)

    try:
        worker_thread = threading.Thread(target=background_worker_process, daemon=True)
        worker_thread.start()
        print("✅ Background Worker Thread Started")
    except Exception as e:
        print(f"❌ Worker thread error: {e}")

    # Telegram runs in webhook mode via FastAPI's own event loop (see startup event above).

    try:
        port = int(os.environ.get("PORT", 8080))
        print(f"✅ Starting FastAPI on port {port}...")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ FastAPI error: {e}")
        traceback.print_exc()
