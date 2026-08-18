import os
import io
import re
import json
import time
import asyncio
import threading
import traceback
from datetime import datetime
from typing import Optional, List, Tuple, Dict
import gspread
import requests
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

# ============================================================
# 1. ENVIRONMENT VARIABLES
# ============================================================
MONGO_URI = os.getenv("MONGO_URI")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GCP_SERVICE_ACCOUNT_JSON = os.getenv("GCP_SERVICE_ACCOUNT_JSON")

GEMINI_KEY_1 = os.getenv("GEMINI_API_KEY_1")
GEMINI_KEY_2 = os.getenv("GEMINI_API_KEY_2")

# ============================================================
# 2. GEMINI MODELS - ONLY FLASH-LITE
# ============================================================
GEMINI_MODELS = [
    {
        "name": "Gemini 3.5 Flash-Lite",
        "model": "gemini-1.5-flash",
        "description": "Fast, efficient, cost-optimized"
    },
    {
        "name": "Gemini 3.1 Flash-Lite", 
        "model": "gemini-1.5-flash-8b",
        "description": "Cost-efficient, high-volume"
    }
]

GEMINI_KEYS = [key for key in [GEMINI_KEY_1, GEMINI_KEY_2] if key]

if GEMINI_KEYS:
    genai.configure(api_key=GEMINI_KEYS[0])
    print(f"✅ Gemini configured with {len(GEMINI_KEYS)} keys")
    print(f"📚 Models: {', '.join([m['name'] for m in GEMINI_MODELS])}")
else:
    print("⚠️ No Gemini keys configured")

# ============================================================
# 3. MONGODB SETUP
# ============================================================
print("📡 Connecting to MongoDB...")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["mcq_agent_db"]
config_col = db["system_config"]

if not config_col.find_one({"_id": "master_config"}):
    print("🆕 Creating initial configuration...")
    config_col.insert_one({
        "_id": "master_config",
        "sheet_url": "",
        "pdf_drive_link": "",
        "worker_status": "running",
        "current_page": 1,  # PDF page index (starts from 1)
        "last_processed_pdf_page": 0,
        "total_questions_generated": 0,
        "pages_completed": [],  # Stores PDF page numbers
        "book_pages_mapping": {},  # {pdf_page: book_page}
        "failed_pages": [],
        "system_prompt": """Generate 5 to 10 exam-oriented MCQs from the given text.
        60% direct questions, 40% tricky/application-based.
        All options must be plausible.
        Correct answer must be strictly A, B, C, D, or E.
        Explanation must be 1-3 lines factual.
        Focus on agricultural concepts.""",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    })
    print("✅ Configuration created!")
else:
    print("✅ Configuration loaded!")
    config = config_col.find_one({"_id": "master_config"})
    if config.get("worker_status") != "running":
        config_col.update_one({"_id": "master_config"}, {"$set": {"worker_status": "running"}})

# ============================================================
# 4. GOOGLE SHEETS & DRIVE
# ============================================================
def get_gspread_client():
    try:
        creds_dict = json.loads(GCP_SERVICE_ACCOUNT_JSON)
        scopes = ["https://www.googleapis.com/auth/spreadsheets", 
                  "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        return gspread.authorize(creds)
    except Exception as e:
        print(f"❌ Google Sheets auth error: {e}")
        raise

def download_pdf_from_drive(drive_link: str, output_path: str = "/tmp/current_book.pdf"):
    """Downloads PDF from Google Drive"""
    try:
        patterns = [
            r"/d/([a-zA-Z0-9_-]+)",
            r"id=([a-zA-Z0-9_-]+)",
            r"file/d/([a-zA-Z0-9_-]+)"
        ]
        
        file_id = None
        for pattern in patterns:
            match = re.search(pattern, drive_link)
            if match:
                file_id = match.group(1)
                break
        
        if not file_id:
            raise ValueError("Invalid Google Drive Link format")
        
        download_url = f"https://drive.google.com/uc?id={file_id}"
        print(f"📥 Downloading PDF...")
        gdown.download(download_url, output_path, quiet=False)
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✅ PDF downloaded: {file_size/1024/1024:.2f} MB")
            return output_path
        else:
            raise Exception("PDF download failed")
            
    except Exception as e:
        print(f"❌ PDF download error: {e}")
        raise

# ============================================================
# 5. GEMINI VISION - EXTRACT TEXT & PAGE NUMBER
# ============================================================
def extract_page_with_gemini_vision(doc, page_index: int) -> Dict:
    """
    Extract text AND printed page number from scanned PDF page.
    Returns: {"text": str, "book_page_number": int or None}
    """
    
    for gemini_model in GEMINI_MODELS:
        for key_idx in range(len(GEMINI_KEYS)):
            try:
                print(f"🔍 OCR with {gemini_model['name']} (Key {key_idx+1})...")
                
                if key_idx > 0:
                    genai.configure(api_key=GEMINI_KEYS[key_idx])
                
                vision_model = genai.GenerativeModel(gemini_model['model'])
                
                page = doc.load_page(page_index)
                pix = page.get_pixmap(dpi=200)
                img_bytes = pix.tobytes("png")
                
                response = vision_model.generate_content([
                    """
                    IMPORTANT: This is a scanned page from an agriculture textbook.
                    
                    TASKS:
                    1. Find the PRINTED PAGE NUMBER in the header or footer of this page.
                       - Look for numbers like "Page 123", "123", or "- 123 -"
                       - If two pages are side by side, find both page numbers
                       - Return the number(s) you find
                    
                    2. Extract ALL readable body text from this page.
                       - Preserve paragraph structure
                       - Ignore scanner watermarks
                       - Ignore headers and footers (except page numbers)
                    
                    RESPOND IN THIS EXACT FORMAT:
                    PAGE_NUMBERS: <comma-separated numbers found, e.g. 3,4 or 123>
                    
                    ---
                    <extracted body text here>
                    """,
                    {"mime_type": "image/png", "data": img_bytes}
                ])
                
                raw = (response.text or "").strip()
                
                # Parse page numbers
                book_page_number = None
                body_text = raw
                
                # Look for PAGE_NUMBERS: line
                if "PAGE_NUMBERS:" in raw.upper():
                    parts = raw.upper().split("PAGE_NUMBERS:", 1)
                    if len(parts) > 1:
                        num_line = parts[1].split("---", 1)[0].strip()
                        # Extract all numbers from the line
                        numbers = re.findall(r'\b(\d{1,4})\b', num_line)
                        if numbers:
                            # Take the first number (or smallest if multiple)
                            book_page_number = int(numbers[0])
                            print(f"📄 Found book page number: {book_page_number}")
                        
                        # Get body text after ---
                        if "---" in raw:
                            body_text = raw.split("---", 1)[1].strip()
                        else:
                            body_text = raw
                
                # If no page number found, try to find any number in first 200 chars
                if book_page_number is None:
                    first_chars = raw[:300]
                    numbers = re.findall(r'\b(\d{1,4})\b', first_chars)
                    # Filter out common non-page numbers
                    for num in numbers:
                        n = int(num)
                        if 1 <= n <= 1000 and n not in [2024, 2025, 2026, 100, 200, 300]:
                            book_page_number = n
                            print(f"📄 Detected possible page number: {book_page_number}")
                            break
                
                if len(body_text) > 50:
                    print(f"✅ OCR succeeded ({len(body_text)} chars)")
                    return {
                        "text": body_text,
                        "book_page_number": book_page_number
                    }
                    
            except Exception as e:
                print(f"⚠️ OCR with {gemini_model['name']} failed: {str(e)[:80]}")
                time.sleep(1)
                continue
    
    return {"text": "", "book_page_number": None}

# ============================================================
# 6. PYDANTIC SCHEMAS
# ============================================================
class SingleMCQ(BaseModel):
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    option_e: str = "None of these"
    correct_answer: str
    explanation: str

class MCQList(BaseModel):
    mcqs: List[SingleMCQ]

# ============================================================
# 7. GEMINI MCQ GENERATOR
# ============================================================
def generate_mcqs_with_gemini(page_text: str, custom_prompt: str) -> MCQList:
    """Generate MCQs using Gemini Flash-Lite models"""
    page_text = page_text[:12000]
    
    full_prompt = f"""
{custom_prompt}

PAGE TEXT:
{page_text}

IMPORTANT: Return ONLY valid JSON with 'mcqs' array.
Each MCQ: {{"question": "...", "option_a": "...", "option_b": "...", "option_c": "...", "option_d": "...", "option_e": "None of these", "correct_answer": "A", "explanation": "..."}}
"""
    
    for gemini_model in GEMINI_MODELS:
        for key_idx in range(len(GEMINI_KEYS)):
            try:
                print(f"🔄 Trying {gemini_model['name']} (Key {key_idx+1})...")
                
                if key_idx > 0:
                    genai.configure(api_key=GEMINI_KEYS[key_idx])
                
                model = genai.GenerativeModel(gemini_model['model'])
                
                response = model.generate_content(
                    full_prompt,
                    generation_config={
                        "temperature": 0.3,
                        "top_p": 0.95,
                        "top_k": 40,
                        "max_output_tokens": 8192,
                        "response_mime_type": "application/json"
                    }
                )
                
                raw_data = response.text.strip()
                raw_data = re.sub(r'^```json\s*', '', raw_data)
                raw_data = re.sub(r'\s*```$', '', raw_data)
                
                parsed_data = json.loads(raw_data)
                
                if "mcqs" not in parsed_data and isinstance(parsed_data, list):
                    parsed_data = {"mcqs": parsed_data}
                elif "mcqs" not in parsed_data:
                    for key, value in parsed_data.items():
                        if isinstance(value, list) and len(value) > 0:
                            parsed_data = {"mcqs": value}
                            break
                
                mcq_list = MCQList(**parsed_data)
                print(f"✅ {gemini_model['name']} succeeded! Generated {len(mcq_list.mcqs)} MCQs")
                return mcq_list
                
            except Exception as e:
                print(f"❌ {gemini_model['name']} failed: {str(e)[:100]}")
                time.sleep(2)
                continue
    
    raise RuntimeError("All Gemini Flash-Lite models failed")

# ============================================================
# 8. TOPIC MAPPING - BASED ON BOOK PAGE NUMBER
# ============================================================
def get_topic_for_page(book_page_num: int) -> str:
    """Maps book page number to topic"""
    if book_page_num is None:
        return "Unknown"
    elif book_page_num <= 27:
        return "General Agriculture"
    elif book_page_num <= 214:
        return "Agronomy"
    elif book_page_num <= 318:
        return "Soil Science"
    elif book_page_num <= 338:
        return "Agrometeorology"
    elif book_page_num <= 407:
        return "Animal Husbandry"
    elif book_page_num <= 466:
        return "Agricultural Extension"
    elif book_page_num <= 540:
        return "Agricultural Economics"
    elif book_page_num <= 571:
        return "Agricultural Statistics"
    elif book_page_num >= 572:
        return "Agricultural Engineering"
    else:
        return "General"

# ============================================================
# 9. TELEGRAM BOT
# ============================================================
AGENT_PROMPT = """
You are the AI Manager for MCQ Generator.

Analyze user input and return JSON action.

Actions:
1. update_sheet: user provides Google Sheet URL
2. update_pdf: user provides Google Drive PDF URL  
3. start_worker: user says "start" or "resume"
4. pause_worker: user says "pause" or "stop"
5. status_report: user asks "status" or "progress"
6. reset_page: user says "reset to page [number]"
7. general_chat: anything else

Return: {"action": "action_name", "extracted_value": "value", "reply_message": "response"}
"""

def get_agent_response(user_text: str) -> dict:
    """Get agent response using Gemini Flash-Lite"""
    
    for gemini_model in GEMINI_MODELS:
        for key_idx in range(len(GEMINI_KEYS)):
            try:
                if key_idx > 0:
                    genai.configure(api_key=GEMINI_KEYS[key_idx])
                
                model = genai.GenerativeModel(gemini_model['model'])
                
                response = model.generate_content(
                    f"{AGENT_PROMPT}\n\nUser: {user_text}",
                    generation_config={
                        "temperature": 0.1,
                        "response_mime_type": "application/json"
                    }
                )
                
                return json.loads(response.text)
                
            except Exception as e:
                print(f"⚠️ Agent failed: {str(e)[:80]}")
                continue
    
    return {
        "action": "general_chat",
        "extracted_value": "",
        "reply_message": "How can I help with MCQ generation?"
    }

async def handle_telegram_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    user_name = update.effective_user.first_name
    print(f"💬 {user_name}: {user_text}")
    
    try:
        intent = get_agent_response(user_text)
        action = intent.get("action")
        value = intent.get("extracted_value", "")
        reply = intent.get("reply_message", "Done")
        
        config = config_col.find_one({"_id": "master_config"})
        
        if action == "update_sheet":
            config_col.update_one({"_id": "master_config"}, {"$set": {"sheet_url": value}})
            reply = f"✅ Sheet updated!\n\n{reply}"
            
        elif action == "update_pdf":
            config_col.update_one({"_id": "master_config"}, {"$set": {"pdf_drive_link": value}})
            if os.path.exists("/tmp/current_book.pdf"):
                os.remove("/tmp/current_book.pdf")
            reply = f"✅ PDF updated!\n\n{reply}"
            
        elif action == "start_worker":
            config_col.update_one({"_id": "master_config"}, {"$set": {"worker_status": "running"}})
            reply = f"🚀 Worker started from page {config['current_page']}\n\n{reply}"
            
        elif action == "pause_worker":
            config_col.update_one({"_id": "master_config"}, {"$set": {"worker_status": "paused"}})
            reply = f"⏸️ Worker paused at page {config['current_page']}\n\n{reply}"
            
        elif action == "reset_page":
            try:
                page_num = int(value)
                config_col.update_one({"_id": "master_config"}, {"$set": {"current_page": page_num}})
                reply = f"✅ Reset to PDF page {page_num}\n\n{reply}"
            except:
                reply = f"❌ Invalid page number\n\n{reply}"
            
        elif action == "status_report":
            pages_completed = len(config.get("pages_completed", []))
            mapping = config.get("book_pages_mapping", {})
            last_book_page = mapping.get(str(config["last_processed_pdf_page"]), "Unknown")
            
            reply = (
                f"📊 **System Status**\n\n"
                f"Status: {config['worker_status'].upper()}\n"
                f"PDF Page: {config['current_page']}\n"
                f"Last Book Page: {last_book_page}\n"
                f"Pages Done: {pages_completed}\n"
                f"MCQs Generated: {config['total_questions_generated']}\n"
                f"PDF: {'✅' if config['pdf_drive_link'] else '❌'}\n"
                f"Sheet: {'✅' if config['sheet_url'] else '❌'}"
            )
        
        await update.message.reply_text(reply)
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(f"❌ Telegram Error: {e}")
        await update.message.reply_text(error_msg)

# ============================================================
# 10. BACKGROUND WORKER - WITH BOOK PAGE DETECTION
# ============================================================
def background_worker_process():
    print("🚀 Background Worker Started!")
    print(f"📚 Using models: {', '.join([m['name'] for m in GEMINI_MODELS])}")
    print("📄 Will detect book page numbers from header/footer")
    
    while True:
        try:
            config = config_col.find_one({"_id": "master_config"})
            
            if config["worker_status"] != "running":
                time.sleep(10)
                continue
                
            if not config["pdf_drive_link"]:
                print("⚠️ Waiting for PDF link...")
                time.sleep(10)
                continue
                
            if not config["sheet_url"]:
                print("⚠️ Waiting for Sheet link...")
                time.sleep(10)
                continue

            pdf_path = "/tmp/current_book.pdf"
            if not os.path.exists(pdf_path):
                print("📥 Downloading PDF...")
                download_pdf_from_drive(config["pdf_drive_link"], pdf_path)
            
            doc = fitz.open(pdf_path)
            current_pdf_page = config["current_page"]  # PDF page index (1-based)
            total_pages = len(doc)
            
            print(f"📖 Processing PDF Page {current_pdf_page}/{total_pages}")
            
            if current_pdf_page > total_pages:
                print("✅ All PDF pages processed!")
                config_col.update_one({"_id": "master_config"}, {"$set": {"worker_status": "completed"}})
                doc.close()
                time.sleep(60)
                continue

            # ============================================================
            # STEP 1: Extract text AND book page number using Vision
            # ============================================================
            print(f"🔍 Scanning PDF page {current_pdf_page} with Gemini Vision...")
            result = extract_page_with_gemini_vision(doc, current_pdf_page - 1)
            
            page_text = result.get("text", "")
            book_page_number = result.get("book_page_number", None)
            
            # If no book page number found, use PDF page as fallback
            if book_page_number is None:
                book_page_number = current_pdf_page
                print(f"⚠️ No book page number found, using PDF page {current_pdf_page}")
            else:
                print(f"📄 Detected book page: {book_page_number}")

            # ============================================================
            # STEP 2: Skip empty pages
            # ============================================================
            if len(page_text) < 50:
                print(f"⏭️ Page {current_pdf_page} empty, skipping...")
                config_col.update_one(
                    {"_id": "master_config"}, 
                    {
                        "$inc": {"current_page": 1},
                        "$set": {"last_processed_pdf_page": current_pdf_page}
                    }
                )
                doc.close()
                time.sleep(2)
                continue

            # ============================================================
            # STEP 3: Generate MCQs
            # ============================================================
            topic = get_topic_for_page(book_page_number)
            print(f"📚 Topic: {topic} | Book Page: {book_page_number} | Text: {len(page_text)} chars")
            
            print("🤖 Generating MCQs with Gemini Flash-Lite...")
            mcq_data = generate_mcqs_with_gemini(page_text, config["system_prompt"])
            print(f"✅ Generated {len(mcq_data.mcqs)} MCQs")

            # ============================================================
            # STEP 4: Save to Google Sheet with BOOK PAGE NUMBER
            # ============================================================
            print("📊 Saving to Google Sheet...")
            gc = get_gspread_client()
            sheet = gc.open_by_url(config["sheet_url"]).sheet1
            
            existing = len(sheet.get_all_values())
            start_serial = max(1, existing)
            
            rows = []
            for idx, item in enumerate(mcq_data.mcqs):
                rows.append([
                    start_serial + idx,           # Serial Number
                    book_page_number,             # ACTUAL BOOK PAGE NUMBER
                    topic,                        # Topic
                    item.question,                # Question
                    item.option_a,                # Option A
                    item.option_b,                # Option B
                    item.option_c,                # Option C
                    item.option_d,                # Option D
                    item.option_e,                # Option E
                    item.correct_answer,          # Answer
                    item.explanation,             # Explanation
                    datetime.now().isoformat(),   # Timestamp
                    current_pdf_page              # PDF Page (for reference)
                ])
            
            sheet.append_rows(rows)
            print(f"📊 Added {len(rows)} rows (Book Page: {book_page_number})")

            # ============================================================
            # STEP 5: Update progress with mapping
            # ============================================================
            pages_completed = config.get("pages_completed", [])
            pages_completed.append(current_pdf_page)
            
            book_pages_mapping = config.get("book_pages_mapping", {})
            book_pages_mapping[str(current_pdf_page)] = book_page_number
            
            config_col.update_one(
                {"_id": "master_config"},
                {
                    "$inc": {
                        "current_page": 1, 
                        "total_questions_generated": len(rows)
                    },
                    "$set": {
                        "last_processed_pdf_page": current_pdf_page,
                        "pages_completed": pages_completed,
                        "book_pages_mapping": book_pages_mapping,
                        "updated_at": datetime.now().isoformat()
                    }
                }
            )
            
            doc.close()
            print(f"⏳ Waiting 3 seconds before next page...")
            time.sleep(3)

        except Exception as e:
            print(f"❌ Worker Error: {e}")
            traceback.print_exc()
            
            try:
                error_log = db["error_logs"]
                error_log.insert_one({
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })
            except:
                pass
            
            time.sleep(30)

# ============================================================
# 11. FASTAPI APP
# ============================================================
app = FastAPI()

telegram_app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_telegram_message))

WEBHOOK_PATH = f"/webhook/{TELEGRAM_BOT_TOKEN}"
EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "").rstrip("/")

@app.on_event("startup")
async def startup():
    await telegram_app.initialize()
    await telegram_app.start()
    if EXTERNAL_URL:
        webhook_url = f"{EXTERNAL_URL}{WEBHOOK_PATH}"
        await telegram_app.bot.set_webhook(url=webhook_url)
        print(f"✅ Webhook: {webhook_url}")

@app.on_event("shutdown")
async def shutdown():
    try:
        await telegram_app.bot.delete_webhook()
    except:
        pass
    await telegram_app.stop()

@app.post(WEBHOOK_PATH)
async def webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return {"ok": True}

@app.get("/")
def home():
    config = config_col.find_one({"_id": "master_config"})
    mapping = config.get("book_pages_mapping", {})
    return {
        "service": "MCQ Generator - Book Page Detection",
        "models": [m["name"] for m in GEMINI_MODELS],
        "status": config["worker_status"],
        "pdf_page": config["current_page"],
        "mcqs": config["total_questions_generated"],
        "pages_mapped": len(mapping)
    }

@app.get("/mapping")
def get_mapping():
    """Get the PDF to Book page mapping"""
    config = config_col.find_one({"_id": "master_config"})
    return config.get("book_pages_mapping", {})

@app.post("/reset/{pdf_page}")
def reset_to_page(pdf_page: int):
    """Reset to specific PDF page"""
    config_col.update_one(
        {"_id": "master_config"}, 
        {"$set": {"current_page": pdf_page}}
    )
    return {"message": f"Reset to PDF page {pdf_page}", "page": pdf_page}

# ============================================================
# 12. MAIN ENTRY
# ============================================================
if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 MCQ Generator - Book Page Detection Edition")
    print("=" * 60)
    print(f"📊 MongoDB: ✅ Connected")
    print(f"🔑 Gemini Keys: {len(GEMINI_KEYS)} configured")
    print(f"📚 Models:")
    for m in GEMINI_MODELS:
        print(f"   - {m['name']}")
    print("📄 Feature: Auto-detect book page numbers from header/footer")
    print("=" * 60)
    
    worker = threading.Thread(target=background_worker_process, daemon=True)
    worker.start()
    print("✅ Worker started")
    
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
