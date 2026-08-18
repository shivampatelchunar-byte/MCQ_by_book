import os
import io
import re
import json
import time
import asyncio
import threading
import traceback
from datetime import datetime
from typing import Optional, List, Tuple
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
GROQ_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
MISTRAL_KEY = os.getenv("MISTRAL_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
CEREBRAS_KEY = os.getenv("CEREBRAS_API_KEY")
SAMBANOVA_KEY = os.getenv("SAMBANOVA_API_KEY")

# ============================================================
# 2. MONGODB SETUP
# ============================================================
print("📡 Connecting to MongoDB...")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["mcq_agent_db"]
config_col = db["system_config"]

# Initialize with RUNNING status so worker starts immediately
if not config_col.find_one({"_id": "master_config"}):
    print("🆕 Creating initial configuration...")
    config_col.insert_one({
        "_id": "master_config",
        "sheet_url": "",
        "pdf_drive_link": "",
        "worker_status": "running",  # ✅ CHANGED: Start immediately
        "current_page": 1,
        "total_questions_generated": 0,
        "last_processed_page": 0,
        "pages_completed": [],
        "failed_pages": [],
        "system_prompt": "Generate 5 to 10 exam-oriented MCQs from the given text. 60% direct, 40% tricky. Output strictly in English.",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    })
    print("✅ Configuration created with RUNNING status!")
else:
    print("✅ Configuration loaded!")
    # Ensure worker status is running if not set
    config = config_col.find_one({"_id": "master_config"})
    if config.get("worker_status") != "running":
        print("🔄 Setting worker status to RUNNING...")
        config_col.update_one({"_id": "master_config"}, {"$set": {"worker_status": "running"}})

# ============================================================
# 3. GOOGLE SHEETS & DRIVE
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
        # Extract file ID from link
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
# 4. GEMINI VISION OCR - FOR SCANNED PDFs
# ============================================================
def ocr_page_with_gemini_vision(doc, page_index: int) -> str:
    """Extract text from scanned PDF page using Gemini Vision"""
    try:
        page = doc.load_page(page_index)
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        
        vision_model = genai.GenerativeModel('gemini-1.5-flash')
        
        response = vision_model.generate_content([
            """
            Extract ALL readable text from this scanned page.
            - Preserve paragraph structure
            - Ignore headers, footers, and page numbers
            - Return only the main body text
            """,
            {"mime_type": "image/png", "data": img_bytes}
        ])
        
        return (response.text or "").strip()
        
    except Exception as e:
        print(f"⚠️ OCR failed for page {page_index + 1}: {e}")
        return ""

# ============================================================
# 5. PYDANTIC SCHEMAS
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
# 6. AI FALLBACK ENGINE
# ============================================================
TIERS = [
    {"name": "Groq", "base_url": "https://api.groq.com/openai/v1", 
     "model": "llama-3.3-70b-versatile", "key": GROQ_KEY},
    {"name": "Mistral", "base_url": "https://api.mistral.ai/v1", 
     "model": "mistral-small-latest", "key": MISTRAL_KEY},
    {"name": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", 
     "model": "openai/gpt-oss-20b:free", "key": OPENROUTER_KEY},
]

def generate_mcqs_with_fallback(page_text: str, custom_prompt: str) -> MCQList:
    page_text = page_text[:12000]
    
    full_prompt = f"""
{custom_prompt}

PAGE TEXT:
{page_text}

Return ONLY valid JSON with 'mcqs' array.
Each MCQ: {{"question": "...", "option_a": "...", "option_b": "...", "option_c": "...", "option_d": "...", "option_e": "None of these", "correct_answer": "A", "explanation": "..."}}
"""
    
    for tier in TIERS:
        if not tier["key"]:
            continue
        try:
            print(f"🔄 Trying {tier['name']}...")
            client = OpenAI(api_key=tier["key"], base_url=tier["base_url"])
            response = client.chat.completions.create(
                model=tier["model"],
                messages=[
                    {"role": "system", "content": "You are an Expert Agricultural Exam Setter. Output valid JSON only."},
                    {"role": "user", "content": full_prompt}
                ],
                response_format={"type": "json_object"},
                timeout=60
            )
            raw_data = response.choices[0].message.content
            parsed_data = json.loads(raw_data)
            if "mcqs" not in parsed_data and isinstance(parsed_data, list):
                parsed_data = {"mcqs": parsed_data}
            print(f"✅ {tier['name']} succeeded!")
            return MCQList(**parsed_data)
        except Exception as e:
            print(f"❌ {tier['name']} failed: {str(e)[:100]}")
            time.sleep(3)
            continue
    raise RuntimeError("All AI providers failed")

# ============================================================
# 7. TOPIC MAPPING
# ============================================================
def get_topic_for_page(page_num: int) -> str:
    if page_num <= 27: return "General Agriculture"
    elif page_num <= 214: return "Agronomy"
    elif page_num <= 318: return "Soil Science"
    elif page_num <= 338: return "Agrometeorology"
    elif page_num <= 407: return "Animal Husbandry"
    elif page_num <= 466: return "Agricultural Extension"
    elif page_num <= 540: return "Agricultural Economics"
    elif page_num <= 571: return "Agricultural Statistics"
    elif page_num >= 572: return "Agricultural Engineering"
    else: return "General"

# ============================================================
# 8. BACKGROUND WORKER - MAIN LOGIC
# ============================================================
def background_worker_process():
    print("🚀 Background Worker Started!")
    
    while True:
        try:
            config = config_col.find_one({"_id": "master_config"})
            
            # Check if worker should run
            if config["worker_status"] != "running":
                time.sleep(10)
                continue
                
            # Check if PDF and Sheet are configured
            if not config["pdf_drive_link"]:
                print("⚠️ Waiting for PDF link...")
                time.sleep(10)
                continue
                
            if not config["sheet_url"]:
                print("⚠️ Waiting for Sheet link...")
                time.sleep(10)
                continue

            # Download PDF if not exists
            pdf_path = "/tmp/current_book.pdf"
            if not os.path.exists(pdf_path):
                print("📥 Downloading PDF...")
                download_pdf_from_drive(config["pdf_drive_link"], pdf_path)
            
            doc = fitz.open(pdf_path)
            current_page = config["current_page"]
            total_pages = len(doc)
            
            print(f"📖 Processing Page {current_page}/{total_pages}")
            
            # Check completion
            if current_page > total_pages:
                print("✅ All pages processed!")
                config_col.update_one({"_id": "master_config"}, {"$set": {"worker_status": "completed"}})
                doc.close()
                time.sleep(60)
                continue

            # ============================================================
            # STEP 1: Extract text
            # ============================================================
            page = doc.load_page(current_page - 1)
            page_text = page.get_text("text").strip()
            
            # If text extraction failed (scanned PDF), use OCR
            if len(page_text) < 50:
                print(f"🔍 Using Gemini Vision OCR for page {current_page}...")
                page_text = ocr_page_with_gemini_vision(doc, current_page - 1)
                print(f"📝 OCR extracted {len(page_text)} chars")

            # ============================================================
            # STEP 2: Skip empty pages
            # ============================================================
            if len(page_text) < 50:
                print(f"⏭️ Page {current_page} empty, skipping...")
                config_col.update_one({"_id": "master_config"}, {"$inc": {"current_page": 1}})
                doc.close()
                time.sleep(2)
                continue

            # ============================================================
            # STEP 3: Generate MCQs
            # ============================================================
            topic = get_topic_for_page(current_page)
            print(f"📚 Topic: {topic} | Text: {len(page_text)} chars")
            
            print("🤖 Generating MCQs...")
            mcq_data = generate_mcqs_with_fallback(page_text, config["system_prompt"])
            print(f"✅ Generated {len(mcq_data.mcqs)} MCQs")

            # ============================================================
            # STEP 4: Save to Google Sheet
            # ============================================================
            print("📊 Saving to Google Sheet...")
            gc = get_gspread_client()
            sheet = gc.open_by_url(config["sheet_url"]).sheet1
            
            existing = len(sheet.get_all_values())
            start_serial = max(1, existing)
            
            rows = []
            for idx, item in enumerate(mcq_data.mcqs):
                rows.append([
                    start_serial + idx,
                    current_page,
                    topic,
                    item.question,
                    item.option_a,
                    item.option_b,
                    item.option_c,
                    item.option_d,
                    item.option_e,
                    item.correct_answer,
                    item.explanation,
                    datetime.now().isoformat()
                ])
            
            sheet.append_rows(rows)
            print(f"📊 Added {len(rows)} rows")

            # ============================================================
            # STEP 5: Update progress
            # ============================================================
            pages_completed = config.get("pages_completed", [])
            pages_completed.append(current_page)
            
            config_col.update_one(
                {"_id": "master_config"},
                {
                    "$inc": {"current_page": 1, "total_questions_generated": len(rows)},
                    "$set": {
                        "last_processed_page": current_page,
                        "pages_completed": pages_completed,
                        "updated_at": datetime.now().isoformat()
                    }
                }
            )
            
            doc.close()
            print(f"⏳ Waiting 3 seconds...")
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
# 9. TELEGRAM BOT
# ============================================================
try:
    genai.configure(api_key=GEMINI_KEY_1)
    agent_model = genai.GenerativeModel('gemini-1.5-flash')
    print("✅ Gemini configured!")
except Exception as e:
    print(f"⚠️ Gemini error: {e}")
    agent_model = None

AGENT_PROMPT = """
You are the AI Manager for MCQ Generator.

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

async def handle_telegram_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    user_name = update.effective_user.first_name
    print(f"💬 {user_name}: {user_text}")
    
    try:
        if agent_model is None:
            await update.message.reply_text("❌ AI not configured")
            return
            
        response = agent_model.generate_content(
            f"{AGENT_PROMPT}\n\nUser: {user_text}",
            generation_config={"response_mime_type": "application/json"}
        )
        
        intent = json.loads(response.text)
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
                reply = f"✅ Reset to page {page_num}\n\n{reply}"
            except:
                reply = f"❌ Invalid page number\n\n{reply}"
            
        elif action == "status_report":
            pages_completed = len(config.get("pages_completed", []))
            reply = (
                f"📊 **Status**\n\n"
                f"Status: {config['worker_status'].upper()}\n"
                f"Current Page: {config['current_page']}\n"
                f"Pages Done: {pages_completed}\n"
                f"MCQs: {config['total_questions_generated']}\n"
                f"PDF: {'✅' if config['pdf_drive_link'] else '❌'}\n"
                f"Sheet: {'✅' if config['sheet_url'] else '❌'}"
            )
        
        await update.message.reply_text(reply)
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(f"❌ Telegram Error: {e}")
        await update.message.reply_text(error_msg)

# ============================================================
# 10. FASTAPI APP
# ============================================================
app = FastAPI()

# Initialize Telegram App
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
    return {
        "service": "MCQ Generator",
        "status": config["worker_status"],
        "page": config["current_page"],
        "mcqs": config["total_questions_generated"]
    }

# ============================================================
# 11. MAIN ENTRY
# ============================================================
if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 MCQ Generator v2.0")
    print("=" * 60)
    print(f"📊 MongoDB: ✅ Connected")
    print(f"🤖 Telegram: {'✅' if TELEGRAM_BOT_TOKEN else '❌'}")
    print(f"📚 AI Providers: {sum(1 for k in [GROQ_KEY, MISTRAL_KEY] if k)}")
    print("=" * 60)
    
    # Start worker
    worker = threading.Thread(target=background_worker_process, daemon=True)
    worker.start()
    print("✅ Worker started")
    
    # Start API
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
