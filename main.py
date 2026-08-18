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

# Gemini API Keys - Only these two models
GEMINI_KEY_1 = os.getenv("GEMINI_API_KEY_1")
GEMINI_KEY_2 = os.getenv("GEMINI_API_KEY_2")

# ============================================================
# 2. GEMINI MODELS CONFIGURATION - ONLY FLASH-LITE
# ============================================================
GEMINI_MODELS = [
    {
        "name": "Gemini 3.5 Flash-Lite",
        "model": "gemini-1.5-flash",
        "description": "Fast, efficient, cost-optimized for high-volume"
    },
    {
        "name": "Gemini 3.1 Flash-Lite", 
        "model": "gemini-1.5-flash-8b",
        "description": "Cost-efficient, high-volume workloads"
    }
]

# Store all Gemini keys for rotation
GEMINI_KEYS = [key for key in [GEMINI_KEY_1, GEMINI_KEY_2] if key]

def get_gemini_client(key_index=0):
    """Get Gemini client with specific key"""
    if key_index < len(GEMINI_KEYS):
        genai.configure(api_key=GEMINI_KEYS[key_index])
        return genai
    return None

# Default configuration
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
        "current_page": 1,
        "total_questions_generated": 0,
        "last_processed_page": 0,
        "pages_completed": [],
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
# 5. GEMINI VISION OCR - WITH FLASH-LITE MODELS
# ============================================================
def ocr_page_with_gemini_vision(doc, page_index: int) -> str:
    """Extract text from scanned PDF using Gemini Flash-Lite models"""
    
    # Try each Gemini Flash-Lite model with key rotation
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
                    Extract ALL readable text from this scanned page.
                    - Preserve paragraph structure
                    - Ignore headers, footers, and page numbers
                    - Return only the main body text
                    - If this is a spread (two pages), extract text from both
                    - Focus on agricultural content
                    """,
                    {"mime_type": "image/png", "data": img_bytes}
                ])
                
                text = (response.text or "").strip()
                if len(text) > 50:
                    print(f"✅ OCR with {gemini_model['name']} succeeded ({len(text)} chars)")
                    return text
                else:
                    print(f"⚠️ OCR returned only {len(text)} chars, trying next...")
                    
            except Exception as e:
                print(f"⚠️ OCR with {gemini_model['name']} (Key {key_idx+1}) failed: {str(e)[:80]}")
                time.sleep(1)
                continue
    
    print("❌ All OCR attempts failed")
    return ""

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
# 7. GEMINI MCQ GENERATOR - ONLY FLASH-LITE MODELS
# ============================================================
def generate_mcqs_with_gemini(page_text: str, custom_prompt: str) -> MCQList:
    """Generate MCQs using only Gemini Flash-Lite models with fallback"""
    page_text = page_text[:12000]
    
    full_prompt = f"""
{custom_prompt}

PAGE TEXT:
{page_text}

IMPORTANT INSTRUCTIONS:
1. Generate 5-10 MCQs based SOLELY on the text above
2. Questions must be exam-oriented and test understanding
3. Options must be relevant and plausible
4. Return ONLY valid JSON with 'mcqs' array

Return exactly this JSON format:
{{"mcqs": [{{"question": "...", "option_a": "...", "option_b": "...", "option_c": "...", "option_d": "...", "option_e": "None of these", "correct_answer": "A", "explanation": "..."}}]}}
"""
    
    # Try each Gemini Flash-Lite model with key rotation
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
                
                # Clean response
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
                
            except json.JSONDecodeError as e:
                print(f"❌ {gemini_model['name']} JSON parse error: {e}")
                # Try to extract JSON from response
                try:
                    json_match = re.search(r'\{.*\}', raw_data, re.DOTALL)
                    if json_match:
                        parsed_data = json.loads(json_match.group())
                        if "mcqs" in parsed_data:
                            mcq_list = MCQList(**parsed_data)
                            print(f"✅ {gemini_model['name']} succeeded (extracted JSON)!")
                            return mcq_list
                except:
                    pass
                continue
                
            except Exception as e:
                print(f"❌ {gemini_model['name']} (Key {key_idx+1}) failed: {str(e)[:100]}")
                time.sleep(2)
                continue
    
    raise RuntimeError("All Gemini Flash-Lite models failed")

# ============================================================
# 8. TELEGRAM BOT - WITH FLASH-LITE MODELS
# ============================================================
AGENT_PROMPT = """
You are the AI Manager for MCQ Generator using Gemini Flash-Lite models.

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
    """Get agent response using Gemini Flash-Lite models with fallback"""
    
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
                print(f"⚠️ Agent {gemini_model['name']} (Key {key_idx+1}) failed: {str(e)[:80]}")
                continue
    
    # Fallback if all models fail
    return {
        "action": "general_chat",
        "extracted_value": "",
        "reply_message": "I'm using Gemini Flash-Lite. How can I help with MCQ generation?"
    }

async def handle_telegram_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    user_name = update.effective_user.first_name
    print(f"💬 {user_name}: {user_text}")
    
    try:
        # Get intent using Gemini Flash-Lite
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
                reply = f"✅ Reset to page {page_num}\n\n{reply}"
            except:
                reply = f"❌ Invalid page number\n\n{reply}"
            
        elif action == "status_report":
            pages_completed = len(config.get("pages_completed", []))
            reply = (
                f"📊 **System Status**\n\n"
                f"Status: {config['worker_status'].upper()}\n"
                f"Current Page: {config['current_page']}\n"
                f"Pages Done: {pages_completed}\n"
                f"MCQs Generated: {config['total_questions_generated']}\n"
                f"PDF: {'✅' if config['pdf_drive_link'] else '❌'}\n"
                f"Sheet: {'✅' if config['sheet_url'] else '❌'}\n\n"
                f"🤖 Models: Gemini 3.5 Flash-Lite, Gemini 3.1 Flash-Lite"
            )
        else:
            # General chat - use Gemini Flash-Lite to respond
            try:
                model = genai.GenerativeModel(GEMINI_MODELS[0]['model'])
                response = model.generate_content(
                    f"""You are a helpful assistant for MCQ Generator.
                    Keep responses short and friendly.
                    User: {user_text}"""
                )
                reply = response.text
            except:
                reply = "I'm here to help! Try: Status?, Start, Pause, or configure PDF/Sheet"
        
        await update.message.reply_text(reply)
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(f"❌ Telegram Error: {e}")
        traceback.print_exc()
        await update.message.reply_text(error_msg)

# ============================================================
# 9. TOPIC MAPPING
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
# 10. BACKGROUND WORKER
# ============================================================
def background_worker_process():
    print("🚀 Background Worker Started!")
    print(f"📚 Using models: {', '.join([m['name'] for m in GEMINI_MODELS])}")
    
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
            current_page = config["current_page"]
            total_pages = len(doc)
            
            print(f"📖 Processing Page {current_page}/{total_pages}")
            
            if current_page > total_pages:
                print("✅ All pages processed!")
                config_col.update_one({"_id": "master_config"}, {"$set": {"worker_status": "completed"}})
                doc.close()
                time.sleep(60)
                continue

            # Extract text
            page = doc.load_page(current_page - 1)
            page_text = page.get_text("text").strip()
            
            # If text extraction failed, use OCR with Flash-Lite
            if len(page_text) < 50:
                print(f"🔍 Using Gemini Vision OCR (Flash-Lite) for page {current_page}...")
                page_text = ocr_page_with_gemini_vision(doc, current_page - 1)

            if len(page_text) < 50:
                print(f"⏭️ Page {current_page} empty, skipping...")
                config_col.update_one({"_id": "master_config"}, {"$inc": {"current_page": 1}})
                doc.close()
                time.sleep(2)
                continue

            # Generate MCQs with Gemini Flash-Lite
            topic = get_topic_for_page(current_page)
            print(f"📚 Topic: {topic} | Text: {len(page_text)} chars")
            
            print("🤖 Generating MCQs with Gemini Flash-Lite...")
            mcq_data = generate_mcqs_with_gemini(page_text, config["system_prompt"])
            print(f"✅ Generated {len(mcq_data.mcqs)} MCQs")

            # Save to Google Sheet
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

            # Update progress
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
# 11. FASTAPI APP
# ============================================================
app = FastAPI()

# Initialize Telegram Bot
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
    else:
        print("⚠️ No EXTERNAL_URL for webhook")

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
        "service": "MCQ Generator - Gemini Flash-Lite",
        "models": [m["name"] for m in GEMINI_MODELS],
        "status": config["worker_status"],
        "page": config["current_page"],
        "mcqs": config["total_questions_generated"]
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "mongo": "connected",
        "gemini_keys": len(GEMINI_KEYS),
        "models": [m["name"] for m in GEMINI_MODELS]
    }

# ============================================================
# 12. MAIN ENTRY
# ============================================================
if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 MCQ Generator - Gemini Flash-Lite Edition")
    print("=" * 60)
    print(f"📊 MongoDB: ✅ Connected")
    print(f"🔑 Gemini Keys: {len(GEMINI_KEYS)} configured")
    print(f"📚 Models:")
    for m in GEMINI_MODELS:
        print(f"   - {m['name']} ({m['description']})")
    print("=" * 60)
    
    # Start worker
    worker = threading.Thread(target=background_worker_process, daemon=True)
    worker.start()
    print("✅ Worker started")
    
    # Start API
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
