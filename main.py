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
# 1. ENVIRONMENT VARIABLES & SECRETS
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
# 2. MONGODB DATABASE SETUP - STATE MANAGEMENT
# ============================================================
print("📡 Connecting to MongoDB...")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["mcq_agent_db"]
config_col = db["system_config"]

# Initialize default configuration with proper tracking
if not config_col.find_one({"_id": "master_config"}):
    print("🆕 Creating initial configuration...")
    config_col.insert_one({
        "_id": "master_config",
        "sheet_url": "",
        "pdf_drive_link": "",
        "worker_status": "paused",  # Start paused, user starts via Telegram
        "current_page": 1,
        "total_questions_generated": 0,
        "last_processed_page": 0,
        "pages_completed": [],
        "failed_pages": [],
        "system_prompt": """
            Generate 5 to 10 exam-oriented MCQs from the given text.
            Rules:
            - 60% direct questions, 40% tricky/application-based
            - All options must be plausible
            - Correct answer must be strictly A, B, C, D, or E
            - Explanation must be 1-3 lines factual
            - Ensure no consecutive duplicate correct answers
            - Focus on agricultural concepts
        """,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    })
    print("✅ Configuration created!")
else:
    print("✅ Configuration loaded!")

# ============================================================
# 3. GOOGLE SHEETS & DRIVE TOOLS
# ============================================================
def get_gspread_client():
    """Get authenticated Google Sheets client"""
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
        # Extract file ID from various Drive link formats
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
            raise ValueError("Invalid Google Drive Link format. Please provide a valid link.")
        
        download_url = f"https://drive.google.com/uc?id={file_id}"
        print(f"📥 Downloading PDF from: {download_url}")
        
        # Use gdown with proper headers
        gdown.download(download_url, output_path, quiet=False)
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✅ PDF downloaded successfully: {file_size/1024/1024:.2f} MB")
            return output_path
        else:
            raise Exception("PDF download failed - file not found")
            
    except Exception as e:
        print(f"❌ PDF download error: {e}")
        raise

# ============================================================
# 4. GEMINI VISION OCR - SCANNED PDF PROCESSING
# ============================================================
def ocr_page_with_gemini_vision(doc, page_index: int) -> dict:
    """
    Renders a PDF page as an image and uses Gemini Vision to extract text.
    This is CRITICAL for scanned/image-based PDFs.
    
    Returns: {"text": str, "page_numbers": list[int]}
    """
    try:
        page = doc.load_page(page_index)
        # Higher DPI = better OCR accuracy
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        
        # Use Gemini Vision for OCR
        vision_model = genai.GenerativeModel('gemini-1.5-flash')
        
        response = vision_model.generate_content([
            """
            IMPORTANT: This is a scanned page from an agriculture textbook.
            
            TASK:
            1. Extract ALL readable text from this image
            2. Preserve paragraph structure and formatting
            3. Ignore scanner watermarks, page numbers, headers, and footers
            4. If the image is a spread (two pages side by side), extract text from both
            
            OUTPUT FORMAT:
            First, identify if you see any printed page number in the header/footer.
            Then output ALL the body text.
            
            IMPORTANT: Return ONLY the extracted text. No explanations or extra commentary.
            """,
            {"mime_type": "image/png", "data": img_bytes}
        ])
        
        raw_text = (response.text or "").strip()
        
        # Try to extract page numbers from the text if present
        page_numbers = []
        page_num_matches = re.findall(r'\b(\d{1,3})\b', raw_text[:200])  # Check first 200 chars
        if page_num_matches:
            # Filter out common non-page numbers
            for num in page_num_matches:
                num_int = int(num)
                if 1 <= num_int <= 1000:
                    page_numbers.append(num_int)
            page_numbers = list(set(page_numbers))  # Remove duplicates
        
        return {"text": raw_text, "page_numbers": page_numbers}
        
    except Exception as e:
        print(f"⚠️ Vision OCR failed for PDF page {page_index + 1}: {e}")
        return {"text": "", "page_numbers": []}

# ============================================================
# 5. PYDANTIC SCHEMAS (STRICT VALIDATION)
# ============================================================
class SingleMCQ(BaseModel):
    question: str = Field(description="1-2 lines concise exam-oriented question")
    option_a: str = Field(description="Option A")
    option_b: str = Field(description="Option B")
    option_c: str = Field(description="Option C")
    option_d: str = Field(description="Option D")
    option_e: str = Field(default="None of these", description="Always 'None of these'")
    correct_answer: str = Field(description="Strictly A, B, C, D, or E")
    explanation: str = Field(description="1-3 lines factual explanation")

class MCQList(BaseModel):
    mcqs: List[SingleMCQ]

# ============================================================
# 6. MULTI-TIER AI FALLBACK ENGINE
# ============================================================
TIERS = [
    {"name": "Groq", "base_url": "https://api.groq.com/openai/v1", 
     "model": "llama-3.3-70b-versatile", "key": GROQ_KEY},
    {"name": "Cerebras", "base_url": "https://api.cerebras.ai/v1", 
     "model": "llama-3.3-70b", "key": CEREBRAS_KEY},
    {"name": "Mistral", "base_url": "https://api.mistral.ai/v1", 
     "model": "mistral-small-latest", "key": MISTRAL_KEY},
    {"name": "GitHub-Azure", "base_url": "https://models.inference.ai.azure.com", 
     "model": "gpt-4o", "key": GITHUB_TOKEN},
    {"name": "SambaNova", "base_url": "https://api.sambanova.ai/v1", 
     "model": "Meta-Llama-3.1-70B-Instruct", "key": SAMBANOVA_KEY},
    {"name": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", 
     "model": "openai/gpt-oss-20b:free", "key": OPENROUTER_KEY}
]

def generate_mcqs_with_fallback(page_text: str, custom_prompt: str) -> MCQList:
    """Generate MCQs with multi-tier fallback - focuses on agricultural content"""
    
    # Clean and truncate text if too long
    page_text = page_text[:15000]  # Keep under token limits
    
    full_prompt = f"""
{custom_prompt}

PAGE TEXT (from agriculture textbook):
{page_text}

IMPORTANT INSTRUCTIONS:
1. Generate 5-10 MCQs based SOLELY on the text provided
2. Questions must be exam-oriented and test understanding
3. Options must be relevant and plausible
4. Return ONLY valid JSON with 'mcqs' array

Return exactly this JSON format:
{{"mcqs": [{{"question": "...", "option_a": "...", "option_b": "...", "option_c": "...", "option_d": "...", "option_e": "None of these", "correct_answer": "A", "explanation": "..."}}]}}
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
            
            # Clean response if needed
            raw_data = re.sub(r'^```json\s*', '', raw_data)
            raw_data = re.sub(r'\s*```$', '', raw_data)
            
            parsed_data = json.loads(raw_data)
            
            if "mcqs" not in parsed_data and isinstance(parsed_data, list):
                parsed_data = {"mcqs": parsed_data}
            elif "mcqs" not in parsed_data:
                # Try to find any array in the response
                for key, value in parsed_data.items():
                    if isinstance(value, list) and len(value) > 0:
                        parsed_data = {"mcqs": value}
                        break
            
            mcq_list = MCQList(**parsed_data)
            print(f"✅ {tier['name']} succeeded! Generated {len(mcq_list.mcqs)} MCQs")
            return mcq_list
            
        except Exception as e:
            print(f"❌ [{tier['name']} Failed]: {str(e)[:100]}")
            time.sleep(3)
            continue
            
    raise RuntimeError("Critical Failure: All AI API tiers exhausted.")

# ============================================================
# 7. DYNAMIC TOPIC MAPPING LOGIC
# ============================================================
def get_topic_for_page(page_num: int) -> str:
    """Maps the page number to the exact syllabus topic."""
    if page_num <= 27:
        return "General Agriculture"
    elif page_num <= 214:
        return "Agronomy"
    elif page_num <= 318:
        return "Soil Science"
    elif page_num <= 338:
        return "Agrometeorology"
    elif page_num <= 407:
        return "Animal Husbandry and Dairy Science"
    elif page_num <= 466:
        return "Agricultural Extension"
    elif page_num <= 540:
        return "Agricultural Economics"
    elif page_num <= 571:
        return "Agricultural Statistics"
    elif page_num >= 572:
        return "Agricultural Engineering"
    else:
        return "Preliminary / Index"

# ============================================================
# 8. BACKGROUND WORKER ENGINE - MAIN PROCESSING LOGIC
# ============================================================
def background_worker_process():
    """Background worker that processes PDF pages one by one"""
    print("🚀 Background Worker Started!")
    
    while True:
        try:
            # Get current configuration
            config = config_col.find_one({"_id": "master_config"})
            
            # Check if worker should be running
            if config["worker_status"] != "running":
                time.sleep(10)
                continue
                
            if not config["sheet_url"]:
                print("⚠️ No Google Sheet configured - waiting...")
                time.sleep(10)
                continue
                
            if not config["pdf_drive_link"]:
                print("⚠️ No PDF Drive link configured - waiting...")
                time.sleep(10)
                continue

            # Load PDF
            pdf_path = "/tmp/current_book.pdf"
            if not os.path.exists(pdf_path):
                print("📥 Downloading PDF from Drive...")
                download_pdf_from_drive(config["pdf_drive_link"], pdf_path)
            
            doc = fitz.open(pdf_path)
            current_page = config["current_page"]
            total_pages = len(doc)
            
            print(f"📖 Processing Page {current_page}/{total_pages}")
            
            # Check if all pages are processed
            if current_page > total_pages:
                print("✅ All pages processed!")
                config_col.update_one(
                    {"_id": "master_config"}, 
                    {"$set": {"worker_status": "completed"}}
                )
                doc.close()
                time.sleep(60)
                continue

            # ============================================================
            # STEP 1: Extract text from page
            # ============================================================
            page_text = doc.load_page(current_page - 1).get_text("text").strip()
            detected_page_numbers = []
            
            # If no text layer, use Gemini Vision OCR
            if len(page_text) < 100:
                print(f"🔍 PDF page {current_page} has no text layer, using Gemini Vision OCR...")
                ocr_result = ocr_page_with_gemini_vision(doc, current_page - 1)
                page_text = ocr_result["text"]
                detected_page_numbers = ocr_result["page_numbers"]
                print(f"📝 OCR extracted {len(page_text)} characters")

            # ============================================================
            # STEP 2: Skip empty pages
            # ============================================================
            if len(page_text) < 100:
                print(f"⏭️ Page {current_page} is empty (only {len(page_text)} chars), skipping...")
                config_col.update_one(
                    {"_id": "master_config"}, 
                    {"$inc": {"current_page": 1}}
                )
                doc.close()
                time.sleep(2)
                continue

            # ============================================================
            # STEP 3: Determine book page number and topic
            # ============================================================
            # Try to find page number from detected numbers or use current
            book_page_num = current_page
            if detected_page_numbers:
                book_page_num = detected_page_numbers[0]
            
            topic = get_topic_for_page(book_page_num)
            print(f"📚 Book Page: {book_page_num} | Topic: {topic} | Text length: {len(page_text)} chars")

            # ============================================================
            # STEP 4: Generate MCQs using AI
            # ============================================================
            print(f"🤖 Generating MCQs for page {book_page_num}...")
            mcq_data = generate_mcqs_with_fallback(page_text, config["system_prompt"])
            print(f"✅ Generated {len(mcq_data.mcqs)} MCQs")

            # ============================================================
            # STEP 5: Append to Google Sheets
            # ============================================================
            gc = get_gspread_client()
            sheet = gc.open_by_url(config["sheet_url"]).sheet1
            
            # Get existing records to maintain serial numbers
            existing_records = len(sheet.get_all_values())
            start_serial = existing_records if existing_records > 0 else 1
            
            rows_to_append = []
            for idx, item in enumerate(mcq_data.mcqs):
                rows_to_append.append([
                    start_serial + idx,           # Serial Number
                    book_page_num,                 # Page Number
                    topic,                        # Topic
                    item.question,                # Question
                    item.option_a,                # Option A
                    item.option_b,                # Option B
                    item.option_c,                # Option C
                    item.option_d,                # Option D
                    item.option_e,                # Option E
                    item.correct_answer,          # Correct Answer
                    item.explanation,             # Explanation
                    datetime.now().isoformat()    # Timestamp
                ])
            
            sheet.append_rows(rows_to_append)
            print(f"📊 Added {len(rows_to_append)} rows to Google Sheet")

            # ============================================================
            # STEP 6: Update State (CRITICAL FOR RESUME CAPABILITY)
            # ============================================================
            pages_completed = config.get("pages_completed", [])
            pages_completed.append(current_page)
            
            config_col.update_one(
                {"_id": "master_config"}, 
                {
                    "$inc": {
                        "current_page": 1, 
                        "total_questions_generated": len(rows_to_append)
                    },
                    "$set": {
                        "last_processed_page": current_page,
                        "pages_completed": pages_completed,
                        "updated_at": datetime.now().isoformat()
                    }
                }
            )

            doc.close()
            
            # ============================================================
            # STEP 7: Wait before next page (rate limiting)
            # ============================================================
            print(f"⏳ Waiting 5 seconds before next page...")
            time.sleep(5)

        except Exception as e:
            print(f"❌ Worker Error: {e}")
            traceback.print_exc()
            
            # Log error to MongoDB for debugging
            try:
                error_log = db["error_logs"]
                error_log.insert_one({
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "config": config_col.find_one({"_id": "master_config"})
                })
            except:
                pass
            
            # Wait longer on error
            time.sleep(30)

# ============================================================
# 9. TELEGRAM BOT - NATURAL LANGUAGE CONTROL
# ============================================================
try:
    genai.configure(api_key=GEMINI_KEY_1)
    agent_model = genai.GenerativeModel('gemini-1.5-flash')
    print("✅ Gemini configured successfully!")
except Exception as e:
    print(f"⚠️ Gemini config warning: {e}")
    agent_model = None

AGENT_PROMPT = """
You are the Autonomous AI Manager for an MCQ Generation System.

Your job is to understand user commands and return JSON actions.

Valid actions and examples:
1. update_sheet: "set sheet to [URL]" or "update google sheet to [URL]"
2. update_pdf: "set pdf to [URL]" or "update book to [URL]"
3. update_prompt: "change prompt to [text]" or "update instructions"
4. start_worker: "start generating", "resume", "begin processing"
5. pause_worker: "pause", "stop", "halt"
6. status_report: "status?", "progress?", "how many done?"
7. reset_page: "reset to page [number]" or "start from page [number]"
8. general_chat: any other conversation

Return ONLY this JSON:
{"action": "action_name", "extracted_value": "value if applicable", "reply_message": "your response"}
"""

async def handle_telegram_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming Telegram messages with natural language understanding"""
    user_text = update.message.text
    user_name = update.effective_user.first_name
    
    print(f"💬 Telegram from {user_name}: {user_text}")
    
    try:
        if agent_model is None:
            await update.message.reply_text("❌ AI Agent not configured. Please check Gemini API key.")
            return
            
        # LLM Intent Routing
        response = agent_model.generate_content(
            f"{AGENT_PROMPT}\n\nUser Input: {user_text}",
            generation_config={"response_mime_type": "application/json"}
        )
        
        intent = json.loads(response.text)
        action = intent.get("action")
        value = intent.get("extracted_value", "")
        reply = intent.get("reply_message", "Action acknowledged.")
        
        print(f"🤖 Intent: {action} | Value: {value[:50] if value else 'None'}")
        
        # Execute actions
        if action == "update_sheet":
            config_col.update_one({"_id": "master_config"}, {"$set": {"sheet_url": value}})
            reply = f"✅ Google Sheet updated successfully!\n\n{reply}"
            
        elif action == "update_pdf":
            config_col.update_one({"_id": "master_config"}, {"$set": {"pdf_drive_link": value}})
            if os.path.exists("/tmp/current_book.pdf"):
                os.remove("/tmp/current_book.pdf")
                reply = f"✅ PDF updated and cache cleared!\n\n{reply}"
            else:
                reply = f"✅ PDF configured!\n\n{reply}"
                
        elif action == "update_prompt":
            config_col.update_one({"_id": "master_config"}, {"$set": {"system_prompt": value}})
            reply = f"✅ System prompt updated!\n\n{reply}"
            
        elif action == "start_worker":
            config_col.update_one({"_id": "master_config"}, {"$set": {"worker_status": "running"}})
            reply = f"🚀 Worker started! Processing from page {config_col.find_one({'_id': 'master_config'})['current_page']}\n\n{reply}"
            
        elif action == "pause_worker":
            config_col.update_one({"_id": "master_config"}, {"$set": {"worker_status": "paused"}})
            reply = f"⏸️ Worker paused.\n\n{reply}"
            
        elif action == "reset_page":
            try:
                page_num = int(value)
                config_col.update_one({"_id": "master_config"}, {"$set": {"current_page": page_num}})
                reply = f"✅ Reset to page {page_num}\n\n{reply}"
            except:
                reply = f"❌ Invalid page number. Please provide a valid number.\n\n{reply}"
            
        elif action == "status_report":
            config = config_col.find_one({"_id": "master_config"})
            pages_completed = len(config.get("pages_completed", []))
            total_mcqs = config.get("total_questions_generated", 0)
            
            reply = (
                f"📊 **System Status**\n\n"
                f"┌─────────────────────────────┐\n"
                f"│ Status: {config['worker_status'].upper()}\n"
                f"│ Current Page: {config['current_page']}\n"
                f"│ Pages Completed: {pages_completed}\n"
                f"│ Total MCQs: {total_mcqs}\n"
                f"│ PDF: {'✅' if config['pdf_drive_link'] else '❌'}\n"
                f"│ Sheet: {'✅' if config['sheet_url'] else '❌'}\n"
                f"└─────────────────────────────┘"
            )
        
        await update.message.reply_text(reply)
        
    except Exception as e:
        error_msg = f"❌ Agent Error: {str(e)}"
        print(f"❌ Telegram Error: {e}")
        traceback.print_exc()
        await update.message.reply_text(error_msg)

# ============================================================
# 10. TELEGRAM WEBHOOK SETUP (NO POLLING)
# ============================================================
telegram_app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_telegram_message))

WEBHOOK_PATH = f"/telegram-webhook/{TELEGRAM_BOT_TOKEN}"
EXTERNAL_URL = (os.getenv("WEBHOOK_URL") or os.getenv("RENDER_EXTERNAL_URL", "")).rstrip("/")

# ============================================================
# 11. FASTAPI APP
# ============================================================
app = FastAPI(title="AI MCQ Generator", version="2.0")

@app.on_event("startup")
async def on_startup():
    await telegram_app.initialize()
    await telegram_app.start()
    if EXTERNAL_URL:
        webhook_url = f"{EXTERNAL_URL}{WEBHOOK_PATH}"
        await telegram_app.bot.set_webhook(url=webhook_url, drop_pending_updates=True)
        print(f"✅ Telegram webhook registered: {webhook_url}")
    else:
        print("⚠️ No external URL found for webhook")

@app.on_event("shutdown")
async def on_shutdown():
    try:
        await telegram_app.bot.delete_webhook()
    except Exception:
        pass
    await telegram_app.stop()

@app.post(WEBHOOK_PATH)
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return {"ok": True}

@app.get("/")
def keep_alive():
    config = config_col.find_one({"_id": "master_config"})
    return {
        "service": "AI_MCQ_Generator",
        "version": "2.0",
        "status": config["worker_status"],
        "current_page": config["current_page"],
        "total_questions_generated": config["total_questions_generated"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
def health_check():
    config = config_col.find_one({"_id": "master_config"})
    return {
        "status": "healthy",
        "uptime": time.time() - start_time,
        "mongo": "connected",
        "worker_status": config["worker_status"],
        "current_page": config["current_page"],
        "api_keys_configured": {
            "gemini": bool(GEMINI_KEY_1),
            "groq": bool(GROQ_KEY),
            "mistral": bool(MISTRAL_KEY)
        }
    }

@app.post("/reset/{page_num}")
def reset_to_page(page_num: int):
    """Manually reset processing to a specific page"""
    config_col.update_one(
        {"_id": "master_config"}, 
        {"$set": {"current_page": page_num}}
    )
    return {"message": f"Reset to page {page_num}", "page": page_num}

# ============================================================
# 12. SYSTEM ENTRY POINT
# ============================================================
start_time = time.time()

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("🚀 AI MCQ Generator System v2.0")
    print("=" * 70)
    print(f"📊 MongoDB: {'✅ Connected' if mongo_client else '❌ Failed'}")
    print(f"🤖 Telegram: {'✅ Configured' if TELEGRAM_BOT_TOKEN else '❌ Missing'}")
    print(f"📚 AI Providers Configured: {sum(1 for k in [GROQ_KEY, MISTRAL_KEY, GEMINI_KEY_1] if k)}")
    print(f"📖 PDF: {'✅ Downloaded' if os.path.exists('/tmp/current_book.pdf') else '⏳ Not yet'}")
    print("=" * 70)
    
    # Start Background Worker
    try:
        worker_thread = threading.Thread(target=background_worker_process, daemon=True)
        worker_thread.start()
        print("✅ Background Worker Thread Started")
    except Exception as e:
        print(f"❌ Worker thread error: {e}")
    
    # Start FastAPI
    try:
        port = int(os.environ.get("PORT", 8080))
        print(f"✅ Starting FastAPI on port {port}...")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ FastAPI error: {e}")
        traceback.print_exc()
