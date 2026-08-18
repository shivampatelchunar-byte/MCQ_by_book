import os
import io
import re
import json
import time
import asyncio
import threading
import traceback
import gspread
import requests
import gdown
import pymupdf as fitz
from fastapi import FastAPI
from pydantic import BaseModel, Field
from pymongo import MongoClient
from google.oauth2.service_account import Credentials
import google.generativeai as genai
from openai import OpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

# ==========================================
# 1. ENVIRONMENT VARIABLES & SECRETS
# ==========================================
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

# ==========================================
# 2. MONGODB DATABASE SETUP
# ==========================================
print("📡 Connecting to MongoDB...")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["mcq_agent_db"]
config_col = db["system_config"]

# Initialize default configuration
if not config_col.find_one({"_id": "master_config"}):
    print("🆕 Creating initial configuration...")
    config_col.insert_one({
        "_id": "master_config",
        "sheet_url": "",
        "pdf_drive_link": "",
        "worker_status": "running",
        "current_page": 1,
        "total_questions_generated": 0,
        "system_prompt": "Generate 5 to 10 exam-oriented MCQs from the given text. 60% direct, 40% tricky. Output strictly in English. Ensure no consecutive duplicate correct answers."
    })
    print("✅ Configuration created!")
else:
    print("✅ Configuration loaded!")

# ==========================================
# 3. GOOGLE SHEETS & DRIVE TOOLS
# ==========================================
def get_gspread_client():
    """Get authenticated Google Sheets client"""
    try:
        creds_dict = json.loads(GCP_SERVICE_ACCOUNT_JSON)
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        return gspread.authorize(creds)
    except Exception as e:
        print(f"❌ Google Sheets auth error: {e}")
        raise

def download_pdf_from_drive(drive_link: str, output_path: str = "/tmp/current_book.pdf"):
    """Downloads PDF from Google Drive to Render's temporary storage."""
    try:
        file_id = re.search(r"/d/([a-zA-Z0-9_-]+)", drive_link)
        if not file_id:
            raise ValueError("Invalid Google Drive Link format.")
        
        download_url = f"https://drive.google.com/uc?id={file_id.group(1)}"
        print(f"📥 Downloading PDF from: {download_url}")
        gdown.download(download_url, output_path, quiet=False)
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✅ PDF downloaded successfully: {file_size} bytes")
            return output_path
        else:
            raise Exception("PDF download failed - file not found")
            
    except Exception as e:
        print(f"❌ PDF download error: {e}")
        raise

# ==========================================
# 4. PYDANTIC SCHEMAS
# ==========================================
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

# ==========================================
# 5. MULTI-TIER AI FALLBACK ENGINE
# ==========================================
TIERS = [
    {"name": "Groq", "base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile", "key": GROQ_KEY},
    {"name": "Cerebras", "base_url": "https://api.cerebras.ai/v1", "model": "llama-3.3-70b", "key": CEREBRAS_KEY},
    {"name": "Mistral", "base_url": "https://api.mistral.ai/v1", "model": "mistral-small-latest", "key": MISTRAL_KEY},
    {"name": "GitHub-Azure", "base_url": "https://models.inference.ai.azure.com", "model": "gpt-4o", "key": GITHUB_TOKEN},
    {"name": "SambaNova", "base_url": "https://api.sambanova.ai/v1", "model": "Meta-Llama-3.1-70B-Instruct", "key": SAMBANOVA_KEY},
    {"name": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", "model": "openai/gpt-oss-20b:free", "key": OPENROUTER_KEY}
]

def generate_mcqs_with_fallback(page_text: str, custom_prompt: str) -> MCQList:
    """Generate MCQs with multi-tier fallback"""
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
                
            print(f"✅ {tier['name']} succeeded!")
            return MCQList(**parsed_data)
            
        except Exception as e:
            print(f"❌ [{tier['name']} Failed]: {e}")
            time.sleep(5)
            continue
            
    raise RuntimeError("Critical Failure: All AI API tiers exhausted.")

# ==========================================
# 6. DYNAMIC TOPIC MAPPING LOGIC
# ==========================================
def get_topic_for_page(page_num: int) -> str:
    """Maps the page number to the exact syllabus topic."""
    if 1 <= page_num <= 27:
        return "General Agriculture"
    elif 28 <= page_num <= 214:
        return "Agronomy"
    elif 215 <= page_num <= 318:
        return "Soil Science"
    elif 319 <= page_num <= 338:
        return "Agrometeorology"
    elif 339 <= page_num <= 407:
        return "Animal Husbandry and Dairy Science"
    elif 408 <= page_num <= 466:
        return "Agricultural Extension"
    elif 467 <= page_num <= 540:
        return "Agricultural Economics"
    elif 541 <= page_num <= 571:
        return "Agricultural Statistics"
    elif page_num >= 572:
        return "Agricultural Engineering"
    else:
        return "Preliminary / Index"

# ==========================================
# 7. BACKGROUND WORKER ENGINE
# ==========================================
def background_worker_process():
    """Background worker with improved error handling and logging"""
    print("🚀 Background Worker Started!")
    
    while True:
        try:
            config = config_col.find_one({"_id": "master_config"})
            
            if config["worker_status"] != "running" or not config["sheet_url"] or not config["pdf_drive_link"]:
                if config["worker_status"] != "running":
                    time.sleep(10)
                    continue
                elif not config["sheet_url"]:
                    print("⚠️ No Google Sheet configured - waiting...")
                    time.sleep(10)
                    continue
                elif not config["pdf_drive_link"]:
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
            
            if current_page > total_pages:
                print("✅ All pages processed!")
                config_col.update_one({"_id": "master_config"}, {"$set": {"worker_status": "completed"}})
                doc.close()
                time.sleep(60)
                continue

            # Process Page
            page_text = doc.load_page(current_page - 1).get_text("text").strip()
            
            if len(page_text) > 100:
                topic = get_topic_for_page(current_page)
                print(f"📚 Topic: {topic} | Text length: {len(page_text)} chars")
                
                mcq_data = generate_mcqs_with_fallback(page_text, config["system_prompt"])
                print(f"✅ Generated {len(mcq_data.mcqs)} MCQs")
                
                # Append to Google Sheets
                gc = get_gspread_client()
                sheet = gc.open_by_url(config["sheet_url"]).sheet1
                
                existing_records = len(sheet.get_all_values())
                start_serial = existing_records if existing_records > 0 else 1
                
                rows_to_append = []
                for idx, item in enumerate(mcq_data.mcqs):
                    rows_to_append.append([
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
                        item.explanation
                    ])
                
                sheet.append_rows(rows_to_append)
                print(f"📊 Added {len(rows_to_append)} rows to Google Sheet")
                
                # Update State
                config_col.update_one(
                    {"_id": "master_config"}, 
                    {"$inc": {"current_page": 1, "total_questions_generated": len(rows_to_append)}}
                )
            else:
                print(f"⏭️ Page {current_page} is empty, skipping...")
                config_col.update_one({"_id": "master_config"}, {"$inc": {"current_page": 1}})

            doc.close()
            time.sleep(5)

        except Exception as e:
            print(f"❌ Worker Error: {e}")
            traceback.print_exc()
            
            # Log error to MongoDB
            try:
                error_log = db["error_logs"]
                error_log.insert_one({
                    "timestamp": time.time(),
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "config": config_col.find_one({"_id": "master_config"})
                })
            except:
                pass
            
            time.sleep(15)

# ==========================================
# 8. AGENTIC TELEGRAM BOT (FIXED FOR EVENT LOOP)
# ==========================================
try:
    genai.configure(api_key=GEMINI_KEY_1)
    agent_model = genai.GenerativeModel('gemini-1.5-flash')
    print("✅ Gemini configured successfully!")
except Exception as e:
    print(f"⚠️ Gemini config warning: {e}")
    agent_model = None

AGENT_PROMPT = """
You are the Autonomous AI Manager for an MCQ Generation System. 
Analyze the user's natural language input and return a STRICT JSON object representing the action to take.
Valid actions:
1. update_sheet: User provided a Google Sheet link. Extract the full URL.
2. update_pdf: User provided a Google Drive PDF link. Extract the full URL.
3. update_prompt: User wants to change how MCQs are generated. Extract the new instructions.
4. start_worker: User wants to start/resume the system.
5. pause_worker: User wants to stop/pause the system.
6. status_report: User is asking for the current status.
7. general_chat: None of the above, just chat.

Return ONLY this JSON format:
{"action": "action_name", "extracted_value": "value if applicable, else empty string", "reply_message": "A natural language response to the user in English confirming the action."}
"""

async def handle_telegram_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming Telegram messages"""
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
        
        # Execute Database Updates Based on Intent
        if action == "update_sheet":
            config_col.update_one({"_id": "master_config"}, {"$set": {"sheet_url": value}})
            reply = f"✅ Google Sheet updated!\n\n{reply}"
            
        elif action == "update_pdf":
            config_col.update_one({"_id": "master_config"}, {"$set": {"pdf_drive_link": value, "current_page": 1}})
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
            reply = f"🚀 Worker started!\n\n{reply}"
            
        elif action == "pause_worker":
            config_col.update_one({"_id": "master_config"}, {"$set": {"worker_status": "paused"}})
            reply = f"⏸️ Worker paused.\n\n{reply}"
            
        elif action == "status_report":
            config = config_col.find_one({"_id": "master_config"})
            reply = (
                f"📊 **System Status**\n\n"
                f"┌─────────────────────────┐\n"
                f"│ Status: {config['worker_status'].upper()}\n"
                f"│ Page: {config['current_page']}\n"
                f"│ MCQs Generated: {config['total_questions_generated']}\n"
                f"│ PDF: {'✅' if config['pdf_drive_link'] else '❌'}\n"
                f"│ Sheet: {'✅' if config['sheet_url'] else '❌'}\n"
                f"└─────────────────────────┘"
            )
        
        await update.message.reply_text(reply)
        
    except Exception as e:
        error_msg = f"❌ Agent Error: {str(e)}\nPlease try rephrasing your request."
        print(f"❌ Telegram Error: {e}")
        traceback.print_exc()
        await update.message.reply_text(error_msg)

# ✅ FIXED: Create event loop in thread
def run_telegram_bot():
    """Run Telegram bot with proper event loop setup"""
    print("🤖 Starting Telegram Bot...")
    
    try:
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Build and run the bot
        app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_telegram_message))
        
        # Run the bot with the event loop
        # stop_signals=None is REQUIRED because this runs in a background
        # thread — signal handlers can only be installed in the main thread.
        app.run_polling(stop_signals=None)
        
    except Exception as e:
        print(f"❌ Telegram Bot Error: {e}")
        traceback.print_exc()

# ==========================================
# 9. FASTAPI KEEP-ALIVE
# ==========================================
app = FastAPI()

@app.get("/")
def keep_alive():
    """Health check endpoint"""
    try:
        config = config_col.find_one({"_id": "master_config"})
        return {
            "service": "AI_MCQ_Agent_Active",
            "status": config["worker_status"],
            "current_page": config["current_page"],
            "total_questions_generated": config["total_questions_generated"],
            "timestamp": time.time()
        }
    except:
        return {
            "service": "AI_MCQ_Agent_Active",
            "status": "starting",
            "timestamp": time.time()
        }

@app.get("/health")
def health_check():
    """Detailed health check"""
    try:
        config = config_col.find_one({"_id": "master_config"})
        return {
            "status": "healthy",
            "uptime": time.time() - start_time,
            "mongo": "connected",
            "worker_status": config["worker_status"],
            "current_page": config["current_page"],
            "api_keys": {
                "gemini": bool(GEMINI_KEY_1),
                "groq": bool(GROQ_KEY),
                "mistral": bool(MISTRAL_KEY),
                "openrouter": bool(OPENROUTER_KEY)
            }
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# ==========================================
# 10. SYSTEM ENTRY POINT
# ==========================================
start_time = time.time()

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 AI MCQ Generator System Starting...")
    print("=" * 60)
    print(f"📊 MongoDB: {'Connected' if mongo_client else 'Failed'}")
    print(f"🤖 Telegram: {'Configured' if TELEGRAM_BOT_TOKEN else 'Missing'}")
    print(f"📚 AI Providers: {sum(1 for k in [GROQ_KEY, MISTRAL_KEY, GEMINI_KEY_1] if k)} configured")
    print("=" * 60)
    
    # 1. Start Background Worker
    try:
        worker_thread = threading.Thread(target=background_worker_process, daemon=True)
        worker_thread.start()
        print("✅ Background Worker Thread Started")
    except Exception as e:
        print(f"❌ Worker thread error: {e}")
    
    # 2. Start Telegram Bot in background thread
    try:
        bot_thread = threading.Thread(target=run_telegram_bot, daemon=True)
        bot_thread.start()
        print("✅ Telegram Bot Thread Started")
    except Exception as e:
        print(f"❌ Telegram bot thread error: {e}")
    
    # 3. Start FastAPI (Main Thread)
    try:
        port = int(os.environ.get("PORT", 8080))
        print(f"✅ Starting FastAPI on port {port}...")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ FastAPI error: {e}")
        traceback.print_exc()
