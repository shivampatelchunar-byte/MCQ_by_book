import os
import io
import re
import json
import time
import asyncio
import threading
import traceback
from datetime import datetime
from typing import Optional, List
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

# ============================================================
# 1. ENVIRONMENT VARIABLES & SECRETS
# ============================================================
MONGO_URI = os.getenv("MONGO_URI")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GCP_SERVICE_ACCOUNT_JSON = os.getenv("GCP_SERVICE_ACCOUNT_JSON")

# All API Keys mapped for Agentic Power
GEMINI_KEYS = [key for key in [os.getenv("GEMINI_API_KEY_1"), os.getenv("GEMINI_API_KEY_2")] if key]
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

if not config_col.find_one({"_id": "master_config"}):
    config_col.insert_one({
        "_id": "master_config",
        "sheet_url": "",
        "pdf_drive_link": "",
        "worker_status": "paused",
        "current_page": 1,
        "total_questions_generated": 0,
        "system_prompt": "Generate 5 to 10 exam-oriented MCQs from the given text. 60% direct questions, 40% tricky/application-based. OUTPUT MUST BE IN PURE ENGLISH ONLY. Do not use a single word of Hindi. Correct answer must not be consecutive same letters.",
    })

# ============================================================
# 3. GEMINI MODELS CONFIGURATION
# ============================================================
if GEMINI_KEYS:
    genai.configure(api_key=GEMINI_KEYS[0])

GEMINI_MODELS = [
    {
        "name": "Gemini 3.5 Flash-Lite",
        "model": "gemini-3.5-flash-lite",
        "description": "Latest fast model for agentic and generation tasks"
    },
    {
        "name": "Gemini 3.1 Flash-Lite",
        "model": "gemini-2.5-flash-lite", 
        "description": "Cost-efficient, high-volume workloads"
    }
]

# ============================================================
# 4. MULTI-TIER LLM NETWORK
# ============================================================
TIERS = [
    {"name": "Groq", "base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile", "key": GROQ_KEY},
    {"name": "Cerebras", "base_url": "https://api.cerebras.ai/v1", "model": "llama-3.3-70b", "key": CEREBRAS_KEY},
    {"name": "Mistral", "base_url": "https://api.mistral.ai/v1", "model": "mistral-small-latest", "key": MISTRAL_KEY},
    {"name": "GitHub-Azure", "base_url": "https://models.inference.ai.azure.com", "model": "gpt-4o", "key": GITHUB_TOKEN},
    {"name": "SambaNova", "base_url": "https://api.sambanova.ai/v1", "model": "Meta-Llama-3.1-70B-Instruct", "key": SAMBANOVA_KEY},
    {"name": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", "model": "openai/gpt-oss-20b:free", "key": OPENROUTER_KEY}
]

# ============================================================
# 5. GOOGLE SHEETS & DRIVE
# ============================================================
def get_gspread_client():
    creds_dict = json.loads(GCP_SERVICE_ACCOUNT_JSON)
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)

def download_pdf_from_drive(drive_link: str, output_path: str = "/tmp/current_book.pdf"):
    file_id_match = re.search(r"/d/([a-zA-Z0-9_-]+)", drive_link)
    if not file_id_match:
        raise ValueError("Invalid Google Drive Link format")
    download_url = f"https://drive.google.com/uc?id={file_id_match.group(1)}"
    gdown.download(download_url, output_path, quiet=False)
    return output_path

# ============================================================
# 6. PYDANTIC SCHEMAS (PURE ENGLISH ENFORCED)
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
# 7. MCQ WORKER FALLBACK ENGINE
# ============================================================
def generate_mcqs_with_fallback(page_text: str, custom_prompt: str) -> MCQList:
    full_prompt = f"{custom_prompt}\n\nPAGE TEXT:\n{page_text[:10000]}\n\nReturn ONLY valid JSON with 'mcqs' array. Every single word MUST be in PURE ENGLISH."
    
    # Try Gemini First
    for gem_model in GEMINI_MODELS:
        for key in GEMINI_KEYS:
            try:
                genai.configure(api_key=key)
                model = genai.GenerativeModel(gem_model['model'])
                response = model.generate_content(full_prompt, generation_config={"response_mime_type": "application/json"})
                return MCQList(**json.loads(response.text))
            except Exception:
                continue

    # Try Open-Source Tiers
    for tier in TIERS:
        if not tier["key"]: continue
        try:
            client = OpenAI(api_key=tier["key"], base_url=tier["base_url"])
            response = client.chat.completions.create(
                model=tier["model"],
                messages=[{"role": "user", "content": full_prompt}],
                response_format={"type": "json_object"},
                timeout=45
            )
            return MCQList(**json.loads(response.choices[0].message.content))
        except Exception:
            continue
            
    raise RuntimeError("All AI providers failed to generate MCQs.")

# ============================================================
# 8. AGENTIC TELEGRAM BOT
# ============================================================
def ask_universal_llm(prompt_text: str) -> str:
    # Try Gemini First
    for key in GEMINI_KEYS:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel(GEMINI_MODELS[0]['model'])
            return model.generate_content(prompt_text).text
        except Exception:
            continue
            
    # Fallback to OpenRouter / Groq / Mistral
    for tier in TIERS:
        if not tier["key"]: continue
        try:
            client = OpenAI(api_key=tier["key"], base_url=tier["base_url"])
            resp = client.chat.completions.create(
                model=tier["model"],
                messages=[{"role": "system", "content": "You are a helpful AI Assistant managing an automated MCQ generation system."},
                          {"role": "user", "content": prompt_text}],
                timeout=20
            )
            return resp.choices[0].message.content
        except Exception:
            continue
            
    return "I am currently facing network issues, but I have registered your command."

async def handle_telegram_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    config = config_col.find_one({"_id": "master_config"})
    
    agent_prompt = f"""
    You are an autonomous AI Manager. The user is talking to you via Telegram.
    Analyze the user's message: "{user_text}"
    
    If the user gives a Google Sheet link, format your reply: [ACTION: UPDATE_SHEET: <link>] + a natural response.
    If the user gives a Google Drive link, format your reply: [ACTION: UPDATE_PDF: <link>] + a natural response.
    If the user says start/resume, format your reply: [ACTION: START] + a natural response.
    If the user says stop/pause, format your reply: [ACTION: PAUSE] + a natural response.
    If the user asks for status, format your reply: [ACTION: STATUS] + a natural response.
    Otherwise, just chat naturally like an intelligent assistant.
    
    Current System Status:
    - Worker is: {config['worker_status']}
    - Current Page: {config['current_page']}
    - Total MCQs: {config['total_questions_generated']}
    """
    
    ai_response = ask_universal_llm(agent_prompt)
    reply_text = ai_response
    
    if "[ACTION: UPDATE_SHEET:" in ai_response:
        link = re.search(r'\[ACTION: UPDATE_SHEET: (.*?)\]', ai_response).group(1)
        config_col.update_one({"_id": "master_config"}, {"$set": {"sheet_url": link.strip()}})
        reply_text = ai_response.replace(f"[ACTION: UPDATE_SHEET: {link}]", "")
        
    elif "[ACTION: UPDATE_PDF:" in ai_response:
        link = re.search(r'\[ACTION: UPDATE_PDF: (.*?)\]', ai_response).group(1)
        config_col.update_one({"_id": "master_config"}, {"$set": {"pdf_drive_link": link.strip(), "current_page": 1}})
        if os.path.exists("/tmp/current_book.pdf"): os.remove("/tmp/current_book.pdf")
        reply_text = ai_response.replace(f"[ACTION: UPDATE_PDF: {link}]", "")
        
    elif "[ACTION: START]" in ai_response:
        config_col.update_one({"_id": "master_config"}, {"$set": {"worker_status": "running"}})
        reply_text = ai_response.replace("[ACTION: START]", "")
        
    elif "[ACTION: PAUSE]" in ai_response:
        config_col.update_one({"_id": "master_config"}, {"$set": {"worker_status": "paused"}})
        reply_text = ai_response.replace("[ACTION: PAUSE]", "")
        
    elif "[ACTION: STATUS]" in ai_response:
        reply_text = ai_response.replace("[ACTION: STATUS]", "")

    await update.message.reply_text(reply_text.strip())

def run_telegram_bot_thread():
    """CRASH-PROOF TELEGRAM BOT THREAD"""
    # 1. Clear Old Webhooks (Fixes 404 Error)
    try:
        requests.get(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/deleteWebhook?drop_pending_updates=true")
        print("🧹 Cleared old Telegram Webhooks")
    except Exception as e:
        print(f"Webhook Clear Error: {e}")

    # 2. Setup Asyncio Loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_telegram_message))
    
    # 3. Start Polling with stop_signals=() (Fixes 'set_wakeup_fd' Error)
    print("🤖 Telegram Agent Polling Started...")
    app.run_polling(stop_signals=())

# ============================================================
# 9. BACKGROUND PDF WORKER
# ============================================================
def get_topic_for_page(page_num: int) -> str:
    if page_num <= 27: return "General Agriculture"
    elif page_num <= 214: return "Agronomy"
    elif page_num <= 318: return "Soil Science"
    elif page_num <= 338: return "Agrometeorology"
    elif page_num <= 407: return "Animal Husbandry and Dairy Science"
    elif page_num <= 466: return "Agricultural Extension"
    elif page_num <= 540: return "Agricultural Economics"
    elif page_num <= 571: return "Agricultural Statistics"
    else: return "Agricultural Engineering"

def background_worker_process():
    while True:
        try:
            config = config_col.find_one({"_id": "master_config"})
            
            if config["worker_status"] != "running" or not config["pdf_drive_link"] or not config["sheet_url"]:
                time.sleep(10)
                continue

            pdf_path = "/tmp/current_book.pdf"
            if not os.path.exists(pdf_path):
                download_pdf_from_drive(config["pdf_drive_link"], pdf_path)
            
            doc = fitz.open(pdf_path)
            current_page = config["current_page"]
            
            if current_page > len(doc):
                config_col.update_one({"_id": "master_config"}, {"$set": {"worker_status": "completed"}})
                continue

            page_text = doc.load_page(current_page - 1).get_text("text").strip()
            
            if len(page_text) > 50:
                topic = get_topic_for_page(current_page)
                mcq_data = generate_mcqs_with_fallback(page_text, config["system_prompt"])
                
                gc = get_gspread_client()
                sheet = gc.open_by_url(config["sheet_url"]).sheet1
                start_serial = max(1, len(sheet.get_all_values()))
                
                rows = []
                for idx, item in enumerate(mcq_data.mcqs):
                    rows.append([
                        start_serial + idx, current_page, topic,
                        item.question, item.option_a, item.option_b, item.option_c,
                        item.option_d, item.option_e, item.correct_answer, item.explanation
                    ])
                sheet.append_rows(rows)
                
                config_col.update_one(
                    {"_id": "master_config"},
                    {"$inc": {"current_page": 1, "total_questions_generated": len(rows)}}
                )
            else:
                config_col.update_one({"_id": "master_config"}, {"$inc": {"current_page": 1}})

            doc.close()
            time.sleep(5)  # 5-second gap for hygiene

        except Exception as e:
            print(f"Worker Error: {e}")
            time.sleep(15)

# ============================================================
# 10. FASTAPI KEEP-ALIVE SERVER (Render)
# ============================================================
app = FastAPI()

@app.get("/")
def home():
    return {"status": "Agentic API and Telegram Worker Running 24/7"}

if __name__ == "__main__":
    import uvicorn
    
    # 1. Start PDF Worker Thread
    threading.Thread(target=background_worker_process, daemon=True).start()
    
    # 2. Start Smart Telegram Agent Thread
    threading.Thread(target=run_telegram_bot_thread, daemon=True).start()
    
    # 3. Start FastAPI Web Server
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
