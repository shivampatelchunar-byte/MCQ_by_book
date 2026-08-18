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
import fitz  # PyMuPDF
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
# Database & Control
MONGO_URI = os.getenv("MONGO_URI")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GCP_SERVICE_ACCOUNT_JSON = os.getenv("GCP_SERVICE_ACCOUNT_JSON")

# AI API Keys
GEMINI_KEY_1 = os.getenv("GEMINI_API_KEY_1")
GEMINI_KEY_2 = os.getenv("GEMINI_API_KEY_2")
GROQ_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
MISTRAL_KEY = os.getenv("MISTRAL_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
CEREBRAS_KEY = os.getenv("CEREBRAS_API_KEY")
SAMBANOVA_KEY = os.getenv("SAMBANOVA_API_KEY")

# ==========================================
# 2. MONGODB DATABASE SETUP (The Brain)
# ==========================================
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["mcq_agent_db"]
config_col = db["system_config"]

# Initialize default configuration if it doesn't exist
if not config_col.find_one({"_id": "master_config"}):
    config_col.insert_one({
        "_id": "master_config",
        "sheet_url": "",
        "pdf_drive_link": "",
        "worker_status": "paused",
        "current_page": 1,
        "total_questions_generated": 0,
        "system_prompt": "Generate 5 to 10 exam-oriented MCQs from the given text. 60% direct, 40% tricky. Output strictly in English. Ensure no consecutive duplicate correct answers."
    })

# ==========================================
# 3. GOOGLE SHEETS & DRIVE TOOLS
# ==========================================
def get_gspread_client():
    creds_dict = json.loads(GCP_SERVICE_ACCOUNT_JSON)
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)

def download_pdf_from_drive(drive_link: str, output_path: str = "/tmp/current_book.pdf"):
    """Downloads PDF from Google Drive to Render's temporary storage."""
    file_id = re.search(r"/d/([a-zA-Z0-9_-]+)", drive_link)
    if not file_id:
        raise ValueError("Invalid Google Drive Link format.")
    
    download_url = f"https://drive.google.com/uc?id={file_id.group(1)}"
    gdown.download(download_url, output_path, quiet=False)
    return output_path

# ==========================================
# 4. PYDANTIC SCHEMAS (Strict Formatting)
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
# Configured based on priority, reliability, and speed
TIERS = [
    {"name": "Groq", "base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile", "key": GROQ_KEY},
    {"name": "Cerebras", "base_url": "https://api.cerebras.ai/v1", "model": "llama-3.3-70b", "key": CEREBRAS_KEY},
    {"name": "Mistral", "base_url": "https://api.mistral.ai/v1", "model": "mistral-small-latest", "key": MISTRAL_KEY},
    {"name": "GitHub-Azure", "base_url": "https://models.inference.ai.azure.com", "model": "gpt-4o", "key": GITHUB_TOKEN},
    {"name": "SambaNova", "base_url": "https://api.sambanova.ai/v1", "model": "Meta-Llama-3.1-70B-Instruct", "key": SAMBANOVA_KEY},
    {"name": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", "model": "openai/gpt-oss-20b:free", "key": OPENROUTER_KEY}
]

def generate_mcqs_with_fallback(page_text: str, custom_prompt: str) -> MCQList:
    full_prompt = f"{custom_prompt}\n\nPAGE TEXT:\n{page_text}\n\nReturn output strictly matching the required JSON schema."
    
    for tier in TIERS:
        if not tier["key"]:
            continue
            
        try:
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
                
            return MCQList(**parsed_data)
            
        except Exception as e:
            print(f"[{tier['name']} Failed]: {e}. Switching in 5 seconds...")
            time.sleep(5)  # 5-second gap between fallback jumps to maintain hygiene
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
    while True:
        try:
            config = config_col.find_one({"_id": "master_config"})
            
            if config["worker_status"] != "running" or not config["sheet_url"] or not config["pdf_drive_link"]:
                time.sleep(10)
                continue

            # Load PDF
            pdf_path = "/tmp/current_book.pdf"
            if not os.path.exists(pdf_path):
                download_pdf_from_drive(config["pdf_drive_link"], pdf_path)
            
            doc = fitz.open(pdf_path)
            current_page = config["current_page"]
            total_pages = len(doc)
            
            if current_page > total_pages:
                config_col.update_one({"_id": "master_config"}, {"$set": {"worker_status": "completed"}})
                continue

            # Process Page
            page_text = doc.load_page(current_page - 1).get_text("text").strip()
            
            if len(page_text) > 100:  # Skip blank pages
                topic = get_topic_for_page(current_page)
                mcq_data = generate_mcqs_with_fallback(page_text, config["system_prompt"])
                
                # Append to Google Sheets
                gc = get_gspread_client()
                sheet = gc.open_by_url(config["sheet_url"]).sheet1
                
                # Fetch last serial number
                existing_records = len(sheet.get_all_values())
                start_serial = existing_records if existing_records > 0 else 1
                
                rows_to_append = []
                for idx, item in enumerate(mcq_data.mcqs):
                    rows_to_append.append([
                        start_serial + idx,           # Column 1: Serial Number
                        current_page,                 # Column 2: Page Number
                        topic,                        # Column 3: Content / Topic
                        item.question,                # Column 4: Question
                        item.option_a,                # Column 5: Option A
                        item.option_b,                # Column 6: Option B
                        item.option_c,                # Column 7: Option C
                        item.option_d,                # Column 8: Option D
                        item.option_e,                # Column 9: Option E
                        item.correct_answer,          # Column 10: Answer
                        item.explanation              # Column 11: Explanation
                    ])
                
                sheet.append_rows(rows_to_append)
                
                # Update State
                config_col.update_one(
                    {"_id": "master_config"}, 
                    {"$inc": {"current_page": 1, "total_questions_generated": len(rows_to_append)}}
                )
            else:
                # If page is empty, just increment page number
                config_col.update_one({"_id": "master_config"}, {"$inc": {"current_page": 1}})

            doc.close()
            
            # Ultra-Hygienic 5-Second Gap Between Pages
            time.sleep(5)

        except Exception as e:
            print(f"Worker Error: {e}")
            traceback.print_exc()
            time.sleep(15)

# ==========================================
# 8. AGENTIC TELEGRAM BOT (LLM Powered)
# ==========================================
genai.configure(api_key=GEMINI_KEY_1)
agent_model = genai.GenerativeModel('gemini-1.5-flash')

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
    user_text = update.message.text
    
    try:
        # LLM Intent Routing
        response = agent_model.generate_content(
            f"{AGENT_PROMPT}\n\nUser Input: {user_text}",
            generation_config={"response_mime_type": "application/json"}
        )
        
        intent = json.loads(response.text)
        action = intent.get("action")
        value = intent.get("extracted_value", "")
        reply = intent.get("reply_message", "Action acknowledged.")
        
        # Execute Database Updates Based on Intent
        if action == "update_sheet":
            config_col.update_one({"_id": "master_config"}, {"$set": {"sheet_url": value}})
        elif action == "update_pdf":
            config_col.update_one({"_id": "master_config"}, {"$set": {"pdf_drive_link": value, "current_page": 1}})
            # Delete old cached PDF
            if os.path.exists("/tmp/current_book.pdf"):
                os.remove("/tmp/current_book.pdf")
        elif action == "update_prompt":
            config_col.update_one({"_id": "master_config"}, {"$set": {"system_prompt": value}})
        elif action == "start_worker":
            config_col.update_one({"_id": "master_config"}, {"$set": {"worker_status": "running"}})
        elif action == "pause_worker":
            config_col.update_one({"_id": "master_config"}, {"$set": {"worker_status": "paused"}})
        elif action == "status_report":
            config = config_col.find_one({"_id": "master_config"})
            reply = (f"📊 **System Status:**\n"
                     f"Status: {config['worker_status'].upper()}\n"
                     f"Current Page: {config['current_page']}\n"
                     f"Total MCQs Generated: {config['total_questions_generated']}\n"
                     f"Active PDF: {'Configured' if config['pdf_drive_link'] else 'Missing'}\n"
                     f"Active Sheet: {'Configured' if config['sheet_url'] else 'Missing'}")
            
        await update.message.reply_text(reply)
        
    except Exception as e:
        await update.message.reply_text(f"Agent Processing Error: {str(e)}. Please try rephrasing your request.")

def run_telegram_bot():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_telegram_message))
    app.run_polling()

# ==========================================
# 9. FASTAPI KEEP-ALIVE (For Render)
# ==========================================
app = FastAPI()

@app.get("/")
def keep_alive():
    config = config_col.find_one({"_id": "master_config"})
    return {
        "service": "AI_MCQ_Agent_Active",
        "status": config["worker_status"],
        "current_page": config["current_page"]
    }

# ==========================================
# 10. SYSTEM ENTRY POINT
# ==========================================
if __name__ == "__main__":
    # 1. Start the Background PDF Worker
    worker_thread = threading.Thread(target=background_worker_process, daemon=True)
    worker_thread.start()
    
    # 2. Start the Autonomous Telegram Agent
    tg_thread = threading.Thread(target=run_telegram_bot, daemon=True)
    tg_thread.start()
    
    # 3. Uvicorn handles the main thread for FastAPI
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
