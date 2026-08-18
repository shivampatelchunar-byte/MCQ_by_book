# 📚 AI MCQ Generator System

## 🚀 Overview
Autonomous system that reads PDF textbooks and generates exam-oriented MCQs using multiple AI models with fallback.

## ✨ Features
- 🤖 AI-powered MCQ generation with 6-provider fallback
- 📖 PDF processing from Google Drive
- 📊 Google Sheets integration for output
- 💬 Telegram bot for natural language control
- 🔄 Self-recovering with MongoDB state
- 🎯 Agricultural exam focused

## 🛠️ Setup

### Environment Variables
```env
MONGO_URI=mongodb+srv://...
TELEGRAM_BOT_TOKEN=...
GCP_SERVICE_ACCOUNT_JSON={...}
GEMINI_API_KEY_1=...
GROQ_API_KEY=...
MISTRAL_API_KEY=...
