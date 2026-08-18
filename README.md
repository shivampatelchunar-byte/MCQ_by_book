# 📚 AI MCQ Generator System v2.0

## 🎯 **What This System Does**

1. **Reads your SCANNED PDF** from Google Drive using Gemini Vision OCR
2. **Extracts text** from each page (even if it's an image-based scan)
3. **Generates MCQs** using multiple AI providers with fallback
4. **Saves to Google Sheets** with proper tracking
5. **Resume capability** - if you stop at page 50, it starts from page 51

## 🚀 **Key Features**

- ✅ **Scanned PDF Support** - Uses Gemini Vision for OCR
- ✅ **Multi-Provider AI** - 6 different providers with fallback
- ✅ **Resume Processing** - MongoDB tracks progress
- ✅ **Telegram Control** - Natural language commands
- ✅ **Error Recovery** - Auto-retry and logging

## 📱 **Telegram Commands**

| Command | What It Does |
|---------|--------------|
| `Status?` | Shows current progress |
| `Start generating` | Starts the worker |
| `Pause system` | Pauses the worker |
| `Set PDF to [link]` | Updates PDF source |
| `Set sheet to [link]` | Updates output sheet |
| `Reset to page [number]` | Resets processing to specific page |

## 🔧 **How Resume Works**

1. MongoDB stores `current_page`, `pages_completed`
2. If system restarts, it continues from last saved page
3. You can manually reset to any page via Telegram
4. No duplicate MCQs - tracking prevents re-processing

## 📊 **Google Sheet Format**

| Column | Content |
|--------|---------|
| 1 | Serial Number |
| 2 | Page Number |
| 3 | Topic |
| 4 | Question |
| 5-9 | Options A-E |
| 10 | Correct Answer |
| 11 | Explanation |
| 12 | Timestamp |
