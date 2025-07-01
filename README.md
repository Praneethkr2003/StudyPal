# Study Pal - AI-Powered Study Assistant Bot

📚 **Study Pal** is an intelligent Telegram bot that helps students and learners by providing AI-powered study assistance. Upload your PDFs, ask questions, get summaries, and find relevant educational content - all within Telegram!

## ✨ Features

- **PDF Processing**: Upload and process PDF documents (text and images)
- **AI-Powered Q&A**: Ask questions about your uploaded documents
- **Document Summarization**: Get concise summaries of your study materials
- **Key Points Extraction**: Extract main ideas and important points from documents
- **YouTube Integration**: Find relevant educational videos for your queries
- **Multi-User Support**: Each user's documents are stored separately
- **Image Extraction**: Extract and reference images from your PDFs

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Telegram Bot Token from [@BotFather](https://t.me/botfather)
- Google API Key with Gemini AI and YouTube Data API v3 enabled
- Poppler (for PDF image extraction)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/study-pal-bot.git
   cd study-pal-bot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```
   TELEGRAM_TOKEN=your_telegram_bot_token
   GOOGLE_API_KEY=your_google_api_key
   ```

4. **Install Poppler**
   - **Windows**: Download from [poppler-for-windows](https://github.com/oschwartz10612/poppler-windows/releases/)
   - **macOS**: `brew install poppler`
   - **Linux**: `sudo apt-get install poppler-utils`

## 🤖 Bot Commands

- `/start` - Welcome message and bot introduction
- `/help` - Show available commands and usage examples
- `/list` - List all your uploaded documents
- `/clear` - Clear chat history (keeps your documents)

## 💡 Usage Examples

1. **Upload a PDF**
   Simply send a PDF file to the bot

2. **Ask about your document**
   ```
   What are the main concepts in this paper?
   ```

3. **Get a summary**
   ```
   Can you summarize this document?
   ```

4. **Extract key points**
   ```
   What are the key points in this PDF?
   ```

5. **Find related videos**
   ```
   Show me videos about machine learning
   ```

## 🛠️ Project Structure

```
study-pal/
├── Telegram_bot.py    # Main bot handler and command logic
├── pipeline.py        # Core processing and AI integration
├── requirements.txt   # Python dependencies
├── .env.example      # Example environment variables
└── data/             # User data and document storage
    └── <chat_id>/    # User-specific data directories
        ├── pdfs/     # Uploaded PDFs
        ├── images/   # Extracted images
        └── vectors/  # Vector store for document search
```

## 🚀 Deployment

### Local Development

```bash
python Telegram_bot.py
```

### Production (PM2)

1. Install PM2:
   ```bash
   npm install -g pm2
   ```

2. Start the bot:
   ```bash
   pm2 start "python Telegram_bot.py" --name "study-pal-bot"
   pm2 save
   pm2 startup
   ```

## 🌐 Webhook Setup (Optional)

For better performance in production, set up webhooks:

1. Configure your domain with SSL (Let's Encrypt)
2. Update the webhook URL:
   ```python
   # In Telegram_bot.py
   application.run_webhook(
       listen="0.0.0.0",
       port=8443,
       webhook_url='https://yourdomain.com/'
   )
   ```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or support, please contact [praneeth](mailto:apraneethkumar2k@gmail.com)


## Workflow

![Image](https://github.com/user-attachments/assets/88da75c4-9dab-411a-8a35-7095cf403568)

---

<div align="center">
  Made with ❤️ for students and lifelong learners
</div>
