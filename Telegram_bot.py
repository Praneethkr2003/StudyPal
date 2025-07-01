import os
import asyncio
import traceback
import re
import logging
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

import telegram
from telegram import Update, Document, InputMediaPhoto, InputFile
from telegram.constants import ChatAction, ParseMode
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from telegram.error import NetworkError, TelegramError, BadRequest
from telegram.request import HTTPXRequest

from pipeline import process_pdf, process_query  # Import your processing functions

# Configuration
PDF_DIR = os.path.abspath("data/pdfs")  # Directory to store uploaded PDFs
VECTOR_STORE_DIR = os.path.abspath("data/vectorstore")  # Directory for vector store

# Create directories if they don't exist
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    welcome_message = (
        "👋 Welcome to Study Pal Bot!\n\n"
        "I can help you with your studies by:\n"
        "• Answering questions about your PDFs\n"
        "• Finding relevant YouTube videos\n"
        "• Extracting text and images from documents\n\n"
        "📚 *How to use me:*\n"
        "1. Send me a PDF file\n"
        "2. Ask questions about the content\n"
        "3. I'll provide detailed answers and relevant resources\n\n"
        "Try these commands:\n"
        "• /help - Show all available commands\n"
        "• /list - Show your uploaded PDFs\n"
        "• /clear - Clean up the chat\n\n"
        "Let's get started! Send me a PDF or ask me a question 😊"
    )
    await update.message.reply_text(welcome_message)


async def list_documents(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """List all PDFs uploaded by the user."""
    try:
        chat_id = str(update.effective_chat.id)
        user_pdf_dir = os.path.join(PDF_DIR, chat_id)
        
        if not os.path.exists(user_pdf_dir):
            await update.message.reply_text("You haven't uploaded any PDFs yet.")
            return
            
        pdfs = [f for f in os.listdir(user_pdf_dir) if f.lower().endswith('.pdf')]
        
        if not pdfs:
            await update.message.reply_text("You haven't uploaded any PDFs yet.")
            return
            
        response = "📚 *Your Uploaded PDFs:*\n\n"
        for i, pdf in enumerate(pdfs, 1):
            file_path = os.path.join(user_pdf_dir, pdf)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
            response += f"{i}. `{pdf}` ({file_size:.2f} MB)\n"
            
        await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)
        
    except Exception as e:
        error_msg = f"❌ Error listing your documents: {str(e)}"
        print(f"ERROR: {error_msg}")
        await update.message.reply_text(error_msg)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /help is issued."""
    help_text = (
        "🤖 *Study Pal Bot Help*\n\n"
        "*Available Commands:*\n"
        "/start - Start the bot and see welcome message\n"
        "/help - Show this help message\n"
        "/list - List all your uploaded PDFs\n"
        "\n*How to use:*\n"
        "1. Send me a PDF file to analyze\n"
        "2. Ask questions about the content\n"
        "3. I'll provide answers based on the document\n"
        "\n*Examples:*\n"
        "• What is the main topic of this document?\n"
        "• Can you summarize this PDF?\n"
        "• Find information about [topic]\n"
        "\nI can also search for relevant YouTube videos if you ask!"
    )
    
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle PDF document uploads."""
    try:
        print("\n" + "="*50)
        print("DEBUG: PDF upload handler triggered")
        
        if not update.message or not update.message.document:
            print("DEBUG: No document found in the message")
            await update.message.reply_text("⚠️ Please upload a PDF file.")
            return
            
        document = update.message.document
        print(f"DEBUG: Received document - Name: {document.file_name}, MIME type: {document.mime_type}")
        
        # Validate file type
        if not document.file_name.lower().endswith('.pdf'):
            print(f"DEBUG: Invalid file type: {document.file_name}")
            await update.message.reply_text("⚠️ Please upload a PDF file.")
            return

        # Get chat ID for user-specific directory
        chat_id = str(update.effective_chat.id)
        user_pdf_dir = os.path.join(PDF_DIR, chat_id)
        os.makedirs(user_pdf_dir, exist_ok=True)
        
        # Create the full file path
        file_path = os.path.abspath(os.path.join(user_pdf_dir, document.file_name))
        
        print(f"DEBUG: Downloading file to: {file_path}")
        await update.message.reply_text("📥 Downloading your PDF file...")
        
        # Download the PDF
        file = await context.bot.get_file(document.file_id)
        await file.download_to_drive(file_path)
        
        print(f"DEBUG: File downloaded successfully to: {file_path}")
        print(f"DEBUG: File size: {os.path.getsize(file_path)} bytes")
        
        await update.message.reply_text("✅ PDF received and saved successfully!")
        
        # Show file information
        file_info = (
            f"📄 File Information:\n"
            f"• Name: {document.file_name}\n"
            f"• Size: {document.file_size} bytes\n"
            f"• MIME Type: {document.mime_type}"
        )
        
        await update.message.reply_text(file_info)
        
        # Process the PDF with the chat ID for user-specific storage
        success, message = process_pdf(file_path, chat_id=chat_id)
        
        if success:
            await update.message.reply_text(f"✅ PDF processed successfully!\n{message}")
        else:
            await update.message.reply_text(f"❌ Failed to process the PDF.\nError: {message}")
        
        await update.message.reply_text("✅ PDF processing complete! You can now ask questions about the document.")
        
    except Exception as e:
        error_msg = f"❌ Error processing PDF: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        await update.message.reply_text(error_msg)

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
logger = logging.getLogger(__name__)

# Load .env only if the file exists (for local dev)
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path, override=True)
    logger.info("✅ Loaded .env file.")
else:
    logger.warning("⚠️ .env file not found. Using environment variables from system.")

# Debug: Print all environment variables (except sensitive ones)
logger.info("Environment variables loaded. Available keys:")
for k, v in os.environ.items():
    if 'key' not in k.lower() and 'token' not in k.lower() and 'secret' not in k.lower():
        logger.info(f"  {k} = {v}")
    else:
        logger.info(f"  {k} = {'*' * min(8, len(v) or 0)}")

# Get configuration from environment variables or use defaults
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

logger.info(f"TELEGRAM_TOKEN is {'set' if TELEGRAM_TOKEN else 'NOT set'}")
logger.info(f"GOOGLE_API_KEY is {'set' if GOOGLE_API_KEY else 'NOT set'}")

if not TELEGRAM_TOKEN:
    logger.error("TELEGRAM_TOKEN not found in environment variables")
    raise ValueError("Please set TELEGRAM_TOKEN in your .env file")

if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not found in environment variables. Some features may be limited.")

def escape_markdown(text: str) -> str:
    """Escape markdown special characters."""
    if not text:
        return ""
    
    # List of special characters that need to be escaped in MarkdownV2
    escape_chars = r'\_*[]()~`>#+-=|{}.!'
    
    # Escape each special character
    escaped = []
    for char in text:
        if char in escape_chars:
            escaped.append(f'\\{char}')
        else:
            escaped.append(char)
            
    return ''.join(escaped)

def split_message(text: str, max_length: int = 4000) -> list[str]:
    """Split long messages into chunks that fit within Telegram's limits."""
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    while text:
        # Find the last newline before max_length
        split_at = text.rfind('\n', 0, max_length)
        if split_at <= 0:  # No newline found, split at max_length
            split_at = max_length
        
        chunk = text[:split_at].strip()
        if chunk:
            chunks.append(chunk)
        text = text[split_at:].strip()
    
    return chunks
# Handle text query
async def handle_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"DEBUG: Received query: {update.message.text}")
    query = update.message.text.strip()
    chat_id = str(update.effective_chat.id)
    
    if not query:
        print("DEBUG: Empty query received")
        await update.message.reply_text("Please provide a question or query.")
        return
    
    try:
        # Check if user is asking for a summary or key points
        is_summary_request = any(term in query.lower() for term in ['summary', 'summarize', 'summarise', 'summarize this', 'summarise this'])
        is_keypoints_request = any(term in query.lower() for term in ['key points', 'main points', 'key takeaways', 'important points'])
        
        # Send typing action
        await context.bot.send_chat_action(
            chat_id=chat_id,
            action=ChatAction.TYPING
        )
        
        # Send initial processing message
        status_message = await update.message.reply_text("🤔 Processing your request...")
        
        # Initialize Gemini 1.5 Flash
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash',
                                    generation_config={
                                        "temperature": 0.3 if (is_summary_request or is_keypoints_request) else 0.7,
                                        "top_p": 0.95,
                                        "top_k": 40,
                                        "max_output_tokens": 2048 if is_summary_request else 1024,
                                    })
        
        # Initialize variables with default values
        response_text = ""
        relevant_images = []
        youtube_videos = []
        
        try:
            # Get context from the vector store
            context_text, relevant_images, _ = process_query(
                query=("Provide a comprehensive summary of the document including all key points and main ideas. " 
                      if is_summary_request else 
                      "List the key points and main ideas from the document." 
                      if is_keypoints_request else query),
                use_gemini=False,  # Just get the context, don't use Gemini yet
                google_api_key=GOOGLE_API_KEY,
                include_youtube=False,
                chat_id=chat_id
            )
            
            # Prepare the prompt with context
            if not context_text or "no relevant information" in context_text.lower():
                # If no relevant documents found, use general knowledge
                if is_summary_request or is_keypoints_request:
                    response_text = "I couldn't find any documents to summarize. Please upload a PDF document first, then ask me to summarize it or extract key points."
                else:
                    prompt = (
                        "You are a helpful study assistant. "
                        f"Answer the following question using your general knowledge.\n\n"
                        f"Question: {query}\n\n"
                        "Please provide a clear, detailed, and accurate response. "
                        "Format your response in clear, readable markdown."
                    )
            else:
                # Use the found context
                if is_summary_request:
                    prompt = f"""You are a helpful study assistant. Create a comprehensive summary of the following document content.
                    
                    Document Content:
                    {context_text}
                    
                    Please provide a well-structured summary that includes:
                    1. The main topic and purpose of the document
                    2. Key sections and their main ideas
                    3. Important details, examples, or evidence
                    4. Any conclusions or recommendations
                    
                    Make the summary clear, concise, and easy to understand."""
                elif is_keypoints_request:
                    prompt = f"""Extract and list the key points from the following document content.
                    
                    Document Content:
                    {context_text}
                    
                    Format the key points as a numbered or bulleted list.
                    Focus on the most important ideas, concepts, and findings.
                    Keep each point concise but informative."""
                else:
                    # Standard question-answering prompt
                    prompt = f"""You are a helpful study assistant. Answer the user's question based on the provided context.
                    
                    Context:
                    {context_text}
                    
                    Question: {query}
                    
                    Provide a detailed and accurate response. If the answer isn't in the context, say so."""
            
            # Generate response if we have a prompt and no response text yet
            if not response_text and 'prompt' in locals() and prompt:
                response = model.generate_content(prompt)
                response_text = response.text if hasattr(response, 'text') else str(response)
            
            # Search for YouTube videos if relevant
            if any(keyword in query.lower() for keyword in ['video', 'youtube', 'watch', 'tutorial', 'how to']):
                try:
                    from pipeline import search_youtube_videos
                    youtube_videos = search_youtube_videos(
                        query=query,
                        max_results=3,
                        api_key=GOOGLE_API_KEY
                    )
                except Exception as yt_error:
                    print(f"Error searching YouTube: {yt_error}")
                    youtube_videos = []
            
        except Exception as e:
            print(f"Error in main processing: {str(e)}")
            # Fallback to the original process_query if Gemini fails
            try:
                response_text, relevant_images, youtube_videos = process_query(
                    query=query,
                    use_gemini=False,
                    google_api_key=GOOGLE_API_KEY,
                    include_youtube=True,
                    chat_id=chat_id
                )
            except Exception as fallback_error:
                print(f"Fallback also failed: {str(fallback_error)}")
                response_text = "❌ Sorry, I encountered an error processing your request. Please try again later."
                relevant_images = []
                youtube_videos = []
        
        # Update status message
        await status_message.edit_text("✅ Here's what I found:")
        
        # Send text response with Markdown formatting
        try:
            # First, escape all special Markdown characters
            safe_text = escape_markdown(response_text)
            
            # Replace common problematic patterns
            safe_text = safe_text.replace('...', '…')
            
            # Split into chunks that fit Telegram's message length limit
            chunks = split_message(safe_text)
            
            for i, chunk in enumerate(chunks):
                try:
                    # Try with Markdown first
                    await update.message.reply_text(
                        chunk,
                        parse_mode=ParseMode.MARKDOWN_V2,
                        disable_web_page_preview=True
                    )
                except BadRequest as md_error:
                    print(f"Markdown parsing error (chunk {i}), falling back to plain text: {md_error}")
                    try:
                        # Try with HTML formatting
                        await update.message.reply_text(
                            chunk,
                            parse_mode=ParseMode.HTML,
                            disable_web_page_preview=True
                        )
                    except Exception as html_error:
                        print(f"HTML parsing error (chunk {i}), falling back to plain text: {html_error}")
                        # Last resort: send as plain text
                        await update.message.reply_text(
                            chunk,
                            parse_mode=None,
                            disable_web_page_preview=True
                        )
                
                # Small delay between chunks to avoid rate limiting
                if i < len(chunks) - 1:
                    await asyncio.sleep(0.5)
                    
        except Exception as e:
            print(f"Unexpected error sending text response: {e}")
            # Try one last time with just the first 1000 characters
            try:
                await update.message.reply_text(
                    safe_text[:1000] + "\n\n[...message truncated due to length...]" if len(safe_text) > 1000 else safe_text,
                    parse_mode=None,
                    disable_web_page_preview=True
                )
            except Exception as final_error:
                print(f"Final fallback also failed: {final_error}")
                await update.message.reply_text(
                    "I had trouble formatting the response. Please try rephrasing your question."
                )
        
        # Send YouTube video recommendations if any
        if youtube_videos:
            try:
                # Create a helpful message about video recommendations
                video_links = "\n".join([
                    f"• [{video.get('title', 'Video')}]({video.get('url', '')})" 
                    for video in youtube_videos[:3]  # Limit to top 3 videos
                ])
                
                message = (
                    "🎥 *Video Recommendations*\n\n"
                    "I found some helpful videos that might answer your question. "
                    "Since I can't show videos directly in our chat, here are some suggestions:\n\n"
                    f"{video_links}\n\n"
                    "*Tips for finding great educational content:*\n"
                    "• Look for channels with high view counts and positive engagement\n"
                    "• Check the video description for timestamps and resources\n"
                    "• Read comments for additional insights and clarifications\n"
                    "• Consider the video's publication date for up-to-date information"
                )
                
                await update.message.reply_text(
                    message,
                    parse_mode=ParseMode.MARKDOWN_V2,
                    disable_web_page_preview=True
                )
                
            except Exception as e:
                print(f"Error in YouTube video handling: {e}")
                # Fallback to just sending the links
                video_links = "\n".join([f"• {v.get('url', '')}" for v in youtube_videos[:3]])
                await update.message.reply_text(
                    f"Here are some videos that might help you with your question:\n\n{video_links}",
                    disable_web_page_preview=True
                )
        
        # Send relevant images if any
        if relevant_images:
            try:
                await update.message.reply_text(
                    "🖼️ *Here are some relevant images from the document:*",
                    parse_mode=ParseMode.MARKDOWN_V2
                )
                
                # Group images in albums of up to 10 (Telegram limit)
                for i in range(0, len(relevant_images), 10):
                    media_group = []
                    for img_path in relevant_images[i:i+10]:
                        try:
                            if os.path.exists(img_path):
                                with open(img_path, 'rb') as img_file:
                                    # For the first image, add caption
                                    if not media_group:
                                        media_group.append(
                                            InputMediaPhoto(
                                                media=img_file,
                                                caption=f"📄 Page {os.path.basename(img_path).split('_')[-1].split('.')[0]}",
                                                parse_mode=ParseMode.MARKDOWN_V2
                                            )
                                        )
                                    else:
                                        media_group.append(
                                            InputMediaPhoto(media=img_file)
                                        )
                        except Exception as img_error:
                            print(f"Error preparing image {img_path}: {img_error}")
                    
                    if media_group:
                        try:
                            await context.bot.send_media_group(
                                chat_id=update.effective_chat.id,
                                media=media_group,
                                disable_notification=True
                            )
                            await asyncio.sleep(1)  # Small delay between media groups
                        except Exception as e:
                            print(f"Error sending media group: {e}")
                            # Fallback to sending images one by one
                            for img in media_group:
                                try:
                                    await context.bot.send_photo(
                                        chat_id=update.effective_chat.id,
                                        photo=img.media,
                                        caption=img.caption,
                                        parse_mode=ParseMode.MARKDOWN_V2
                                    )
                                    await asyncio.sleep(0.5)
                                except Exception as img_err:
                                    print(f"Error sending single image: {img_err}")
            except Exception as e:
                print(f"Error in image handling: {e}")
                await update.message.reply_text(
                    "⚠️ Couldn't load the images. Please try again or ask a different question.",
                    parse_mode=ParseMode.MARKDOWN_V2
                )
        
        # Add a helpful footer
        await update.message.reply_text(
            "💡 *Tip:* You can ask follow-up questions or request more details!",
            parse_mode='Markdown'
        )
            
    except Exception as e:
        error_msg = f"❌ *Error processing your query:* {str(e)}"
        print(f"ERROR in handle_query: {error_msg}")
        print(traceback.format_exc())
        
        # Try to send a more helpful error message
        try:
            if "quota" in str(e).lower():
                await update.message.reply_text(
                    "⚠️ *API Quota Exceeded*\n\n"
                    "I've reached my limit for AI responses. Please try again later or contact the administrator.",
                    parse_mode='Markdown'
                )
            else:
                await update.message.reply_text(
                    "❌ *Something went wrong*\n\n"
                    "I encountered an error while processing your request. The error has been logged.",
                    parse_mode='Markdown'
                )
        except:
            # Last resort if even the error handling fails
            await update.message.reply_text("Sorry, something went wrong. Please try again later.")

async def clear_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear all messages in the chat."""
    try:
        chat_id = update.effective_chat.id
        
        # Send a message to acknowledge the command
        message = await update.message.reply_text("🧹 Clearing all messages in this chat...")
        
        # Get the message ID of the first message in the chat
        first_message = update.message.message_id
        
        # Try to delete all messages up to the current one
        try:
            # This will delete messages in chunks of 100 (Telegram's limit)
            while True:
                messages = await context.bot.get_updates(offset=-1, limit=100, timeout=1)
                if not messages:
                    break
                    
                # Filter messages from this chat
                chat_messages = [msg.message for msg in messages 
                               if hasattr(msg, 'message') and 
                               hasattr(msg.message, 'chat') and 
                               msg.message.chat.id == chat_id and
                               msg.message.message_id < first_message]
                
                if not chat_messages:
                    break
                    
                # Delete messages in batches
                for msg in chat_messages:
                    try:
                        await context.bot.delete_message(
                            chat_id=chat_id,
                            message_id=msg.message_id
                        )
                        # Small delay to avoid hitting rate limits
                        await asyncio.sleep(0.1)
                    except Exception as e:
                        print(f"Could not delete message {msg.message_id}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error clearing messages: {e}")
            
        # Delete the command message and our status message
        try:
            await context.bot.delete_message(
                chat_id=chat_id,
                message_id=update.message.message_id
            )
            await message.delete()
        except Exception as e:
            print(f"Error cleaning up clear command: {e}")
            
    except Exception as e:
        print(f"Error in clear_chat: {e}")
        try:
            await update.message.reply_text(
                "⚠️ Couldn't clear all messages. I might not have permission to delete messages in this chat."
            )
        except:
            pass

# Error handler
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Log the error and send a message to the user.
    error_msg = f"Update {update} caused error {context.error}"
    print(f"DEBUG: {error_msg}")
    print(f"DEBUG: Error details: {context.error.__class__.__name__}: {str(context.error)}")
    print(f"DEBUG: Error traceback: {''.join(context.error.__traceback__.format()) if hasattr(context.error, '__traceback__') and context.error.__traceback__ else 'No traceback'}")
    
    try:
        # Send a message to the user
        if update and hasattr(update, 'effective_message') and update.effective_message:
            print("DEBUG: Sending error message to user")
            update.effective_message.reply_text(
                "An error occurred while processing your request. Please try again later."
            )
    except Exception as e:
        print(f"DEBUG: Error in error handler: {e}")


async def main():
    """Start the bot."""
    application = None
    loop = None
    
    try:
        # Set up asyncio event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        logger.info("Initializing bot...")
        
        # Initialize the bot with better defaults
        request = HTTPXRequest(
            connection_pool_size=8,
            read_timeout=30.0,
            write_timeout=30.0,
            connect_timeout=30.0
        )

        logger.info("Creating application...")
        application = (
            Application.builder()
            .token(TELEGRAM_TOKEN)
            .request(request)
            .build()
        )

        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/images", exist_ok=True)

        # Add handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("list", list_documents))
        application.add_handler(CommandHandler("clear", clear_chat))
        
        # Handle document uploads (PDFs)
        application.add_handler(MessageHandler(
            filters.Document.ALL & ~filters.COMMAND, 
            handle_document
        ))
        
        # Handle text messages
        application.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND, 
            handle_query
        ))
        
        # Error handler
        async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
            """Log Errors caused by Updates."""
            logger.error("Exception while handling an update:", exc_info=context.error)
            if update and hasattr(update, 'message') and update.message:
                try:
                    await update.message.reply_text(
                        "An error occurred while processing your request. Please try again."
                    )
                except Exception as e:
                    logger.error(f"Error sending error message: {e}")
        
        application.add_error_handler(error_handler)
        
        # Start the bot
        logger.info("Starting bot...")
        await application.initialize()
        await application.start()
        
        # Start the bot in polling mode
        await application.updater.start_polling()
        logger.info("Bot started successfully")
        
        # Keep the application running
        while True:
            await asyncio.sleep(3600)  # Sleep for 1 hour
            
    except asyncio.CancelledError:
        logger.info("Bot received cancellation signal")
        raise
        
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        raise
        
    finally:
        logger.info("Shutting down bot...")
        if application:
            try:
                if hasattr(application, 'updater') and application.updater:
                    await application.updater.stop()
                if hasattr(application, 'stop') and callable(application.stop):
                    await application.stop()
                if hasattr(application, 'shutdown') and callable(application.shutdown):
                    await application.shutdown()
            except Exception as e:
                logger.error(f"Error during shutdown: {e}", exc_info=True)
        logger.info("Bot has stopped")

def run_bot():
    """Run the bot with proper cleanup."""
    loop = None
    try:
        # Create and set the event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the main function
        loop.run_until_complete(main())
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        raise
    finally:
        # Clean up
        if loop:
            try:
                # Cancel all running tasks
                tasks = asyncio.all_tasks(loop=loop)
                for task in tasks:
                    task.cancel()
                
                # Run the event loop until all tasks are cancelled
                if tasks:
                    loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                
                # Close the loop
                if loop.is_running():
                    loop.stop()
                if not loop.is_closed():
                    loop.close()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    run_bot()
