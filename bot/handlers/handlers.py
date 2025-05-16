from telegram import Update
from telegram.ext import ContextTypes
import asyncio
import os

from service.data_extractor import *
from service.session import Session

user_sessions = {}


async def test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Id:", update.effective_user.id)
    print("Username:", update.effective_user.username)
    print("Message:", update.message.text)
    print("Args:", context.args)
    print()
    await update.message.reply_text("OK")


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["expecting_data"] = True
    await update.message.reply_text("Please send your data (text or file).")


async def reset_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_sessions.pop(user_id, None)
    context.user_data.clear()
    await update.message.reply_text("Your data has been reset. Use /start to upload new data.")


async def data_upload_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if not context.user_data.get("expecting_data"):
        await update.message.reply_text("You're already in a session. Send a task or /reset to start over.")
        return

    if update.message.document:
        document = update.message.document
        file = await document.get_file()

        file_name = document.file_name or "unknown"
        file_ext = Path(file_name).suffix.lower()

        file_path = f"{os.getcwd()}/tmp/{user_id}_{file_name}"
        await file.download_to_drive(file_path)
        data = file_path
        source = file_ext

    elif update.message.text:
        data = update.message.text
        source = "str"
    else:
        await update.message.reply_text("Unsupported input. Please send a file or text.")
        return

    try:
        formatted_data = extract_data(data, source)
        user_sessions[user_id] = Session(formatted_data)
        context.user_data["expecting_data"] = False
        await update.message.reply_text("Data uploaded successfully. You can now send tasks.")
    except Exception as e:
        await update.message.reply_text(f"Error processing your file: {e}")


async def questions_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if user_id not in user_sessions:
        await update.message.reply_text("No data found. Use /start to upload data.")
        return

    questions = update.message.text
    if not questions:
        await update.message.reply_text("Please send a text task.")
        return

    session = user_sessions[user_id]
    session.add_questions(extract_questions(questions))
    await update.message.reply_text("Questions remembered.")


async def answer_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in user_sessions:
        await update.message.reply_text("No data found. Use /start to upload data.")
        return

    session = user_sessions[user_id]
    # Show questions asked
    print(session.questions)
    await update.message.reply_text(f"Questions asked: {format_questions(session.questions)}")

    # Answer questions
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, session.search)

    await update.message.reply_text(f"Answers: {format_answers(result)}")

async def general_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if context.user_data.get("expecting_data"):
        await data_upload_handler(update, context)
    elif user_id in user_sessions:
        await questions_handler(update, context)
    else:
        await update.message.reply_text("Please use /start to upload data first.")
