from datetime import datetime
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
    await update.message.reply_text("Добро пожаловать в RAG-bot!")
    await help_handler(update, context)


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('''/upload - загрузить данные (текст в сообщении, или файл). После успешной загрузки, можно задавать вопросы. Каждая новая строка является вопросом. Вопросы можно задавать несколькими сообщениями

/answer - получить ответ на заданные вопросы

/reset - сбросить данные, чтобы загрузить новые

/help - помощь
''')


async def upload_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["expecting_data"] = True
    await update.message.reply_text("Прикрепите данные (текст или файл)")


async def reset_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session = user_sessions.pop(user_id, None)
    context.user_data.clear()
    if session is not None:
        session.clear()
    await update.message.reply_text("Данные сброшены. Используйте /upload чтобы загрузить их заново")


async def data_upload_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if not context.user_data.get("expecting_data"):
        await update.message.reply_text("Для вас уже существует сессия. Задайте вопрос, или используйте /reset чтобы начать заново.")
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
        await update.message.reply_text("Неподдерживаемый формат.")
        return

    try:
        formatted_data = extract_data(data, source)
        user_sessions[user_id] = Session(formatted_data)
        context.user_data["expecting_data"] = False
        await update.message.reply_text("Данные загружены. Можно задавать вопросы.")
    except Exception as e:
        print(e)
        await update.message.reply_text(f"Ошибка при обработке данных")


async def questions_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if user_id not in user_sessions:
        await update.message.reply_text("Данные не найдены. Используйте /upload чтобы загрузить их.")
        return

    questions = update.message.text
    if not questions:
        await update.message.reply_text("Задайте вопрос.")
        return

    session = user_sessions[user_id]
    session.add_questions(extract_questions(questions))
    await update.message.reply_text("Вопрос сохранен.")


async def answer_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in user_sessions:
        await update.message.reply_text("Данные не найдены. Используйте /upload чтобы загрузить их.")
        return

    session = user_sessions[user_id]
    # Show questions asked
    print(f"{datetime.now()} User: {update.effective_user.username} Questions asked: {session.questions}")
    await update.message.reply_text(f"Вопросы: {format_questions(session.questions)}")

    # Answer questions
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, session.search)

    answers = format_answers(result)
    print(f"{datetime.now()} User: {update.effective_user.username} Answers: {result}")
    await update.message.reply_text(f"Ответы: {answers}")

async def general_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if context.user_data.get("expecting_data"):
        await data_upload_handler(update, context)
    elif user_id in user_sessions:
        await questions_handler(update, context)
    else:
        await update.message.reply_text("Используйте /upload чтобы загрузить данные")
