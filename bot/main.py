from telegram.ext import (
    CommandHandler, MessageHandler,
    filters, ApplicationBuilder
)
import config
from handlers import handlers
from handlers.post_init import post_init
from setup.setup import setup

user_sessions = {}


def main():
    token = config.telegram_token
    app = ApplicationBuilder().token(token).post_init(post_init).build()

    app.add_handler(CommandHandler("start", handlers.start_handler))
    app.add_handler(CommandHandler("help", handlers.help_handler))
    app.add_handler(CommandHandler("upload", handlers.upload_handler))
    app.add_handler(CommandHandler("reset", handlers.reset_handler))
    app.add_handler(CommandHandler("answer", handlers.answer_handler))
    app.add_handler(MessageHandler(filters.Document.ALL | filters.TEXT & ~filters.COMMAND,
                                   handlers.general_message_handler))

    app.run_polling()

if __name__ == "__main__":
    setup()
    main()
