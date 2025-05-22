from telegram import BotCommand

async def post_init(application):
    commands = [
        BotCommand("start", "Start the bot"),
        BotCommand("help", "Show help"),
        BotCommand("upload", "Upload data"),
        BotCommand("answer", "Get answers"),
        BotCommand("reset", "Reset data")
    ]
    await application.bot.set_my_commands(commands)