# telegram_bot.py
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from qa_pipeline import get_qa_pipeline
from data_pipeline import build_index
from dotenv import load_dotenv

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

load_dotenv()

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')  # Replace with your actual token in .env file

# Initialize index
build_index = build_index()  # comment this line if valid index already exists

# Initialize QA pipeline
qa_pipeline = get_qa_pipeline()


def start(update, context):
    update.message.reply_text("Привіт! Я чат-бот ФІОТ КПІ. Ставте свої питання українською!")


def handle_user_message(update, context):
    user_question = update.message.text.strip()

    # Run the pipeline by passing the query under the key "text"
    result = qa_pipeline.run(
        {
            "embedder": {"text": user_question},
            "chat_prompt_builder": {"question": user_question}
        },
        include_outputs_from={"retriever", 'chat_prompt_builder'}
    )
    print(result)

    answers = result["llm"]["replies"]
    if not answers:
        bot_answer = "На жаль, я не зміг знайти відповідь на ваше питання :( cформулюйте питання детальніше"
    else:
        bot_answer = answers[0].text.replace("**", "")

    update.message.reply_text(bot_answer)


def main():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_user_message))

    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
