import logging
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ParseMode
from telegram.error import NetworkError, TimedOut
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters


# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Загрузка переменных окружения из .env в корне проекта
root_dir = Path(__file__).parent.parent
load_dotenv(root_dir / ".env")

BOT_TOKEN = os.getenv("BOT_TOKEN")  
PREDICT_API_URL = os.getenv("PREDICT_API_URL", "http://127.0.0.1:8000/predict")
TG_PROXY = os.getenv("TG_PROXY")


async def send_welcome(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обработчик команды /start. Приветствует пользователя и объясняет функционал.
    """
    await update.message.reply_text(
        "Привет! Я ML-бот для оценки вилки зарплат по описанию вакансии.\n\n"
        "Отправь мне текст вакансии (требования, обязанности, описание, опыт), и я попробую предсказать примерную зарплату на его основе"
    )


async def handle_description(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обработчик обычных сообщений. Принимает текст вакансии и возвращает предсказанную зарплату.
    """
    if not update.message or not update.message.text:
        return

    await update.message.reply_text("Анализирую текст вакансии... ")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                PREDICT_API_URL,
                json={"description": update.message.text},
            )
            response.raise_for_status()
            pred = response.json()

        # Модель обучалась на зарплатах в тысячах рублей, поэтому переводим в рубли.
        min_s = pred["salary_min"] * 1000
        max_s = pred["salary_max"] * 1000

        # Форматирование с разделением тысяч для читаемости (например, 100 000 руб.)
        min_formatted = f"{int(min_s):,}".replace(",", " ")
        max_formatted = f"{int(max_s):,}".replace(",", " ")

        response = (
            f" *Предсказанная вилка зарплаты:*\n"
            f"От: `{min_formatted}` руб.\n"
            f"До: `{max_formatted}` руб."
        )
        await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}", exc_info=True)
        await update.message.reply_text("Произошла ошибка при анализе текста. Возможно, модель еще запускается.")


def run_bot() -> None:
    retry_delay = 5

    while True:
        builder = Application.builder().token(BOT_TOKEN)
        if TG_PROXY:
            logger.info("Используется прокси для Telegram API")
            builder = builder.proxy(TG_PROXY).get_updates_proxy(TG_PROXY)

        app = builder.build()
        app.add_handler(CommandHandler("start", send_welcome))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_description))

        try:
            logger.info("Start polling")
            app.run_polling(
                drop_pending_updates=True,
                close_loop=False,
                stop_signals=None,
            )
            break
        except (TimedOut, NetworkError, httpx.TimeoutException, httpx.TransportError) as e:
            logger.warning("Сетевая ошибка Telegram (%s). Повтор через %s сек.", e, retry_delay)
            time.sleep(retry_delay)
        except Exception:
            logger.exception("Неожиданная ошибка в polling. Повтор через %s сек.", retry_delay)
            time.sleep(retry_delay)


def main() -> None:
    """
    Главная функция запуска: стартует поллинг бота.
    """
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN не найден в переменных окружения")

    logger.info("Запуск Telegram бота...")
    run_bot()


if __name__ == "__main__":
    main()
