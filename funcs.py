
import os
import re
import base64
from typing import Optional
import aiofiles
from functools import lru_cache
import logging

from aiogram import Bot
from aiogram.types import Message
from aiogram.utils.keyboard import InlineKeyboardButton, InlineKeyboardBuilder
from openai import OpenAI, AsyncOpenAI
from g4f.client import Client
from db import Database
from dotenv import load_dotenv
from PIL import Image
import io
from g4f.Provider import RetryProvider, OpenaiChat

from config import allowed_models



load_dotenv()
env = os.getenv


db = Database('database.db')
bot = Bot(env("BOT_TOKEN"))



def format_answer(answer: str) -> str:
    # разделение по пунктам
    return "\n\n".join(answer.split('\n'))

def clean_output(text):
    return re.sub(r'\\boxed\{([^}]*)\}', r'\1', text)


async def download_photo(file_id: str, path: str):
    try:
        os.makedirs("har_and_cookies", exist_ok=True)
        os.chmod("har_and_cookies", 0o755)

        file = await bot.get_file(file_id)
        await bot.download_file(file.file_path, path)
    except Exception:
        pass


def clean_markdown(text: str) -> str:
    patterns = [
        # (r'```.*?\n(.*?)\n```', r'\1', re.DOTALL), можно убрать код (если нужно)
        # (r'`(.*?)`', r'\1'),
        (r'\*\*(.*?)\*\*', r'*\1*'),  # Жирный → корректный Markdown
        (r'^#+\s*(.+)$', r'*\1*', re.MULTILINE),  # Заголовки → жирный
    ]

    for pattern in patterns:
        if len(pattern) == 3:
            p, r, f = pattern
            text = re.sub(p, r, text, flags=f)
        else:
            p, r = pattern
            text = re.sub(p, r, text)

    return text



async def encode_img(img_path: str) -> str:
    with Image.open(img_path) as img:
        img.thumbnail((1024, 1024))  # Ресайз
        buffer = io.BytesIO()
        img.save(buffer, "JPEG", quality=85)  # Сжатие
        return base64.b64encode(buffer.getvalue()).decode()



async def cleanup_image(img_path: str):
    try:
        if img_path and os.path.exists(img_path):
            os.remove(img_path)
    except Exception as e:
        logging.warning(f"Failed to cleanup image {img_path}: {str(e)}")



def get_client(client_type, api_key=None):
    if client_type == 'openrouter':
        return AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    elif client_type == 'gpt_client':
        return Client()
    return 


async def update_keyboard(message: Message, user_id: int):
    """Обновляет клавиатуру с моделями для указанного пользователя"""
    current_model = db.get_model(user_id)
    builder = InlineKeyboardBuilder()

    for name, data in allowed_models.items():
        # Добавляем галочку к текущей выбранной модели
        is_selected = (data['code'] == current_model)
        builder.button(
            text=f"{'✅ ' if is_selected else ''}{name}",
            callback_data=f"model_{data['code']}"
        )

    builder.adjust(3, 1, 1, 1, 1, 1, 2, 2, 2)
    await message.edit_reply_markup(reply_markup=builder.as_markup())

 


async def send_long_message(text, message):
    if len(text) <= 4096:
        await message.answer(text, parse_mode='MARKDOWN')
    else:
        parts = [text[i:i+4096] for i in range(0, len(text), 4096)]
        for part in parts:
            await message.answer(part, parse_mode='MARKDOWN')



