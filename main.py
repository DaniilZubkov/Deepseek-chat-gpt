import os
from aiogram import Bot, Dispatcher, F, types, Router
from aiogram.filters import Command
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message, FSInputFile, CallbackQuery
from aiogram.utils.keyboard import InlineKeyboardBuilder, InlineKeyboardButton
from aiogram.enums import ParseMode
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext

from keyboards import main_keyboard
from types import MappingProxyType

import asyncio
from asyncio import Semaphore

from openai import OpenAI
from g4f.client import Client
from g4f.Provider import You, DeepInfra
import g4f
import re
from db import Database
from dotenv import load_dotenv
import base64
import random
import logging
from functools import lru_cache
import time

from config import allowed_models, system_prompt, img_generation_models
from funcs import format_answer, clean_output, download_photo, clean_markdown, encode_img, cleanup_image, get_client, update_keyboard, \
send_long_message
from ai.model_funcs import create_response, _prepare_messages


load_dotenv()
env = os.getenv


bot = Bot(env("BOT_TOKEN"))
dp = Dispatcher(storage=MemoryStorage())
db = Database('database.db')

router = Router()



class ModelSelection(StatesGroup):
    choosing_model = State()
    choosing_version = State()




async def handle_model_requests(message,
                                all_models: list,
                                model_title: str,
                                system_prompt: str,
                                img_support=True,
                                img_path: str | None=None,
                                ):
    try:

        if message.photo and not img_support:
            await message.answer("❌ Эта модель не поддерживает обработку изображений")
            return

        enable_message = await message.answer(
                f'🛠️ ***Пожалуйста подождите, {model_title} обрабатывает ваш запрос...***', parse_mode="MARKDOWN")

        message_text = str(message.caption if message.photo else message.text)[:8000] 

        if len(message_text) <= 8:
            await message.answer('***Пожалуйста, введите запрос не меньше 8 символов.***', parse_mode='MAKRDOWN')
            return

        # async with asyncio.TaskGroup() as tg:
        success = False
        for attempt, model_info in enumerate(all_models, 1):
            try:
                if attempt > 1:
                    await asyncio.sleep(1)

                start_time = time.monotonic()

                # task = tg.create_task(
                #     asyncio.wait_for(
                #         create_response(
                #             model=model_info['model'],
                #             prompt=system_prompt,
                #             text=message_text,
                #             client=model_info['client'],
                #             img_path=img_path,
                #         ),
                #         timeout=40.0  
                #     )
                # )


                try:
                    completion = await create_response(
                        model=model_info['model'],
                        prompt=system_prompt,
                        text=message_text,
                        client=model_info['client'],
                        img_path=img_path,
                    )

                    
                    # completion = await task
                    if not completion.choices:
                        logging.warning(f"Attempt {attempt}: Empty choices")
                        continue

                    model_answer = completion.choices[0].message.content
                    if not model_answer.strip():
                        logging.warning(f"Attempt {attempt}: Empty response")
                        continue

                    model_new_answer = clean_output(clean_markdown(model_answer))
                    if not model_new_answer.strip():
                        logging.warning("Получен пустой ответ от модели")
                        continue

                    await enable_message.delete()
                    await send_long_message(model_new_answer, message)
                    success = True
                    break

                    # return True

                except asyncio.TimeoutError:
                    logging.warning(f"Timeout for model {model_info['model']}")
                    continue

            except Exception as e:
                logging.warning(f"Attempt {attempt} failed: {str(e)}")
                continue

            if success is False:
                await message.answer('⚠️ ***Не удалось получить ответ от модели***\n\n' \
                            '___Попробуйте переформулировать вопрос или попробовать позже...___', parse_mode="MARKDOWN")

                return

    except Exception as e:
        logging.error(f"Error in handle_model_request: {str(e)}")
        await message.answer('❌ ***Произошла ошибка при обработке запроса или модель не работает...***', parse_mode="MARKDOWN")

        return

    finally:
        await cleanup_image(img_path)
        logging.warning(f"Model response time: {time.monotonic() - start_time:.2f}s")

        return
    






@router.message(Command('start'))
async def start(message: Message):
    try:
        db.create_tables()
        black_photo_path = 'fotos/black_img.jpg'

        if (not db.user_exists(message.from_user.id)):
            db.add_user(message.from_user.id)
            db.set_nickname(message.from_user.id, message.from_user.username)
            db.set_signup(message.from_user.id, 'done')

        await message.answer_photo(photo=FSInputFile(black_photo_path),
                                   caption=f'Привет, {message.from_user.first_name}. Я AI ассистент в Telegram. Здесь доступны многие модели такие как: ***GPT, DEEPSEEK, GEMIMI и другие.*** \n\n'
                                           f'***Ты можешь выбрать удобную для себя модель по кнопке.*** 👇',
                                   parse_mode="MARKDOWN", reply_markup=main_keyboard())

    except Exception as e:
        print(e)





@router.message()
async def get_message(message: Message):
    try:
        current_model = db.get_model(message.from_user.id)
        
        if not current_model:
            raise ValueError("Model not set for user")

        model_info = next((v for v in allowed_models.values() if v['code'] == current_model), None)

        if not model_info:
            raise ValueError(f"Model not found or not supported: {current_model}")

        img_path = None
        if message.photo:
            img_path = f'user_photos/{message.photo[-1].file_id}.jpg'
            await download_photo(message.photo[-1].file_id, img_path)


        clients = []
        for model_conf in model_info['models']:
            client_type = model_conf['client']
            if client_type == 'openrouter':
                clients.append({
                    'model': model_conf['model'],
                    'client': get_client(client_type, model_info['api-key'])
                })
            else:
                clients.append({
                    'model': model_conf['model'],
                    'client': get_client(client_type)
                })

        model_title = next((k for k, v in allowed_models.items() if v['code'] == current_model), current_model)
        
        await handle_model_requests(
            message=message,
            img_support=model_info['img_support'],
            img_path=img_path,
            all_models=clients,
            model_title=model_title,
            system_prompt=system_prompt
        )

        return

    except Exception as e:
        print(e)
        await message.answer('❌ ***Произошла ошибка генерации ответа или модель не работает...***', parse_mode='MARKDOWN')
        return



# СМЕНА ТЕКСТОВОЙ МОДЕЛИ
@router.callback_query(F.data == 'change_model')
async def change_txt_model(callback_query: CallbackQuery):
    black_photo_path = 'fotos/black_img.jpg'

    try:
        builder = InlineKeyboardBuilder()
        cur_model = db.get_model(callback_query.from_user.id)


        for model_name, model_data in allowed_models.items():
            if model_data['code'] == cur_model:
                builder.button(text=f"✅ {model_name}", callback_data=f"model_{model_data['code']}")
            else:
                builder.button(text=model_name, callback_data=f"model_{model_data['code']}")

        builder.adjust(3, 1, 1, 1, 1, 1, 2, 2, 2)

        await callback_query.message.answer(f'В разделе есть модели такие как <b>ChatGPT, Claude, Gemini, Deepseek и многие другие</b>:\n\n'
                                               f'<b>🐼 Deepseepk-R1</b> - Модель для сложных задач с глубоким рассуждением\n'
                                               f'<b>🐳 Deepseek-V3</b> - Китайская текстовая модель, созданая Ляном Вэньфэном\n'
                                               f'<b>⚡ Deepseek-QWEN</b> - Deepseek на базе китайской модели QWEN\n\n'
                                               f'<b>🍓 OpenAI-O3</b> - Рассуждающая модель с наилучшими решениями\n'
                                               f'<b>🧠 OpenAI-O4 mini</b> - Для кодинга и точных наук\n\n'
                                               f'<b>✨ GPT-4 Turbo</b> – Мощная и быстрая модель OpenAI с увеличенным контекстом.\n'
                                               f'<b>💥 GPT-4.1</b> – Улучшенная версия GPT-4 с более точными ответами.\n'
                                               f'<b>💎 GPT-4o</b> – Оптимизированная для скорости и эффективности версия GPT-4.\n'
                                               f'<b>🍃 GPT-4.1 Mini</b> – Компактная и экономичная версия GPT-4.1.\n\n'
                                               f'<b>🔮 Claude 3.7 Sonnet</b> – Сбалансированная модель от Anthropic с высокой точностью.\n'
                                               f'<b>☁ Claude 3.7 Sonnet (thinking)</b> – Версия с увеличенным временем "размышления" для более глубоких ответов.\n\n'
                                               f'<b>💬 Qwen3 235B A22B</b> – Масштабная модель Qwen с 235 млрд параметров, высокая производительность в сложных задачах.\n'
                                               f'<b>🤖 Qwen3 30B A3B</b> – Более компактная, но эффективная версия Qwen3, подходит для баланса скорости и качества.\n\n'
                                               f'<b>💡 Gemini 2.0 Flash Lite</b> – Облегчённая и быстрая версия Gemini 2.0, оптимизирована для оперативных запросов.',
                                       parse_mode="html", reply_markup=builder.as_markup())
    except Exception as e:
        print(e)




@router.callback_query(F.data.startswith('model_'))
async def choose_txt_model(callback_query: CallbackQuery):
    try:
        new_model = callback_query.data.replace('model_', '')
        db.set_model(callback_query.from_user.id, new_model)

        await update_keyboard(callback_query.message, callback_query.from_user.id)
    
    except Exception as e:
        print(e)












# IMG GENERATION MODELS
@router.callback_query(F.data == 'change_model_photo_categ')
async def change_img_model(callback_query: CallbackQuery):
    try:
        builder = InlineKeyboardBuilder()
        for model_name in img_generation_models.keys():
            builder.button(text=model_name, callback_data=f'img_selected_model_{model_name}')

        builder.adjust(1)

        await callback_query.message.answer('asdadsadsada', parse_mode='MARKDOWN', reply_markup=builder.as_markup())
    except Exception as e:
        pass


@router.callback_query(F.data.startswith('img_selected_model_'))
async def change_img_model_version(callback_query: CallbackQuery):
    try:
        model_name = callback_query.data.split("_")[-1]
        print(model_name)
    
        if model_name not in img_generation_models:
            await callback_query.message.answer("❌ Модель не найдена!", parse_mode='MARKDOWN')
            return
    
        versions = img_generation_models[model_name]['versions']
        builder = InlineKeyboardBuilder()

        for version in versions:
            builder.button(text=version['model'], callback_data=f"select_img_version_{version['code']}")

        builder.adjust(1)
    
        await callback_query.message.edit_text(
            f"⚙️ Доступные версии {model_name}:",
            reply_markup=builder.as_markup()
        )
    except Exception as e:
        pass


@router.callback_query(F.data.startswith('select_img_version_'))
async def select_img_model(callback_query: CallbackQuery):
    try:
        version_code = callback_query.data.split("_")[-1]
        db.set_model_img(callback_query.from_user.id, version_code)

        model_name = None
        for m_name, m_data in img_generation_models.items():
            if any(v['code'] == version_code for v in m_data['versions']):
                model_name = m_name
                break

        if model_name:
            versions = img_generation_models[model_name]['versions']
            builder = InlineKeyboardBuilder()
            
            for version in versions:
                if version['code'] == version_code:
                    builder.button(text=f"✅ {version['model']}", callback_data=f"select_img_version_{version['code']}")
                else:
                    builder.button(text=version['model'], callback_data=f"select_img_version_{version['code']}")
            
            builder.adjust(1)

            await callback_query.message.edit_text(
                f"⚙️ Доступные версии {model_name}:",
                reply_markup=builder.as_markup()
            )

    except Exception as e:
        print(e)








# POLLING
async def main():
    dp.include_router(router)
    await dp.start_polling(bot)

if '__main__' == __name__:
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('Goodbye!')
