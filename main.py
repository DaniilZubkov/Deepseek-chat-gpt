import os
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message, FSInputFile, CallbackQuery
from aiogram.utils.keyboard import InlineKeyboardBuilder, InlineKeyboardButton
from aiogram.enums import ParseMode


from keyboards import main_keyboard
from types import MappingProxyType

import asyncio
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


load_dotenv()
env = os.getenv


bot = Bot(env("BOT_TOKEN"))
dp = Dispatcher(storage=MemoryStorage())
db = Database('database.db')


system_prompt = """🤖✨ You are an expert multilingual AI assistant and developer with extensive experience. Follow these advanced guidelines:

 🌐 1. Language Processing (Intelligent Multilingual Handling) 🧠
     - 🔍 Perform 3-step language analysis:
       1️⃣ 1. Detect primary language using linguistic patterns 🕵️♂️
       2️⃣ 2. Identify secondary languages if code-mixing exceeds 30% 🌍
       3️⃣ 3. Recognize technical terms that should remain untranslated ⚙️
     - 📢 Response language mirroring:
       * 🎯 Match the user's primary language with 98% accuracy
       * 🔒 Preserve original terminology for: proper nouns, technical terms, cultural concepts
       * 🌈 For mixed input (e.g., Hinglish, Spanglish), maintain the dominant language base

 📝 2. Advanced Response Formatting (Structured & Precise) 🎨
     - 🗂 Apply hierarchical organization:
       • 🚀 **<concise 15-word summary>**
       • 📌 Supporting arguments (bullet points)
       • 💻 Examples (indented code blocks if technical)
       • 🌍 Cultural/localization notes (italic when relevant)
     - ⏱ Strict length management:
       * 📏 Real-time character count including Markdown (max 4096)
       * ✂️ Auto-truncation algorithm:
         - 🔄 Preserve complete sentences
         - 🎯 Prioritize core information
         - ➕ Add "[...]" if truncated
     - 🎭 Important style work (other Markdown and emojis):
       * 😊 Use 3-5 relevant emojis per response section
       * 🔀 Use different fonts (MARKDOWN + EMOJI combinations)

 💼 3. Specialized Content Handling ⚙️
     - 👨💻 Technical material:
       > 🔧 Maintain original English terms with localized explanations
       > 💻 Use ```code blocks``` for all commands/APIs
     - 🌏 Cultural adaptation:
       * 📏 Adjust measurements (metric/imperial)
       * 💰 Localize examples (currency, idioms)
       * 🚨 Recognize region-specific sensitivities

 ✅ 4. Quality Assurance Protocols 🔍
     - 🔄 Run pre-response checks:
       1. 📚 Language consistency validation
       2. 📊 Information density audit
       3. 🌐 Cultural appropriateness scan
     - 🧐 Post-generation review:
       * ✔️ Verify factual accuracy
       * 🎚 Ensure tone alignment (professional → friendly spectrum)
       * 📖 Confirm readability score >80%

 📤 Output template:
   ✨ **Title/Subject (if applicable)**
   
   • 🎯 Key point 1
   
   • 🔑 Key point 2
   
   - 📍 Supporting detail
   
   - 💡 Example/excerpt
   
   🌟 Additional tip (optional)
   
 🌍 <cultural/localization note if relevant>



"""



allowed_models = MappingProxyType({
    # DEEPSEEK family
    'Deepseek-R1': {
        'img_support': False,
        'models': [
            {'model': 'deepseek/deepseek-r1', 'client': 'openrouter'},
            {'model': g4f.models.deepseek_r1, 'client': 'gpt_client'}
        ],
        'api-key': env("DEEPSEEK_API_R1"),
        'code': 'deepseek-r1',
    },
    'Deepseek-V3': {
        'img_support': False,
        'models': [
            {'model': 'deepseek/deepseek-chat-v3-0324', 'client': 'openrouter'},
            {'model': g4f.models.deepseek_v3, 'client': 'gpt_client'}
        ],
        'code': 'deepseek-v3',
        'api-key': env("DEEPSEEK_API_V3"),
    },
    'Deepseek-R1 (QWEN)': {
        'img_support': False,
        'models': [
            {'model': 'deepseek/deepseek-r1-distill-qwen-32b', 'client': 'openrouter'},
            {'model': g4f.models.deepseek_r1_distill_qwen_32b, 'client': 'gpt_client'}
        ],
        'code': 'deepseek-r1-qwen',
        'api-key': env("DEEPSEEK_API_QWEN"),
    },


   # GPT family
   'GPT-4 Turbo': {
        'img_support': True,
        'models': [
            {'model': 'gpt-4-turbo', 'client': 'gpt_client'}
        ],
        'code': 'gpt4-turbo',
        'api-key': env("GPT_4_TURBO_API"),
   },
   'GPT-4.1': {
        'img_support': True,
        'models': [
            {'model': 'gpt-4.1', 'client': 'gpt_client'}
        ],
        'code': 'gpt4.1',
        'api-key': env("GPT_4_1_API"),
   },
   'GPT-4o': {
       'img_support': True,
       'models': [
            {'model': 'gpt-4o', 'client': 'gpt_client'}
       ],
       'code': 'gpt4-o',
       'api-key': env("GPT_4_O_API"),
   },

   # MINI GPT`s family
   'GPT-4.1 Mini': {
       'img_support': False,
       'models': [
            {'model': 'gpt-4.1-mini', 'client': 'gpt_client'}
       ],
       'code': 'gpt4.1-mini',
       'api-key': env("GPT_4_1_MINI_API"), 
   },
   'GPT-4o Mini': {
       'img_support': True,
       'models': [
            {'model': 'gpt-4o-mini', 'client': 'gpt_client'}
       ],
       'code': 'gpt4-o-mini',
       'api-key': env("GPT_4_O_MINI_API"),
   },

   # CLAUDE family
   'Claude 3.7 Sonnet': {
       'img_support': True,
       'models': [
            {'model': 'claude-3.7-sonnet', 'client': 'gpt_client'}
       ],
       'code': 'claude3.7-sonnet',
       'api-key': env("CLAUDE_37_API"),
   },
   'Claude 3.7 Sonnet (thinking)': {
       'img_support': True,
       'models': [
            {'model': 'clude-3.7-sonnet-thinking', 'client': 'gpt_client'}
       ],
       'code': 'claude3.7-sonnet-thinking',
       'api-key': env("CLAUDE_37_TH_API"),
   },

   # Open AI family
   'OpenAI o3': {
       'img_support': True,
       'models': [
            {'model': 'openai/o3', 'client': 'openrouter'}
       ],
       'code': 'open-ai-o3',
       'api-key': env("OAI_O3"),
   },
   'Open AI o4 Mini': {
       'img_support': True,
       'models': [
            {'model': g4f.models.o4_mini, 'client': 'gpt_client'}
       ],
       'code': 'open-ai-o4-mini',
       'api-key': env("OAI_O4_MINI"),
   },


   # QWEN family
   'Qwen3 235B A22B': {
       'img_support': False,
       'models': [
            {'model': 'qwen/qwen3-235b-a22b', 'client': 'openrouter'},
            {'model': 'qwen-3-235b', 'client': 'gpt_client'}
       ],
       'code': 'qwen3-235B-A22B',
       'api-key': env("QWEN_3_235"),
   },
   'Qwen3 30B A3B': {
       'img_support': False,
       'models': [
            {'model': 'qwen/qwen3-30b-a3b', 'client': 'openrouter'},
            {'model': 'qwen-3-30b', 'client': 'gpt_client'}
       ],
       'code': 'qwen3-30b-a3b',
       'api-key': env("QWEN_3_30"),
   },



   # Gemini family
   'Gemini 2.0 Flash Lite': {
       'img_support': True,
       'models': [
            {'model': g4f.models.gemini_2_0_flash_thinking, 'client': 'gpt_client'}
       ],
       'code': 'gemini-2.0-flash-lite',
       'api-key': env("GEMINI_API"),
   },
})




def format_answer(answer: str) -> str:
    # разделение по пунктам
    return "\n\n".join(answer.split('\n'))

def clean_output(text):
    return re.sub(r'\\boxed\{([^}]*)\}', r'\1', text)


async def download_photo(file_id: str, path: str):
    file = await bot.get_file(file_id)
    await bot.download_file(file.file_path, path)


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



def encode_img(img_path):
    with open(img_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")
        return image_b64


def cleanup_image(img_path):
    if img_path and os.path.exists(img_path):
        try:
            os.remove(img_path)
        except OSError as e:
            print(f"Error deleting image: {e}")
    else:
        return


def get_client(client_type, api_key=None):
    if client_type == 'openrouter':
        return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
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



async def create_response(model,
                          prompt: str,
                          text: str,
                          client,
                          temperature: float = 0.7,
                          top_p: float = 0.9,
                          fp: float = 0.2,
                          presence_penalty: float = 0.2,
                          max_tokens: int | None=None,
                          img_path: str | None=None,
                          provider: str | None=None,
                          headers: dict | None=None):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": []}
    ]

    if img_path:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Img Path NOT Found: {img_path}")
        
        if not os.access(img_path, os.R_OK):
            raise PermissionError(f"Нет прав на чтение изображения: {img_path}")

        img_b64 = encode_img(img_path)
        messages[1]["content"].extend([
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
        ])
    else:
        messages[1]["content"] = text

    # ОБЩИЕ ПАРАМЕТРЫ ЗАПРОСА
    params = {
        "model": model,
        "messages": messages,
        "temperature": temperature, # Креативность (0-1)
        "top_p": top_p, # Разнообразие ответов
        "frequency_penalty": fp, # Меньше повторов => меньше воды
        "presence_penalty": presence_penalty, # Поощряет новые темы
        "max_tokens": max_tokens,
        "headers": headers,
        "provider": provider
    }

    return client.chat.completions.create(**params)



async def handle_model_requests(message,
                                all_models: list,
                                model_title: str,
                                system_prompt: str,
                                img_support=True,
                                img_path: str | None=None,
                                ):
    try:
        if message.photo and not img_support:
            await message.answer('НЕТ ПОДДЕРЖКИ ФОТО В ДАННОЙ МОДЕЛИ')
            cleanup_image(img_path)

            return

        enable_message = await message.answer(
                f'🛠️ ***Пожалуйста подождите, {model_title} обрабатывает ваш запрос...***', parse_mode="MARKDOWN")

        message_text = message.caption if message.photo else message.text

        models = all_models

        for attempt, model_info in enumerate(models, 1):
            try:
                completion = await create_response(model=model_info['model'], prompt=system_prompt,
                                                   text=message_text, client=model_info['client'], img_path=img_path)

                if isinstance(message_text, list):
                    message_text = " ".join(message_text)

                # Создаем папку har_and_cookies если ее нет и даем права
                os.makedirs("har_and_cookies", exist_ok=True)
                os.chmod("har_and_cookies", 0o755)  # Права на чтение и запись

                if not completion.choices:
                    await message.answer(f'❌ ***{model_title} ничего не вернул***', parse_mode='MARKDOWN')
                    cleanup_image(img_path)
                    
                    return
                 
                model_answer = completion.choices[0].message.content
                model_new_answer = clean_output(clean_markdown(model_answer))

                await enable_message.delete()
                await send_long_message(model_new_answer, message)

                return

            except Exception as e:
                logging.warning(f"Attempt {attempt} failed: {str(e)}")

                if attempt == len(all_models):
                    await message.answer('⚠️ ***Не удалось получить ответ от модели, попробуйте еще раз позже...***', parse_mode="MARKDOWN")
    
        cleanup_image(img_path)
    
    except Exception as e:
        logging.error(f"Error in handle_model_request: {str(e)}")
        await message.answer('❌ ***Произошла ошибка при обработке запроса или модель не работает...***', parse_mode="MARKDOWN")

    finally:
        cleanup_image(img_path)









@dp.message(Command('start'))
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





@dp.message()
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
        await message.answer('❌ ***Произошла ошибка генерации ответа или модель не работает...***', parse_mode='MARKDOWN')
        return








@dp.callback_query(lambda F: True)
async def change_model(callback_query: CallbackQuery):
    black_photo_path = 'fotos/black_img.jpg'

    try:
        if callback_query.data == 'change_model':
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


        if callback_query.data.startswith('model_'):
            new_model = callback_query.data.replace('model_', '')
            db.set_model(callback_query.from_user.id, new_model)

            await update_keyboard(callback_query.message, callback_query.from_user.id)


    except Exception as e:
        print(e)





# POLLING
async def main():
    await dp.start_polling(bot)

if '__main__' == __name__:
    asyncio.run(main())
