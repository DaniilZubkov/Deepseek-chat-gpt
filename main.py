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


load_dotenv()
env = os.getenv


bot = Bot(env("BOT_TOKEN"))
dp = Dispatcher(storage=MemoryStorage())
db = Database('database.db')



# DEEPSEEK SETTINGS (Deepseek-r1 qwen 32b) and other models



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
        'code': 'deepseek-r1',
        'api-key': env("DEEPSEEK_API_R1"),
    },
    'Deepseek-V3': {
        'code': 'deepseek-v3',
        'api-key': env("DEEPSEEK_API_V3"),
    },
    'Deepseek-R1 (QWEN)': {
        'code': 'deepseek-r1-qwen',
        'api-key': env("DEEPSEEK_API_QWEN"),
    },


   # GPT family
   'GPT-4 Turbo': {
        'code': 'gpt4-turbo',
        'api-key': env("GPT_4_TURBO_API"),
   },
   'GPT-4.1': {
        'code': 'gpt4.1',
        'api-key': env("GPT_4_1_API"),
   },
   'GPT-4o': {
       'code': 'gpt4-o',
       'api-key': env("GPT_4_O_API"),
   },

   # MINI GPT`s family
   'GPT-4.1 Mini': {
       'code': 'gpt4.1-mini',
       'api-key': env("GPT_4_1_MINI_API"), 
   },
   'GPT-4o Mini': {
       'code': 'gpt4-o-mini',
       'api-key': env("GPT_4_O_MINI_API"),
   },

   # CLAUDE family
   'Claude 3.7 Sonnet': {
       'code': 'claude3.7-sonnet',
       'api-key': env("CLAUDE_37_API"),
   },
   'Claude 3.7 Sonnet (thinking)': {
       'code': 'claude3.7-sonnet-thinking',
       'api-key': env("CLAUDE_37_TH_API"),
   },

   # Open AI family
   'OpenAI o3': {
       'code': 'open-ai-o3',
       'api-key': env("OAI_O3"),
   },
   'Open AI o4 Mini': {
       'code': 'open-ai-o4-mini',
       'api-key': env("OAI_O4_MINI"),
   },


   # QWEN family
   'Qwen3 235B A22B': {
       'code': 'qwen3-235B-A22B',
       'api-key': env("QWEN_3_235"),
   },
   'Qwen3 30B A3B': {
       'code': 'qwen3-30b-a3b',
       'api-key': env("QWEN_3_30"),
   },



   # Gemini family
   'Gemini 2.0 Flash Lite': {
       'code': 'gemini-2.0-flash-lite',
       'api-key': env("GEMINI_API"),
   },
})




def format_answer(answer: str) -> str:
    # Пример форматирования: разделение по пунктам
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
                          provider=None):
    if img_path:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Img Path NOT Found: {img_path}")
        
        if not os.access(img_path, os.R_OK):
            raise PermissionError(f"Нет прав на чтение изображения: {img_path}")

        with open(img_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
            
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                    ],
                },
            ],
            provider=provider,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=fp,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens
        )
        return completion

    
    completion = client.chat.completions.create(
        extra_body={},
        model=model,
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": text
            }
        ],
        provider=provider,
        temperature=temperature,  # Контроль "креативности" (0–1)
        top_p=top_p,  # Влияет на разнообразие ответов
        frequency_penalty=fp,  # Уменьшает повторения
        presence_penalty=presence_penalty,  # Поощряет новые темы
        max_tokens=max_tokens
    )
    return completion









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
    async def send_long_message(text, message):
        if len(text) <= 4096:
            await message.answer(text, parse_mode='MARKDOWN')
        else:
            parts = [text[i:i+4096] for i in range(0, len(text), 4096)]
            for part in parts:
                await message.answer(part, parse_mode='MARKDOWN')


    try:
        current_model = db.get_model(message.from_user.id)

        img_path = None
        if message.photo:
            img_path = f'user_photos/{message.photo[-1].file_id}.jpg'
            await download_photo(message.photo[-1].file_id, img_path)


        new_api = ''
        model_title = ''
        for model, api in allowed_models.items():
            if api['code'] == current_model:
                new_api = api['api-key']
                model_title = model
                break

        if not new_api or not current_model:
            raise ValueError(f"API key not found for model: {current_model}")


        client = OpenAI(api_key=f"{new_api}", base_url="https://openrouter.ai/api/v1")
        gpt_client = Client()




        # DEEPSEEK R1 requests
        if current_model == 'deepseek-r1':
            enable_message = await message.answer(
                f'🛠️ ***Пожалуйста подождите, {model_title} обрабатывает ваш запрос...***', parse_mode="MARKDOWN")

            # ПОПЫТКА 1
            try:
                completion = await create_response(model='deepseek/deepseek-r1', prompt=system_prompt,
                                                   text=message.text, client=client)

                if img_path and os.path.exists(img_path):
                    os.remove(img_path)
                
                if not completion.choices:
                    await message.answer(f'❌ ***{model_title} ничего не вернул***', parse_mode='MARKDOWN')

                
                deepseek_answer = completion.choices[0].message.content
                new_deepseek_answer = clean_output(clean_markdown(deepseek_answer))

                await enable_message.delete()
                await send_long_message(new_deepseek_answer, message)
                print('deepseek 1')


            # ПОПЫТКА 2
            except Exception:
                completion = await create_response(model=g4f.models.deepseek_r1, prompt=system_prompt, text=message.text, client=gpt_client)

                
                if img_path and os.path.exists(img_path):
                    os.remove(img_path)
               

                if not completion.choices:
                    await message.answer(f'❌ ***{model_title} ничего не вернул***', parse_mode='MARKDOWN')

                deepseek_answer = completion.choices[0].message.content

                print(deepseek_answer)
                print(clean_markdown(deepseek_answer))
                print(len(clean_output(clean_markdown(deepseek_answer))))

                new_deepseek_answer = clean_output(clean_markdown(deepseek_answer))

            # if '`' in new_deepseek_answer[0:2] and '`' in new_deepseek_answer[-3:-1]:
            #     if len([char for char in new_deepseek_answer]) >= 4096:
            #         while new_deepseek_answer:
            #             await message.answer(new_deepseek_answer[:4096], parse_mode='MARKDOWN')
            #             # удаление отправленной части текста
            #             new_deepseek_answer = new_deepseek_answer[4096:]
            #     await message.answer(new_deepseek_answer[2:-3], parse_mode='MARKDOWN')
            #
                await enable_message.delete()
                await send_long_message(new_deepseek_answer, message)
                print('deepseek 2')




        # DEEPSEEK FAMILY
        if current_model == 'deepseek-v3':
            enable_message = await message.answer(f'🛠️ ***Пожалуйста подождите, {model_title} обрабатывает ваш запрос...***',
                                 parse_mode="MARKDOWN")
            try:
                completion = await create_response(model=g4f.models.deepseek_v3, text=message.text, prompt=system_prompt, client=gpt_client)

                if not completion.choices:
                    await message.answer(f'❌ ***{model_title} ничего не вернул***', parse_mode='MARKDOWN')

                deepseek_answer = completion.choices[0].message.content
                new_deepseek_answer = clean_output(clean_markdown(deepseek_answer))
                
                await enable_message.delete()
                await send_long_message(new_deepseek_answer, message)


            except Exception as e: 
                completion = await create_response(model='deepseek/deepseek-chat-v3-0324', text=message.text, prompt=system_prompt, client=client)
                deepseek_answer = completion.choices[0].message.content


                print(deepseek_answer)
                print(clean_markdown(deepseek_answer))
                print(len(clean_output(clean_markdown(deepseek_answer))))
                new_deepseek_answer = clean_output(clean_markdown(deepseek_answer))


                await enable_message.delete()
                await send_long_message(new_deepseek_answer, message)







        # DEEPSEK r1 (qwen)
        if current_model == 'deepseek-r1-qwen':
            enable_message = await message.answer(f'🛠️ ***Пожалуйста подождите, {model_title} обрабатывает ваш запрос...***',
                                 parse_mode="MARKDOWN")

            try:
                completion = await create_response(model=g4f.models.deepseek_r1_distill_qwen_32b, text=message.text, prompt=system_prompt, client=gpt_client)
                
                if not completion.choices:
                    await message.answer(f'❌ ***{model_title} ничего не вернул***', parse_mode='MARKDOWN')

                deepseek_answer = completion.choices[0].message.content
                new_deepseek_answer = clean_output(clean_markdown(deepseek_answer))

                await enable_message.delete()
                await send_long_message(new_deepseek_answer, message)

            except Exception:
                completion = await create_response(model='deepseek/deepseek-r1-distill-qwen-32b', text=message.text, prompt=system_prompt, client=client)

                if not completion.choices:
                    await message.answer(f'❌ ***{model_title} ничего не вернул***', parse_mode='MARKDOWN')
        
                deepseek_answer = completion.choices[0].message.content
                new_deepseek_answer = clean_output(clean_markdown(deepseek_answer))

                await enable_message.delete()
                await send_long_message(new_deepseek_answer, message)







        # GPT
        if current_model == 'gpt4-turbo':
            enable_message = await message.answer(
                f'🛠️ ***Пожалуйста подождите, {model_title} обрабатывает ваш запрос...***',
                parse_mode="MARKDOWN")
            completion = await create_response(model='gpt-4-turbo', text=message.text, prompt=system_prompt, client=gpt_client)

            if not completion.choices:
                await message.answer(f'❌ ***{model_title} ничего не вернул***', parse_mode='MARKDOWN')
        
            gpt_answer = completion.choices[0].message.content
            new_gpt_answer = clean_output(clean_markdown(gpt_answer))

            await enable_message.delete()
            await send_long_message(gpt_answer, message)






        if current_model == 'gpt4.1':            
            try:
                print('Начало обработки кода', flush=True)
                
                enable_message = await message.answer(
                    f'🛠️ ***Пожалуйста подождите, {model_title} обрабатывает ваш запрос...***',
                    parse_mode="MARKDOWN")
                
                message_text = message.caption if message.photo else message.text

                print(f'Сообщение "{message_text}", тип: {type(message_text)}', flush=True)


                if isinstance(message_text, list):
                    message_text = " ".join(message_text)


                # Создаем папку har_and_cookies если ее нет и даем права
                os.makedirs("har_and_cookies", exist_ok=True)
                os.chmod("har_and_cookies", 0o755)  # Права на чтение и запись


                completion = await create_response(text=message_text, client=gpt_client, model='gpt-4.1', prompt=system_prompt, img_path=img_path)

                if img_path and os.path.exists(img_path):
                    os.remove(img_path)

                if not completion.choices:
                    await message.answer(f'❌ ***{model_title} ничего не вернул***', parse_mode='MARKDOWN')

                gpt_answer = clean_output(clean_markdown(completion.choices[0].message.content))
                await enable_message.delete()


                await send_long_message(gpt_answer, message)
            
            except PermissionError as e:
                print(f'❌ Ошибка доступа: {str(e)}')
    
            except Exception as e:
                print(f'❌ Произошла ошибка: {str(e)}')









        if current_model == 'gpt4-o':
            try:
                enable_message = await message.answer(
                    f'🛠️ ***Пожалуйста подождите, {model_title} обрабатывает ваш запрос...***',
                    parse_mode="MARKDOWN")


                message_text = message.caption if message.photo else message.text

                print(f'Сообщение "{message_text}", тип: {type(message_text)}', flush=True)

                if isinstance(message_text, list):
                    message_text = " ".join(message_text)


                # Создаем папку har_and_cookies если ее нет и даем права
                os.makedirs("har_and_cookies", exist_ok=True)
                os.chmod("har_and_cookies", 0o755)  # Права на чтение и запись

                completion = await create_response(model='gpt-4o', prompt=system_prompt, client=gpt_client, text=message_text, img_path=img_path)

                if img_path and os.path.exists(img_path):
                    os.remove(img_path)

                if not completion.choices:
                    await message.answer(f'❌ ***{model_title} ничего не вернул***', parse_mode='MARKDOWN')

                gpt_answer = clean_output(clean_markdown(completion.choices[0].message.content))
                await enable_message.delete()

                await send_long_message(gpt_answer, message)

            except PermissionError as e:
                print(f'❌ Ошибка доступа: {str(e)}')
    
            except Exception as e:
                print(f'❌ Произошла ошибка: {str(e)}')




        if current_model == 'gpt4.1-mini':
            if message.photo:
                if img_path and os.path.exists(img_path):
                    os.remove(img_path)

                await message.answer('НЕТ ПОДДЕРЖКИ ФОТО')
                return

            enable_message = await message.answer(
                    f'🛠️ ***Пожалуйста подождите, {model_title} обрабатывает ваш запрос...***',
                    parse_mode="MARKDOWN")

            message_text = message.text

            if isinstance(message_text, list):
                message_text = " ".join(message_text)

            
            completion = await create_response(text=message_text, model='gpt-4.1-mini', prompt=system_prompt, client=gpt_client)


            if not completion.choices:
                await message.answer(f'❌ ***{model_title} ничего не вернул***', parse_mode='MARKDOWN')

            gpt_answer = clean_output(clean_markdown(completion.choices[0].message.content))
            await enable_message.delete()

            await send_long_message(gpt_answer, message)







        if current_model == 'gpt4-o-mini':
            enable_message = await message.answer(
                f'🛠️ ***Пожалуйста подождите, {model_title} обрабатывает ваш запрос...***',
                parse_mode="MARKDOWN")

            completion = await create_response(text=message.text, model='gpt-4o-mini', client=gpt_client, prompt=system_prompt)

            if not completion.choices:
                await message.answer(f'❌ ***{model_title} ничего не вернул***', parse_mode='MARKDOWN')

            gpt_answer = clean_output(clean_markdown(completion.choices[0].message.content))
            await enable_message.delete()

            await send_long_message(gpt_answer, message)








        # CLAUDE family
        if current_model == 'claude3.7-sonnet':
            enable_message = await message.answer(
                f'🛠️ ***Пожалуйста подождите, {model_title} обрабатывает ваш запрос...***',
                parse_mode="MARKDOWN")

            completion = await create_response(text=message.text, model=g4f.models.claude_3_7_sonnet, client=gpt_client, prompt=system_prompt)

            if not completion.choices:
                await message.answer(f'❌ ***{model_title} ничего не вернул***', parse_mode='MARKDOWN')


            claude_answer = clean_output(clean_markdown(completion.choices[0].message.content))
            await enable_message.delete()

            await send_long_message(claude_answer, message)







        if current_model == 'claude3.7-sonnet-thinking':
            enable_message = await message.answer(
                f'🛠️ ***Пожалуйста подождите, {model_title} обрабатывает ваш запрос...***',
                parse_mode="MARKDOWN")

            completion = await create_response(text=message.text, client=gpt_client, model=g4f.models.claude_3_7_sonnet_thinking, prompt=system_prompt)

            if not completion.choices:
                await message.answer(f'❌ ***{model_title} ничего не вернул***', parse_mode='MARKDOWN')


            claude_answer = clean_output(clean_markdown(completion.choices[0].message.content))
            await enable_message.delete()

            await send_long_message(claude_answer, message)






        # OPEN AI FAMILY
        if current_model == 'open-ai-o3':
            enable_message = await message.answer(
                f'🛠️ ***Пожалуйста подождите, {model_title} обрабатывает ваш запрос...***',
                parse_mode="MARKDOWN")

            completion = await create_response(text=message.text, client=client, model='openai/o3', prompt=system_prompt)

            if not completion.choices:
                await message.answer(f'❌ ***{model_title} ничего не вернул***', parse_mode='MARKDOWN')

            oai_answer = clean_output(clean_markdown(completion.choices[0].message.content))
            await enable_message.delete()

            await send_long_message(oai_answer, message)





        if current_model == 'open-ai-o4-mini':
            enable_message = await message.answer(
                f'🛠️ ***Пожалуйста подождите, {model_title} обрабатывает ваш запрос...***',
                parse_mode="MARKDOWN")

            completion = await create_response(text=message.text, client=gpt_client, model=g4f.models.o4_mini, prompt=system_prompt)

            if not completion.choices:
                await message.answer(f'❌ ***{model_title} ничего не вернул***', parse_mode='MARKDOWN')

            oai_answer = clean_output(clean_markdown(completion.choices[0].message.content))
            await enable_message.delete()

            await send_long_message(oai_answer, message)




        # QWEN FAMILY
        if current_model == 'qwen3-235B-A22B':
            enable_message = await message.answer(
                f'🛠️ ***Пожалуйста подождите, {model_title} обрабатывает ваш запрос...***',
                parse_mode="MARKDOWN")

            try:
                completion = await create_response(text=message.text, client=client, model='qwen/qwen3-235b-a22b', prompt=system_prompt)

                if not completion.choices:
                    await message.answer(f'❌ ***{model_title} ничего не вернул***', parse_mode='MARKDOWN')
        
                qwen_answer = clean_output(clean_markdown(completion.choices[0].message.content))
                await enable_message.delete()

                await send_long_message(qwen_answer, message)
                print('qwen 1')
            
            except Exception:
                completion = await create_response(text=message.text, client=gpt_client, model='qwen-3-235b', prompt=system_prompt)

                if not completion.choices:
                    await message.answer(f'❌ ***{model_title} ничего не вернул***', parse_mode='MARKDOWN')
        
                qwen_answer = clean_output(clean_markdown(completion.choices[0].message.content))
                await enable_message.delete()

                await send_long_message(qwen_answer, message)






        if current_model == 'qwen3-30b-a3b':
            enable_message = await message.answer(
                f'🛠️ ***Пожалуйста подождите, {model_title} обрабатывает ваш запрос...***',
                parse_mode="MARKDOWN")

            try:
                completion = await create_response(text=message.text, client=client, model='qwen/qwen3-30b-a3b', prompt=system_prompt)

                if not completion.choices:
                    await message.answer(f'❌ ***{model_title} ничего не вернул***', parse_mode='MARKDOWN')
        
                qwen_answer = clean_output(clean_markdown(completion.choices[0].message.content))
                await enable_message.delete()

                await send_long_message(qwen_answer, message)
            
            except Exception:
                completion = await create_response(text=message.text, client=gpt_client, model='qwen-3-30b', prompt=system_prompt)

                if not completion.choices:
                    await message.answer(f'❌ ***{model_title} ничего не вернул***', parse_mode='MARKDOWN')
        
                qwen_answer = clean_output(clean_markdown(completion.choices[0].message.content))
                await enable_message.delete()

                await send_long_message(qwen_answer, message)



        # GEMINI FAMILY
        if current_model == 'gemini-2.0-flash-lite':
            enable_message = await message.answer(
                f'🛠️ ***Пожалуйста подождите, {model_title} обрабатывает ваш запрос...***',
                parse_mode="MARKDOWN")

            completion = await create_response(text=message.text, client=gpt_client, model=g4f.models.gemini_2_0_flash_thinking, prompt=system_prompt)

            if not completion.choices:
                await message.answer(f'❌ ***{model_title} ничего не вернул***', parse_mode='MARKDOWN')

            gemini_answer = clean_output(clean_markdown(completion.choices[0].message.content))
            await enable_message.delete()
            
            await send_long_message(gemini_answer, message)




    except Exception as e:
        print(e)
        await message.answer('❌ ***Произошла ошибка генерации ответа или модель не работает...***', parse_mode='MARKDOWN')








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
                                               f'<b>🐳 Deepseek-V3</b>* - Китайская текстовая модель, созданая Ляном Вэньфэном\n'
                                               f'<b>⚡ Deepseek-QWEN</b> - Deepseek на базе китайской модели QWEN\n\n'
                                               f'<b>🍓 OpenAI-O3</b> - Рассуждающая модель с наилучшими решениями\n'
                                               f'<b>🧠 OpenAI-O4 mini</b> - Для кодинга и точных наук\n\n'
                                               f'<b>✨ GPT-4 Turbo</b> – Мощная и быстрая модель OpenAI с увеличенным контекстом.\n'
                                               f'<b>💥 PT-4.1</b> – Улучшенная версия GPT-4 с более точными ответами.\n'
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
