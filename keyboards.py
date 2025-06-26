from aiogram.utils.keyboard import InlineKeyboardBuilder, InlineKeyboardButton, InlineKeyboardMarkup





def main_keyboard():
    builder = InlineKeyboardBuilder()
    builder.add(
        InlineKeyboardButton(text='📄 GPT/CLAUDE/GEMINI', callback_data='change_model'), 
        InlineKeyboardButton(text='🎨 СОЗДАТЬ КАРТИНКУ', callback_data='change_model_photo_categ'),
        InlineKeyboardButton(text='️🛠️ Канал прогера', url='https://t.me/+1A9f6ZFMJBgxMjRi'),
    )
    builder.adjust(2)
    return builder.as_markup()
