from aiogram.utils.keyboard import InlineKeyboardBuilder, InlineKeyboardButton, InlineKeyboardMarkup





def main_keyboard():
    builder = InlineKeyboardBuilder()
    builder.add(
        InlineKeyboardButton(text='🧑‍💻 Выбор модели', callback_data='change_model'),
        InlineKeyboardButton(text='🛠️ Канал прогера', url='https://t.me/+1A9f6ZFMJBgxMjRi'),
    )
    return builder.as_markup()
