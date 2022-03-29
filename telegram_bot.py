secret_key = "your secret token"

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from engine import get_model
from utils import test_augmentation

import os
import torch
from PIL import Image


if not os.path.exists('static'):
    os.makedirs('static')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = get_model(device)
fmt_str = '{:.1%}'


def start(updater, _):
    updater.message.reply_text("Пожалуйста, отправьте фото собаки")


def procces_foto(updater, context):
    img = updater.message.photo[-1].get_file()
    path = os.path.join('static', img.file_unique_id + '.jpg')
    img.download(path)

    img = Image.open(path)
    img = test_augmentation(img).to(device)

    with torch.no_grad():
        out = model(img[None, :])[0].to('cpu')

    out[out < 0] = 0
    out -= out.min()
    out /= sum(out)
    out = out.softmax(0).tolist()

    answer = ""

    for i in sorted(out, reverse=True)[:3]:
        if i == 0:
            break
        class_name = model.classes[out.index(i)].replace('_', ' ').title()
        answer += f"{class_name} - {fmt_str.format(i)}\n"

    updater.message.reply_text(answer)


updater = Updater(secret_key)

updater.dispatcher.add_handler(CommandHandler("start", start))

updater.dispatcher.add_handler(MessageHandler(Filters.photo, procces_foto))

updater.start_polling()
updater.idle()
