import os
from flask import Flask, request, render_template

from PIL import Image
import torch

from engine import get_model
from utils import test_augmentation


ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']

if not os.path.exists('static'):
    os.makedirs('static')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = get_model(device)
model.eval()

fmt_str = '{:.1%}'


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['file']
        if file.filename.split('.')[-1].lower() not in ALLOWED_EXTENSIONS:
            return render_template('index.html', error=True)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(img_path)

        img = Image.open(file.stream)
        img = test_augmentation(img).to(device)
        out = model(img[None, :])[0].to('cpu')
        out[out < 0] = 0
        out -= out.min()
        out /= sum(out)
        out = out.softmax(0).tolist()

        answer = []
        for i in sorted(out, reverse=True)[:3]:
            class_name = model.classes[out.index(i)].replace('_', ' ').title()
            answer.append(f"{class_name} - {fmt_str.format(i)}")

        return render_template('index.html', result=answer, img_path=img_path)


if __name__ == '__main__':
    app.run(debug=True)
