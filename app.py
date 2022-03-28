import os
from flask import Flask, request, render_template

from PIL import Image
import torch

from engine import get_model
from utils import test_augmentation


ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = get_model(device)
model.eval()


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
        print(out / sum(out), sum(out / sum(out)))
        out -= out.min()
        print(out)
        out /= sum(out)
        print(out, sum(out))

        return render_template('index.html', result=out, img_path=img_path)


if __name__ == '__main__':
    app.run(debug=True)
