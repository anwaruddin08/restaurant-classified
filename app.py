import os

from flask import Flask, render_template, request, redirect,jsonify

from inference import get_prediction
from commons import format_class_name

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_name': class_name})


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
