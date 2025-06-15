from flask import Flask, request, send_file
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

def convert_to_sketch(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    inv_blur = 255 - blur
    sketch = cv2.divide(gray, inv_blur, scale=256.0)
    return sketch

def convert_to_cartoon(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(image, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

@app.route('/convert', methods=['POST'])
def convert_image():
    file = request.files['image']
    effect = request.form.get('effect')

    image = Image.open(file.stream)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    if effect == 'sketch':
        output = convert_to_sketch(image)
    elif effect == 'grayscale':
        output = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif effect == 'blur':
        output = cv2.GaussianBlur(image, (15, 15), 0)
    elif effect == 'cartoon':
        output = convert_to_cartoon(image)
    else:
        return 'Invalid effect', 400

    _, buffer = cv2.imencode('.png', output)
    io_buf = io.BytesIO(buffer)

    return send_file(io_buf, mimetype='image/png')

# IMPORTANT: This line must use host='0.0.0.0'
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=10000)
