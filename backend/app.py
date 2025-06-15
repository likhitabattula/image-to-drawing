from flask import Flask, request, send_file
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

def apply_effect(image, effect):
    if effect == 'sketch':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    elif effect == 'grayscale':
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    elif effect == 'blur':
        return cv2.GaussianBlur(image, (15, 15), 0)

    elif effect == 'cartoon':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255,
                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(image, 9, 250, 250)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cartoon

    else:
        return image

@app.route('/convert', methods=['POST'])
def convert():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    file = request.files['image']
    effect = request.form.get('effect', 'sketch')

    in_memory_file = file.read()
    npimg = np.frombuffer(in_memory_file, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    output = apply_effect(image, effect)
    if len(output.shape) == 2:  # grayscale
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

    _, buffer = cv2.imencode('.png', output)
    io_buf = io.BytesIO(buffer)

    return send_file(io_buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
