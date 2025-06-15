from flask import Flask, request, send_file
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

def apply_effect(image, effect):
    if effect == "sketch":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    
    elif effect == "grayscale":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    elif effect == "blur":
        return cv2.GaussianBlur(image, (15, 15), 0)
    
    elif effect == "cartoon":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255,
                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(image, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cartoon
    
    else:
        return image

@app.route('/convert', methods=['POST'])
def convert():
    try:
        file = request.files['image']
        effect = request.form.get('effect')

        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)

        processed = apply_effect(image, effect)

        # Convert result to sendable PNG
        if len(processed.shape) == 2:  # grayscale
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        else:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        result = Image.fromarray(processed)
        output = io.BytesIO()
        result.save(output, format='PNG')
        output.seek(0)
        return send_file(output, mimetype='image/png')
    except Exception as e:
        print("Error:", e)
        return "Failed to convert image", 500

if __name__ == '__main__':
    app.run(debug=True)
