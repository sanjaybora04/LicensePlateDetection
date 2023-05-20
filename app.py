from flask import Flask, request
from flask_cors import CORS
import cv2
import numpy as np
import base64
import model

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST'])
def post_handler():
    image_data = request.form['imageData']
    image_data = image_data.split(',')[1]  # Remove the data URL prefix (e.g., "data:image/png;base64,")
    
    # Convert the base64-encoded image data to a numpy array
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result = model.detect(image)["image"]

    # Encode the image array as base64
    _, img_encoded = cv2.imencode('.jpg', result)
    image = base64.b64encode(img_encoded).decode('utf-8')

    return {'result':image}

if __name__ == '__main__':
    app.run(port=5000)