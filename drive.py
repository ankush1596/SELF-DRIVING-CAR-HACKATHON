import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2

# Initialize socketio server
sio = socketio.Server()

# Initialize Flask app
app = Flask(__name__)

# Define speed limit
speed_limit = 10

# Image preprocessing function
def img_preprocess(img):
    img = img[60:135, :, :]  # Crop the image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # Convert to YUV
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Apply Gaussian blur
    img = cv2.resize(img, (200, 66))  # Resize image
    img = img / 255.0  # Normalize
    return img

# Connect event handler
@sio.on('connect')
def connect(sid, environ):
    print('connected')
    send_control(0, 0)

# Send control data to car
def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })

# Telemetry event handler
@sio.on('telementry')
def telementry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed / speed_limit
    print(f'{steering_angle} {throttle} {speed}')
    send_control(steering_angle, throttle)

# Main function to load the model and run the app
if __name__ == '__main__':
    model = load_model(r"C:\self driving\model\model.h5")  # Ensure to use a raw string or forward slashes
    app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)  # Wrap Flask app with SocketIO middleware
    app.run(host='0.0.0.0', port=5000)