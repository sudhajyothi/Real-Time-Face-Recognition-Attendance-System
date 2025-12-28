from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import sqlite3
import pickle
import datetime
import os

app = Flask(__name__)
socketio = SocketIO(app)

# ------------------------------
# Load LBPH model and label dictionary
# ------------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("lbph_model.yml")
with open("labels.pkl", "rb") as f:
    label_dict = pickle.load(f)

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Frame skipping for performance
frame_skip = 3
frame_counter = 0

# ------------------------------
# Initialize database if not exists
# ------------------------------
DB_FILE = 'attendance.db'

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ------------------------------
# Function to log attendance
# ------------------------------
def mark_attendance(name):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    now = datetime.datetime.now()
    today = now.date()

    # Check if the person has already been marked today
    cursor.execute("SELECT * FROM attendance WHERE name=? AND DATE(timestamp)=?", (name, today))
    result = cursor.fetchone()

    if not result:
        # Insert new attendance record
        cursor.execute("INSERT INTO attendance (name, timestamp) VALUES (?, ?)", (name, now))
        conn.commit()
    else:
        # Optional: update timestamp if already exists
        cursor.execute("UPDATE attendance SET timestamp=? WHERE name=? AND DATE(timestamp)=?", (now, name, today))
        conn.commit()

    conn.close()

# ------------------------------
# Flask route
# ------------------------------
@app.route('/')
def dashboard():
    return render_template('admin.html')

# ------------------------------
# SocketIO event: receive image frames
# ------------------------------
@socketio.on('image')
def handle_image(data_image):
    global frame_counter
    frame_counter += 1
    if frame_counter % frame_skip != 0:
        return

    # Decode image from base64
    img_data = data_image.split(",")[1]
    img = base64.b64decode(img_data)
    npimg = np.frombuffer(img, dtype=np.uint8)
    frame = cv2.imdecode(npimg, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)

        if conf < 80:
            name = label_dict.get(id_, "Unknown")
            mark_attendance(name)
        else:
            name = "Unknown"

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Fetch today's attendance
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    today = datetime.datetime.now().date()
    cursor.execute("SELECT id, name, timestamp FROM attendance WHERE DATE(timestamp)=? ORDER BY timestamp DESC", (today,))
    today_records = cursor.fetchall()

    # Fetch all attendance
    cursor.execute("SELECT id, name, timestamp FROM attendance ORDER BY timestamp DESC")
    all_records = cursor.fetchall()
    conn.close()

    # Encode frame to send to client
    _, buffer = cv2.imencode('.jpg', frame)
    frame_data = base64.b64encode(buffer).decode('utf-8')

    emit('response_back', {
        'image': 'data:image/jpeg;base64,' + frame_data,
        'today_attendance': today_records,
        'all_attendance': all_records
    })

# ------------------------------
# Run app
# ------------------------------
if __name__ == '__main__':
    socketio.run(app, debug=True)
