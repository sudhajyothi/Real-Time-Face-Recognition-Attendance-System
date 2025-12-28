import sqlite3
import os

db_path = r"C:\Users\sudha\Downloads\Real-Time-Face-Detection-with-Opencv-and-Flask-main\Real-Time-Face-Detection-with-Opencv-and-Flask-main\attendance.db"

# Check if file exists
if not os.path.exists(db_path):
    print("Database file does not exist.")
else:
    print("Database file found.")

# Connect to database
conn = sqlite3.connect(db_path)
c = conn.cursor()

# List all tables
c.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = c.fetchall()
print("Tables:", tables)

# Check table structure
if ('attendance',) in tables:
    c.execute("PRAGMA table_info(attendance);")
    print("Attendance table structure:", c.fetchall())
else:
    print("Attendance table not found!")

conn.close()
