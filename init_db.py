import sqlite3
import os

# Ensure database is created in current folder
db_path = os.path.join(os.getcwd(), 'attendance.db')

conn = sqlite3.connect(db_path)
c = conn.cursor()

# Create attendance table
c.execute('''
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

conn.commit()
conn.close()

print(f"Database created at {db_path}")
