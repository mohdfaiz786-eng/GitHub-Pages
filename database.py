# database.py - Database Operations

import sqlite3
import hashlib
import json
import os

class Database:
    def __init__(self):
        self.db_path = os.path.join(os.path.dirname(__file__), "users.db")
        self.init_db()
    
    def init_db(self):
        """Initialize database with proper schema"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Users table
        c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            email TEXT,
            full_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            login_count INTEGER DEFAULT 0
        )
        """)
        
        # Predictions table
        c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            risk_level TEXT,
            risk_percentage REAL,
            model_used TEXT,
            patient_data TEXT
        )
        """)
        
        conn.commit()
        conn.close()
    
    def hash_pwd(self, pwd):
        """Hash password"""
        return hashlib.sha256((pwd + "cardioai_salt_2024").encode()).hexdigest()
    
    def add_user(self, username, password, email, full_name):
        """Add new user"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            c.execute(
                "INSERT INTO users (username, password, email, full_name) VALUES (?,?,?,?)",
                (username, self.hash_pwd(password), email, full_name)
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()
    
    def verify_user(self, username, password):
        """Verify user credentials"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username=?", (username,))
        data = c.fetchone()
        conn.close()
        
        if data and data[0] == self.hash_pwd(password):
            self.update_login_count(username)
            return True
        return False
    
    def update_login_count(self, username):
        """Update login count"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "UPDATE users SET login_count = login_count + 1 WHERE username=?",
            (username,)
        )
        conn.commit()
        conn.close()
    
    def get_user_stats(self, username):
        """Get user statistics"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "SELECT full_name, email, login_count, created_at FROM users WHERE username=?",
            (username,)
        )
        data = c.fetchone()
        conn.close()
        return data
    
    def user_exists(self, username):
        """Check if user exists"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT username FROM users WHERE username=?", (username,))
        data = c.fetchone()
        conn.close()
        return data is not None
    
    def save_prediction(self, username, risk_level, risk_percentage, model_used, patient_data):
        """Save prediction to database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            c.execute(
                """INSERT INTO predictions 
                   (username, risk_level, risk_percentage, model_used, patient_data) 
                   VALUES (?,?,?,?,?)""",
                (username, risk_level, risk_percentage, model_used, json.dumps(patient_data))
            )
            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving prediction: {e}")
            return False
        finally:
            conn.close()
    
    def get_user_predictions(self, username, limit=50):
        """Get user's prediction history"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            """SELECT id, date, risk_level, risk_percentage, model_used, patient_data 
               FROM predictions WHERE username=? ORDER BY date DESC LIMIT ?""",
            (username, limit)
        )
        data = c.fetchall()
        conn.close()
        return data