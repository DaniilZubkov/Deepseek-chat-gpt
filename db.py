import sqlite3
import time


db = sqlite3.connect('database.db')

class Database:
    def __init__(self, db_file):
        self.connection = sqlite3.connect(db_file)
        self.cursor = self.connection.cursor()

    def create_tables(self):
        cursor = self.connection.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY NOT NULL,
        user_id INTEGER NOT NULL,
        nickname TEXT,
        signup VARCHAR(60) DEFAULT 'setnickname',
        model TEXT DEFAULT 'deepseek-r1'
        )""")

    def add_user(self, user_id):
        with self.connection:
            return self.cursor.execute("INSERT INTO users (user_id) VALUES (?)", (user_id,))

    def user_exists(self, user_id):
        with self.connection:
            result = self.cursor.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchall()
            return bool(len(result))

    def set_nickname(self, user_id, nickname):
        with self.connection:
            return self.cursor.execute("UPDATE users SET nickname=? WHERE user_id=?", (nickname, user_id,))

    def get_signup(self, user_id):
        with self.connection:
            result = self.cursor.execute("SELECT signup FROM users WHERE user_id=?", (user_id,)).fetchall()
            for row in result:
                signup = str(row[0])
            return signup

    def set_signup(self, user_id, signup):
        with self.connection:
            return self.cursor.execute("UPDATE users SET signup=? WHERE user_id=?", (signup, user_id,))

    def get_nickname(self, user_id):
        with self.connection:
            result = self.cursor.execute("SELECT nickname FROM users WHERE user_id=?", (user_id,)).fetchall()
            for row in result:
                nickname = str(row[0])
            return nickname



    # MODELS
    def set_model(self, user_id, model):
        with self.connection:
            return self.cursor.execute("UPDATE users SET model=? WHERE user_id=?", (model, user_id,))


    def get_model(self, user_id):
        with self.connection:
            result = self.cursor.execute("SELECT model FROM users WHERE user_id=?", (user_id,)).fetchall()
            for row in result:
                model = str(row[0])
            return model
