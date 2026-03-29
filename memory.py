import sqlite3

DB_NAME = "memory.db"

def init_db():
    conn = conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        role TEXT,
        content TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


def save_message(user_id, role, content):
    conn = conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    c = conn.cursor()

    c.execute("""
    INSERT INTO conversations (user_id, role, content)
    VALUES (?, ?, ?)
    """, (user_id, role, content))

    conn.commit()
    conn.close()


def get_history(user_id, limit=10):
    conn = conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    c = conn.cursor()

    c.execute("""
    SELECT role, content FROM conversations
    WHERE user_id=?
    ORDER BY id DESC
    LIMIT ?
    """, (user_id, limit))

    rows = c.fetchall()
    conn.close()

    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]