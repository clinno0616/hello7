# 創建一個新的 init_db.py 文件來執行數據庫初始化
import os
from database import Database

def initialize_database():
    """Initialize the database with all required tables"""
    # 如果數據庫文件已存在，先刪除它
    db_name = "chat_app.db"
    if os.path.exists(db_name):
        try:
            os.remove(db_name)
            print(f"Removed existing database: {db_name}")
        except Exception as e:
            print(f"Error removing existing database: {e}")
            return False

    try:
        # 創建數據庫實例
        db = Database(db_name)
        
        # 初始化數據庫
        db.init_database()
        
        print("Database initialized successfully!")
        return True
    except Exception as e:
        print(f"Error initializing database: {e}")
        return False

if __name__ == "__main__":
    clear = lambda: os.system('cls')
    clear()
    initialize_database()

