#pip install ldap3
import sqlite3
import hashlib
from cryptography.fernet import Fernet
from datetime import datetime,timedelta
import os
from ldap3 import Server, Connection, SUBTREE, ALL, SIMPLE
import logging
from typing import Tuple
import uuid
import urllib.parse
import socket

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
import smtplib

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def get_verification_url(token: str) -> str:
    """Generate verification URL"""
    base_url = os.getenv('VERIFICATION_URL', 'http://192.168.6.111:8501')
    # Instead of /verify, use query parameter 'page=verify'
    return f"{base_url}/?page=verify&token={urllib.parse.quote(token)}"

class EmailVerification:
    """Email verification class using simplified Gmail SMTP configuration with improved testing"""
    
    def __init__(self, gmail_user=None, gmail_app_password=None):
        """Initialize email verification with Gmail configurations"""
        self.gmail_user = gmail_user or os.getenv('GMAIL_USER')
        self.gmail_app_password = gmail_app_password or os.getenv('GMAIL_APP_PASSWORD')
        
        # Debug 資訊記錄
        logger.info("Gmail Configuration:")
        logger.info(f"Gmail User: {self.gmail_user if self.gmail_user else 'Not set'}")
        logger.info(f"App Password Set: {'Yes' if self.gmail_app_password else 'No'}")
        
        # 驗證設定
        self._verify_config()
        
    def _verify_config(self) -> bool:
        """驗證 Gmail 設定並測試連線"""
        # 檢查必要設定
        if not all([self.gmail_user, self.gmail_app_password]):
            missing = []
            if not self.gmail_user: missing.append("GMAIL_USER")
            if not self.gmail_app_password: missing.append("GMAIL_APP_PASSWORD") 
            logger.error(f"Missing Gmail configurations: {', '.join(missing)}")
            return False
            
        # 測試 SMTP 連線，設定超時時間
        try:
            with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as server:
                server.ehlo()  # 使用 EHLO 替代 HELO
                server.starttls()
                # 設定較短的登入超時
                server.login(self.gmail_user, self.gmail_app_password)
                logger.info("Gmail connection test successful")
                return True
                
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"Gmail authentication failed: {str(e)}")
            return False
        except socket.timeout:
            logger.error("Gmail connection timed out")
            return False
        except Exception as e:
            logger.error(f"Gmail connection test failed: {str(e)}")
            return False
            
    def test_connection(self) -> dict:
        """測試 Gmail 連線狀態並返回詳細資訊"""
        status = {
            "success": False,
            "connection": False,
            "auth": False,
            "error": None
        }
        
        try:
            # 1. 首先測試網路連線
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            status["connection"] = True
            
            # 2. 測試 SMTP 連線
            with smtplib.SMTP("smtp.gmail.com", 587, timeout=5) as server:
                server.ehlo()
                server.starttls()
                
                # 3. 測試認證
                server.login(self.gmail_user, self.gmail_app_password)
                status["auth"] = True
                status["success"] = True
                
        except socket.timeout:
            status["error"] = "Connection timed out"
        except smtplib.SMTPAuthenticationError:
            status["error"] = "Authentication failed"
        except Exception as e:
            status["error"] = str(e)
            
        return status

    def send_verification_email(self, recipient_email: str, verification_token: str) -> bool:
        """Send verification email using Gmail with improved error handling"""
        try:
            if not self._verify_config():
                return False
                
            verification_url = get_verification_url(verification_token)
            message = self._create_email_message(recipient_email, verification_url)
            
            # 使用較短的超時時間
            with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as server:
                server.ehlo()
                server.starttls()
                server.login(self.gmail_user, self.gmail_app_password)
                server.send_message(message)
                logger.info(f"Verification email sent successfully to {recipient_email}")
                return True
                
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"Gmail Authentication failed: {str(e)}")
            return False
        except socket.timeout:
            logger.error("Connection timed out while sending email")
            return False
        except Exception as e:
            logger.error(f"Failed to send verification email: {str(e)}")
            return False
            
    def _create_email_message(self, recipient_email: str, verification_url: str) -> MIMEMultipart:
        """Create email message with both plain text and HTML versions"""
        message = MIMEMultipart("alternative")
        message["From"] = self.gmail_user
        message["To"] = recipient_email
        message["Subject"] = "Verify Your Chat Application Account"
        
        # Plain text version
        text_content = f"""
        Welcome to the Chat Application!
        
        Please verify your email address by clicking the following link:
        {verification_url}
        
        This link will expire in 24 hours.
        """
        message.attach(MIMEText(text_content, "plain"))
        
        # HTML version
        html_content = f"""
        <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <h2 style="color: #2c3e50;">Welcome to the Chat Application!</h2>
                    <p>Please verify your email address by clicking the button below:</p>
                    <div style="text-align: center; margin: 30px 0;">
                        <a href="{verification_url}" 
                           style="background-color: #3498db; color: white; padding: 12px 24px; 
                                  text-decoration: none; border-radius: 4px; display: inline-block;">
                            Verify Email Address
                        </a>
                    </div>
                    <p style="font-size: 14px; color: #666;">
                        Or copy and paste this link:<br>
                        <span style="color: #3498db;">{verification_url}</span>
                    </p>
                    <p style="font-size: 12px; color: #666;">
                        This link will expire in 24 hours.
                    </p>
                </div>
            </body>
        </html>
        """
        message.attach(MIMEText(html_content, "html"))
        return message


class ADConfig:
    def __init__(self):
        # LDAP Server Configuration
        self.ldap_server = os.getenv('LDAP_SERVER')
        self.ldap_port = os.getenv('LDAP_PORT')
        self.use_starttls = False
        self.root_dn = os.getenv('ROOT_DN')
        self.organization = ' '
        self.uid_field = 'sAMAccountName'
        
        # Bind Account Credentials
        self.bind_dn = os.getenv('BIND_DN')
        self.bind_password = os.getenv('BIND_PASSWORD')
        
    def validate_credentials(self, username, password):
        try:
            # Initialize server
            server = Server(
                self.ldap_server,
                port=self.ldap_port,
                get_info=ALL
            )
            
            logger.debug(f"Connecting to LDAP server: {self.ldap_server}")
            
            # First bind with service account
            service_conn = Connection(
                server,
                user=self.bind_dn,
                password=self.bind_password,
                authentication=SIMPLE
            )
            
            if not service_conn.bind():
                logger.error("Service account bind failed")
                return False
                
            logger.debug("Service account bind successful")
            
            # Search for user
            search_filter = f"({self.uid_field}={username})"
            logger.debug(f"Searching with filter: {search_filter}")
            
            service_conn.search(
                search_base=self.root_dn,
                search_filter=search_filter,
                search_scope=SUBTREE,
                attributes=[self.uid_field]
            )
            
            if not service_conn.entries:
                logger.error("User not found in AD")
                return False
                
            user_dn = service_conn.entries[0].entry_dn
            logger.debug(f"Found user DN: {user_dn}")
            
            # Try binding with user credentials
            user_conn = Connection(
                server,
                user=user_dn,
                password=password,
                authentication=SIMPLE
            )
            
            if user_conn.bind():
                logger.debug("User authentication successful")
                return True
            
            logger.error("User authentication failed")
            return False
            
        except Exception as e:
            logger.error(f"AD Authentication error: {str(e)}")
            return False


class Database:
    def __init__(self, db_name="chat_app.db"):
        self.db_name = db_name
        self.ad_config = ADConfig()
        self.email_verification = EmailVerification()
        #self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # Drop existing tables if they exist
        cursor.execute("DROP TABLE IF EXISTS roles")
        cursor.execute("DROP TABLE IF EXISTS conversations")
        cursor.execute("DROP TABLE IF EXISTS uploaded_files")
        cursor.execute("DROP TABLE IF EXISTS chat_sessions")
        cursor.execute("DROP TABLE IF EXISTS users")
        cursor.execute("DROP TABLE IF EXISTS chroma_collections")
        cursor.execute("DROP TABLE IF EXISTS uploaded_images")
        cursor.execute("DROP TABLE IF EXISTS document_chunks")
        cursor.execute("DROP TABLE IF EXISTS conversation_sources")
        
        # Create roles table
        cursor.execute('''
        CREATE TABLE roles (
            role_id INTEGER PRIMARY KEY,
            role_name TEXT NOT NULL UNIQUE
        )
        ''')

        # Insert default roles
        cursor.execute('''
        INSERT INTO roles (role_id, role_name) VALUES 
            (1, 'admin'),
            (2, 'manager'),
            (3, 'user')
        ''')

        # Create users table with role_id
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT UNIQUE,
            is_verified INTEGER DEFAULT 0,
            openai_api_key TEXT,
            anthropic_api_key TEXT,
            role_id INTEGER DEFAULT 3,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (role_id) REFERENCES roles(role_id)
        )
        ''')
        
        # Create admin user
        admin_password_hash = self.hash_password('admin123')
        cursor.execute('''
        INSERT INTO users (username, password_hash, role_id, is_verified, email)
        VALUES ('admin', ?, 1, 1, 'admin@example.com')
        ''', (admin_password_hash,))


        # Create test user
        password_hash = self.hash_password('1234')
        cursor.execute('''
        INSERT INTO users (username, password_hash, role_id, is_verified, email)
        VALUES ('sa', ?, 3, 1, 'sa@example.com')
        ''', (password_hash,))
        
        cursor.execute('''
        INSERT INTO users (username, password_hash, role_id, is_verified, email)
        VALUES ('test', ?, 3, 0, 'test@example.com')
        ''', (password_hash,))

        # Create chat_sessions table
        cursor.execute('''
        CREATE TABLE chat_sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            api_source TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        ''')
        
        # Create conversations table with session_id and hidden flag
        cursor.execute('''
        CREATE TABLE conversations (
            conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            api_source TEXT NOT NULL,
            user_message TEXT NOT NULL,
            assistant_response TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            hidden BOOLEAN DEFAULT FALSE,
            FOREIGN KEY (user_id) REFERENCES users (user_id),
            FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id)
        )
        ''')

        # Create uploaded_files table
        cursor.execute('''
        CREATE TABLE uploaded_files (
            file_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_id INTEGER NOT NULL,
            original_filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_type TEXT NOT NULL,
            upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id),
            FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id)
        )
        ''')

        # Add new table for ChromaDB collections
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chroma_collections (
            session_id INTEGER PRIMARY KEY,
            collection_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id)
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS uploaded_images (
        image_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        session_id INTEGER NOT NULL,
        original_filename TEXT NOT NULL,
        file_path TEXT NOT NULL,
        file_type TEXT NOT NULL,
        upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (user_id),
        FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id)
        )
        ''')
        
        # 新增 chunks tracking table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_chunks (
        chunk_id TEXT PRIMARY KEY,
        chunk_name TEXT NOT NULL,
        file_name TEXT NOT NULL,
        session_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id),
        FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversation_sources (
        source_id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER NOT NULL,
        source_file TEXT NOT NULL,
        chunk_number INTEGER NOT NULL,
        chunk_content TEXT NOT NULL,
        distance REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
        )
        ''')

        # Create verification tokens table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS verification_tokens (
            token_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            is_used INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
        ''')

        conn.commit()
        conn.close()
    
    def hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, username: str, password: str, email: str) -> Tuple[int, str]:
        """Create new user with improved datetime handling"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            # Check existing email
            cursor.execute("SELECT user_id FROM users WHERE email = ?", (email,))
            if cursor.fetchone():
                return None, "Email already registered"
            
            # Create user
            password_hash = self.hash_password(password)
            cursor.execute(
                """INSERT INTO users (username, password_hash, email, is_verified)
                VALUES (?, ?, ?, ?)""",
                (username, password_hash, email, 0)  # Use 0 instead of False
            )
            
            user_id = cursor.lastrowid
            
            # Generate verification token
            verification_token = str(uuid.uuid4())
            expires_at = datetime.utcnow() + timedelta(hours=24)
            
            # Format datetime without microseconds
            expires_at_str = expires_at.strftime('%Y-%m-%d %H:%M:%S')
            
            # Save token
            cursor.execute(
                """INSERT INTO verification_tokens (user_id, token, expires_at)
                VALUES (?, ?, ?)""",
                (user_id, verification_token, expires_at_str)
            )
            
            conn.commit()
            
            # Send verification email
            if not self.email_verification.send_verification_email(email, verification_token):
                logger.warning(f"Failed to send verification email to {email}")
                return user_id, "Account created but verification email failed to send. Please contact support."
            
            return user_id, "Sending Verification Email successfully! Please check your email to verify your account."
            
        except sqlite3.IntegrityError as e:
            if "username" in str(e):
                return None, "Username already exists"
            return None, "Registration failed"
        except Exception as e:
            logger.error(f"Error in create_user: {str(e)}")
            return None, "Registration failed"
        finally:
            conn.close()

    def verify_user(self, username, password, auth_type='local'):
        """
        Verify user credentials and verification status
        Returns: Tuple[Optional[int], str] - (user_id, message) or (None, error_message)
        """
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            logger.debug(f"Attempting {auth_type} authentication for user: {username}")
            
            if auth_type == 'local':
                password_hash = self.hash_password(password)
                
                # First check if user exists and get their verification status
                cursor.execute(
                    """SELECT user_id, username, role_id, is_verified, email 
                    FROM users 
                    WHERE username = ?""",
                    (username,)
                )
                user_result = cursor.fetchone()
                
                if not user_result:
                    logger.debug(f"User not found: {username}")
                    return None, "Invalid username or password"
                    
                user_id, db_username, role_id, is_verified, email = user_result
                
                # Now check password
                cursor.execute(
                    """SELECT user_id 
                    FROM users 
                    WHERE username = ? AND password_hash = ?""",
                    (username, password_hash)
                )
                auth_result = cursor.fetchone()
                
                if not auth_result:
                    logger.debug(f"Invalid password for user: {username}")
                    return None, "Invalid username or password"
                
                # Check verification status
                if not bool(is_verified):
                    logger.debug(f"Unverified user attempt to login: {username}")
                    return None, f"Please verify your email address to login. Check {email} for the verification link."
                
                logger.debug(f"User successfully authenticated: {username}")
                return user_id, "Login successful"
                
            elif auth_type == 'ad':
                if self.ad_config.validate_credentials(username, password):
                    cursor.execute(
                        """SELECT user_id, is_verified 
                        FROM users 
                        WHERE username = ?""",
                        (username,)
                    )
                    result = cursor.fetchone()
                    
                    if result:
                        user_id, is_verified = result
                        if not bool(is_verified):
                            return None, "Please contact administrator to verify your account."
                        return user_id, "Login successful"
                    else:
                        # Create new AD user as verified by default
                        try:
                            cursor.execute(
                                """INSERT INTO users 
                                (username, password_hash, role_id, is_verified) 
                                VALUES (?, 'AD_USER', 3, 1)""",
                                (username,)
                            )
                            print(username)
                        except Exception as e:
                            print(f"錯誤: {str(e)}")
                        #finally:
                        #    input("按任意鍵繼續...") # 這會顯示暫停提示並等待使用者按下任意鍵
                        conn.commit()
                        return cursor.lastrowid, "Login successful"
                        
                return None, "Invalid credentials"
                
        except Exception as e:
            logger.error(f"Database error during authentication: {str(e)}")
            return None, f"Authentication error: {str(e)}"
        finally:
            conn.close()


    """
    def verify_user(self, username, password):
        #Verify user credentials
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        password_hash = self.hash_password(password)
        cursor.execute(
            "SELECT user_id FROM users WHERE username = ? AND password_hash = ?",
            (username, password_hash)
        )
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    
    def save_api_key(self, user_id, api_type, api_key):
        #Save API key for a user
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        column = f"{api_type}_api_key"
        cursor.execute(
            f"UPDATE users SET {column} = ? WHERE user_id = ?",
            (api_key, user_id)
        )
        conn.commit()
        conn.close()
    

    def get_api_key(self, user_id, api_type):
        #Get API key for a user
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        column = f"{api_type}_api_key"
        cursor.execute(
            f"SELECT {column} FROM users WHERE user_id = ?",
            (user_id,)
        )
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    """
        
    def create_chat_session(self, user_id, api_source):
        """Create a new chat session and return its ID"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO chat_sessions (user_id, api_source)
            VALUES (?, ?)""",
            (user_id, api_source)
        )
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return session_id
    
    def save_conversation(self, session_id, user_id, api_source, user_message, assistant_response):
        """Save a conversation and return updated conversation list for the session"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # Save the new conversation
        cursor.execute(
            """INSERT INTO conversations 
            (session_id, user_id, api_source, user_message, assistant_response)
            VALUES (?, ?, ?, ?, ?)""",
            (session_id, user_id, api_source, user_message, assistant_response)
        )
        conn.commit()
        
        # Retrieve updated conversations for this session
        cursor.execute(
            """SELECT 
                conversation_id,
                session_id,
                user_message,
                assistant_response,
                timestamp
            FROM conversations 
            WHERE session_id = ? AND user_id = ?
            ORDER BY timestamp ASC""",
            (session_id, user_id)
        )
        updated_conversations = cursor.fetchall()
        
        conn.close()
        return updated_conversations

    def get_user_conversations(self, user_id):
        """Get all conversations for a user grouped by session"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute(
            """SELECT 
                s.session_id,
                c.api_source,
                c.user_message,
                c.assistant_response,
                c.timestamp,
                s.created_at
            FROM chat_sessions s
            JOIN conversations c ON s.session_id = c.session_id
            WHERE s.user_id = ? AND c.hidden = FALSE
            ORDER BY s.created_at DESC, c.timestamp ASC""",
            (user_id,)
        )
        conversations = cursor.fetchall()
        conn.close()
        return conversations

    def save_uploaded_file(self, user_id, session_id, original_filename, file_path, file_type):
        """Save uploaded file information"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO uploaded_files 
            (user_id, session_id, original_filename, file_path, file_type)
            VALUES (?, ?, ?, ?, ?)""",
            (user_id, session_id, original_filename, file_path, file_type)
        )
        file_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return file_id

    def get_session_files(self, session_id):
        """Get all files uploaded in a specific chat session"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute(
            """SELECT file_id, original_filename, file_path, file_type, upload_timestamp 
            FROM uploaded_files 
            WHERE session_id = ? 
            ORDER BY upload_timestamp DESC""",
            (session_id,)
        )
        files = cursor.fetchall()
        conn.close()
        return files

    def get_file_by_id(self, file_id):
        """Get file information by file ID"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute(
            """SELECT original_filename, file_path, file_type 
            FROM uploaded_files 
            WHERE file_id = ?""",
            (file_id,)
        )
        file_info = cursor.fetchone()
        conn.close()
        return file_info if file_info else None

    def hide_conversation(self, conversation_id):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("UPDATE conversations SET hidden = TRUE WHERE conversation_id = ?", (conversation_id,))
        conn.commit()
        conn.close()

    def unhide_conversation(self, conversation_id):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("UPDATE conversations SET hidden = FALSE WHERE conversation_id = ?", (conversation_id,))
        conn.commit()
        conn.close()

    def remove_file(self, file_id):
        """Remove a file from the database"""
        try:
            conn = sqlite3.connect(self.db_name)
            conn.execute(
                "DELETE FROM uploaded_files WHERE file_id = ?", 
                (file_id,)
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error removing file: {e}")
            return False

    def save_collection_name(self, session_id, collection_name):
        """Save ChromaDB collection name for a session"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            # Check if entry exists
            cursor.execute(
                "SELECT collection_name FROM chroma_collections WHERE session_id = ?",
                (session_id,)
            )
            existing = cursor.fetchone()
            
            if existing:
                # Update existing entry
                cursor.execute(
                    """UPDATE chroma_collections 
                    SET collection_name = ? 
                    WHERE session_id = ?""",
                    (collection_name, session_id)
                )
            else:
                # Insert new entry
                cursor.execute(
                    """INSERT INTO chroma_collections (session_id, collection_name) 
                    VALUES (?, ?)""",
                    (session_id, collection_name)
                )
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving collection name: {e}")
            return False
        finally:
            conn.close()
        
    def get_collection_name(self, session_id):
        """Get ChromaDB collection name for a session"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT collection_name FROM chroma_collections WHERE session_id = ?",
                (session_id,)
            )
            result = cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            print(f"Error getting collection name: {e}")
            return None
        finally:
            conn.close()

    def save_uploaded_image(self, user_id, session_id, original_filename, file_path, file_type):
        """Save uploaded image with duplicate check"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            # Check for existing image with same name in session
            cursor.execute(
                """SELECT image_id FROM uploaded_images 
                WHERE session_id = ? AND original_filename = ?""",
                (session_id, original_filename)
            )
            existing = cursor.fetchone()
            
            if existing:
                conn.close()
                return existing[0]
                
            # Insert new image if not exists
            cursor.execute(
                """INSERT INTO uploaded_images 
                (user_id, session_id, original_filename, file_path, file_type)
                VALUES (?, ?, ?, ?, ?)""",
                (user_id, session_id, original_filename, file_path, file_type)
            )
            
            image_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return image_id
            
        except Exception as e:
            print(f"Error saving image: {e}")
            return None

    def get_session_images(self, session_id):
        """Get all images uploaded in a specific chat session, ordered by upload time"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            cursor.execute(
                """SELECT image_id, original_filename, file_path, file_type, upload_timestamp 
                FROM uploaded_images 
                WHERE session_id = ? 
                ORDER BY upload_timestamp DESC""",
                (session_id,)
            )
            images = cursor.fetchall()
            conn.close()
            return images
        except Exception as e:
            print(f"Error getting session images: {e}")
            return []

    def remove_image(self, image_id):
        """Remove an image from the database"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            # Get file path before deletion
            cursor.execute(
                "SELECT file_path FROM uploaded_images WHERE image_id = ?", 
                (image_id,)
            )
            result = cursor.fetchone()
            
            if result:
                cursor.execute(
                    "DELETE FROM uploaded_images WHERE image_id = ?", 
                    (image_id,)
                )
                conn.commit()
                return result[0]  # Return file path for physical deletion
            return None
        except Exception as e:
            print(f"Error removing image: {e}")
            return None
        finally:
            conn.close()

    def save_chunk_info(self, chunk_id, chunk_name, file_name, session_id, user_id):
        """Save document chunk information"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO document_chunks 
                (chunk_id, chunk_name, file_name, session_id, user_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (chunk_id, chunk_name, file_name, session_id, user_id))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving chunk info: {e}")
            return False
        finally:
            conn.close()

    def get_chunks_by_session(self, session_id):
        """Get all chunks for a specific session"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT chunk_id, chunk_name, file_name, created_at
                FROM document_chunks
                WHERE session_id = ?
                ORDER BY created_at DESC
            ''', (session_id,))
            return cursor.fetchall()
        finally:
            conn.close()

    def get_file_chunks(self, file_id):
        """Get all chunks associated with a specific file"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            # First get the file info
            cursor.execute(
                """SELECT session_id, original_filename 
                FROM uploaded_files 
                WHERE file_id = ?""", 
                (file_id,)
            )
            file_info = cursor.fetchone()
            
            if not file_info:
                return None, None
                
            session_id, filename = file_info
            
            # Then get all chunks associated with this file
            cursor.execute(
                """SELECT chunk_id 
                FROM document_chunks 
                WHERE file_name = ? AND session_id = ?""",
                (filename, session_id)
            )
            
            chunk_ids = [row[0] for row in cursor.fetchall()]
            return chunk_ids, filename
            
        except Exception as e:
            print(f"Error getting file chunks: {e}")
            return None, None
        finally:
            conn.close()

    def remove_file_chunks(self, filename, session_id):
        """Remove all chunks associated with a file"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            cursor.execute(
                """DELETE FROM document_chunks 
                WHERE file_name = ? AND session_id = ?""",
                (filename, session_id)
            )
            conn.commit()
            return True
        except Exception as e:
            print(f"Error removing file chunks: {e}")
            return False
        finally:
            conn.close()

    def remove_file(self, file_id):
        """Remove a file and its associated data"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            # Get file path before deletion
            cursor.execute(
                """SELECT file_path, original_filename, session_id 
                FROM uploaded_files 
                WHERE file_id = ?""",
                (file_id,)
            )
            result = cursor.fetchone()
            
            if result:
                file_path, filename, session_id = result
                
                # Delete from uploaded_files
                cursor.execute(
                    "DELETE FROM uploaded_files WHERE file_id = ?", 
                    (file_id,)
                )
                
                # Delete from document_chunks
                cursor.execute(
                    """DELETE FROM document_chunks 
                    WHERE file_name = ? AND session_id = ?""",
                    (filename, session_id)
                )
                
                conn.commit()
                return file_path, filename, session_id
            return None, None, None
            
        except Exception as e:
            print(f"Error removing file: {e}")
            return None, None, None
        finally:
            conn.close()

    def save_conversation_with_sources(self, session_id, user_id, api_source, user_message, assistant_response, sources):
        """Save conversation and its document sources with content"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            # Save the conversation
            cursor.execute(
                """INSERT INTO conversations 
                (session_id, user_id, api_source, user_message, assistant_response)
                VALUES (?, ?, ?, ?, ?)""",
                (session_id, user_id, api_source, user_message, assistant_response)
            )
            conversation_id = cursor.lastrowid
            
            # Save document sources with their content
            if sources:
                for source in sources:
                    chunk_content = source.get('chunk_content')  # 只從 source 對象獲取內容
                    
                    cursor.execute(
                        """INSERT INTO conversation_sources
                        (conversation_id, source_file, chunk_number, chunk_content, distance)
                        VALUES (?, ?, ?, ?, ?)""",
                        (conversation_id, 
                         source['source'],
                         source['chunk'],
                         chunk_content,
                         source['distance'])
                    )
            
            conn.commit()
            
            # Return updated conversations with sources
            cursor.execute(
                """SELECT 
                    c.conversation_id,
                    c.session_id,
                    c.user_message,
                    c.assistant_response,
                    c.timestamp,
                    GROUP_CONCAT(
                        json_object(
                            'source', cs.source_file,
                            'chunk', cs.chunk_number,
                            'content', cs.chunk_content,
                            'distance', cs.distance
                        )
                    ) as sources
                FROM conversations c
                LEFT JOIN conversation_sources cs ON c.conversation_id = cs.conversation_id
                WHERE c.session_id = ? AND c.user_id = ?
                GROUP BY c.conversation_id
                ORDER BY c.timestamp ASC""",
                (session_id, user_id)
            )
            
            return cursor.fetchall()
            
        except Exception as e:
            print(f"Error saving conversation with sources: {e}")
            return None
        finally:
            conn.close()

    def get_user_conversations_with_sources(self, user_id):
        """Get all conversations with their sources for a user grouped by session"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            cursor.execute(
                """SELECT 
                    s.session_id,
                    c.api_source,
                    c.user_message,
                    c.assistant_response,
                    c.timestamp,
                    s.created_at,
                    GROUP_CONCAT(
                        json_object(
                            'source', cs.source_file,
                            'chunk', cs.chunk_number,
                            'content', cs.chunk_content,
                            'distance', cs.distance
                        )
                    ) as sources
                FROM chat_sessions s
                JOIN conversations c ON s.session_id = c.session_id
                LEFT JOIN conversation_sources cs ON c.conversation_id = cs.conversation_id
                WHERE s.user_id = ? AND c.hidden = FALSE
                GROUP BY c.conversation_id
                ORDER BY s.created_at DESC, c.timestamp ASC""",
                (user_id,)
            )
            
            return cursor.fetchall()
            
        except Exception as e:
            print(f"Error getting conversations with sources: {e}")
            return []
        finally:
            conn.close()
    def get_chunk_content(self, session_id, source_file, chunk_number):
        """Get content for a specific document chunk"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            # First try to get from conversation_sources
            cursor.execute("""
                SELECT cs.chunk_content
                FROM conversation_sources cs
                JOIN conversations c ON cs.conversation_id = c.conversation_id
                WHERE c.session_id = ? 
                AND cs.source_file = ? 
                AND cs.chunk_number = ?
                ORDER BY c.timestamp DESC
                LIMIT 1
            """, (session_id, source_file, chunk_number))
            
            result = cursor.fetchone()
            
            if result and result[0]:
                return result[0]
            
            # If not found, try to get from document_chunks
            cursor.execute("""
                SELECT dc.chunk_content
                FROM document_chunks dc
                WHERE dc.session_id = ?
                AND dc.file_name = ?
                AND dc.chunk_name = ?
                LIMIT 1
            """, (session_id, source_file, f"chunk_{chunk_number}"))
            
            result = cursor.fetchone()
            return result[0] if result else None
            
        except Exception as e:
            print(f"Error getting chunk content: {e}")
            return None
        finally:
            conn.close()

    # Add a test function to verify email configuration
    def test_email_configuration():
        """Test email configuration and SMTP connection"""
        email_verification = EmailVerification()
        
        # Print current configuration
        print("\nEmail Configuration Test")
        print("-" * 50)
        print(f"SMTP Server: {email_verification.smtp_server}")
        print(f"SMTP Port: {email_verification.smtp_port}")
        print(f"Sender Email: {email_verification.sender_email}")
        print(f"Password Set: {'Yes' if email_verification.sender_password else 'No'}")
        
        # Test SMTP connection
        try:
            with smtplib.SMTP(email_verification.smtp_server, int(email_verification.smtp_port)) as server:
                server.starttls()
                server.login(email_verification.sender_email, email_verification.sender_password)
                print("\n✅ SMTP connection successful!")
        except Exception as e:
            print(f"\n❌ SMTP connection failed: {str(e)}")
        
        return email_verification
    
    def verify_email(self, token: str) -> Tuple[bool, str]:
        """
        Verify user's email using verification token
        
        Args:
            token: The verification token from the email link
            
        Returns:
            tuple[bool, str]: (success, message)
        """
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            # Get token information
            cursor.execute(
                """
                SELECT vt.user_id, vt.expires_at, vt.is_used, u.username, u.is_verified
                FROM verification_tokens vt
                JOIN users u ON vt.user_id = u.user_id
                WHERE vt.token = ?
                """,
                (token,)
            )
            result = cursor.fetchone()
            
            if not result:
                logger.error(f"Invalid verification token: {token}")
                return False, "Invalid verification token"
                
            user_id, expires_at_str, is_used, username, is_verified = result
            
            # Check if user is already verified
            if is_verified:
                logger.info(f"User {username} is already verified")
                return True, "Your email is already verified. You can log in."
            
            # Parse the datetime string, handling different possible formats
            try:
                # First try parsing with microseconds
                expires_at = datetime.strptime(expires_at_str.split('.')[0], '%Y-%m-%d %H:%M:%S')
            except ValueError:
                try:
                    # Try without microseconds
                    expires_at = datetime.strptime(expires_at_str, '%Y-%m-%d %H:%M:%S')
                except ValueError as e:
                    logger.error(f"Error parsing datetime: {e}")
                    return False, "Invalid verification token"
            
            # Check if token is expired
            if datetime.utcnow() > expires_at:
                logger.warning(f"Expired verification token for user {username}")
                return False, "Verification link has expired. Please request a new one."
            
            # Check if token is already used
            if is_used:
                logger.warning(f"Used verification token attempt for user {username}")
                return False, "This verification link has already been used."
            
            # Mark user as verified
            cursor.execute(
                "UPDATE users SET is_verified = 1 WHERE user_id = ?",
                (user_id,)
            )
            
            # Mark token as used
            cursor.execute(
                "UPDATE verification_tokens SET is_used = 1 WHERE token = ?",
                (token,)
            )
            
            conn.commit()
            logger.info(f"Successfully verified email for user {username}")
            return True, "Email verification successful! You can now log in."
            
        except Exception as e:
            logger.error(f"Error in verify_email: {str(e)}")
            return False, f"Verification failed: {str(e)}"
        finally:
            conn.close()


    def resend_verification_email(self, username: str) -> bool:
        """Resend verification email with fixed datetime handling"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            # Get user information
            cursor.execute(
                """SELECT user_id, email, is_verified 
                FROM users 
                WHERE username = ?""",
                (username,)
            )
            result = cursor.fetchone()
            
            if not result:
                logger.error(f"User not found: {username}")
                return False
                
            user_id, email, is_verified = result
            
            if is_verified:
                logger.info(f"User {username} is already verified")
                return False
            
            # Generate new verification token
            verification_token = str(uuid.uuid4())
            expires_at = datetime.utcnow() + timedelta(hours=24)

            expires_at_str = expires_at.strftime('%Y-%m-%d %H:%M:%S')
            
            # Save new token
            cursor.execute(
                """INSERT INTO verification_tokens (user_id, token, expires_at)
                VALUES (?, ?, ?)""",
                (user_id, verification_token, expires_at_str)
            )
            
            conn.commit()
            
            # Send verification email
            return self.email_verification.send_verification_email(email, verification_token)
            
        except Exception as e:
            logger.error(f"Error resending verification email: {str(e)}")
            return False
        finally:
            conn.close()

    # Add utility method to check verification status
    def check_verification_status(self, username: str) -> tuple[bool, str]:
        """
        Check if a user's email is verified
        
        Args:
            username: Username to check
            
        Returns:
            tuple[bool, str]: (is_verified, message)
        """
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT is_verified, email FROM users WHERE username = ?",
                (username,)
            )
            result = cursor.fetchone()
            
            if not result:
                return False, "User not found"
                
            is_verified, email = result
            
            if is_verified:
                return True, "Email is verified"
            else:
                return False, f"Email {email} is not verified"
                
        except Exception as e:
            logger.error(f"Error checking verification status: {str(e)}")
            return False, f"Error checking verification status: {str(e)}"
        finally:
            conn.close()

    # Add method to manually verify user (for testing/admin purposes)
    def set_verification_status(self, username: str, verified: bool = True) -> bool:
        """
        Manually set user verification status
        
        Args:
            username: Username to update
            verified: Verification status to set
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            cursor.execute(
                "UPDATE users SET is_verified = ? WHERE username = ?",
                (verified, username)
            )
            
            if cursor.rowcount == 0:
                logger.error(f"User not found: {username}")
                return False
                
            conn.commit()
            logger.info(f"Updated verification status for {username} to {verified}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting verification status: {str(e)}")
            return False
        finally:
            conn.close()

    def change_password(self, user_id: int, old_password: str, new_password: str) -> Tuple[bool, str]:
        """
        Change user's password with validation
        
        Args:
            user_id: User's ID
            old_password: Current password
            new_password: New password to set
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            # Verify old password
            old_password_hash = self.hash_password(old_password)
            cursor.execute(
                """SELECT username FROM users 
                WHERE user_id = ? AND password_hash = ?""",
                (user_id, old_password_hash)
            )
            result = cursor.fetchone()
            
            if not result:
                return False, "Current password is incorrect"
                
            # Update to new password
            new_password_hash = self.hash_password(new_password)
            cursor.execute(
                """UPDATE users 
                SET password_hash = ? 
                WHERE user_id = ?""",
                (new_password_hash, user_id)
            )
            
            conn.commit()
            return True, "Password changed successfully"
            
        except Exception as e:
            logger.error(f"Error in change_password: {str(e)}")
            return False, f"Failed to change password: {str(e)}"
        finally:
            conn.close()

    def update_profile(self, user_id: int, old_password: str, new_password: str = None, new_email: str = None) -> Tuple[bool, str]:
        """
        Update user's password and/or email with validation
        
        Args:
            user_id: User's ID
            old_password: Current password for verification
            new_password: Optional new password
            new_email: Optional new email
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            # Verify old password and get current email
            old_password_hash = self.hash_password(old_password)
            cursor.execute(
                """SELECT username, email FROM users 
                WHERE user_id = ? AND password_hash = ?""",
                (user_id, old_password_hash)
            )
            result = cursor.fetchone()
            
            if not result:
                return False, "Current password is incorrect"
                
            username, current_email = result
            
            # Check if new email already exists
            if new_email and new_email != current_email:
                cursor.execute("SELECT user_id FROM users WHERE email = ?", (new_email,))
                if cursor.fetchone():
                    return False, "Email address already in use"
            
            # Prepare update query
            update_fields = []
            params = []
            
            if new_password:
                update_fields.append("password_hash = ?")
                params.append(self.hash_password(new_password))
                
            if new_email and new_email != current_email:
                update_fields.append("email = ?")
                update_fields.append("is_verified = 0")  # Reset verification status
                params.append(new_email)
            
            if not update_fields:
                return False, "No changes requested"
                
            # Update user record
            query = f"""UPDATE users 
                    SET {', '.join(update_fields)}
                    WHERE user_id = ?"""
            params.append(user_id)
            
            cursor.execute(query, params)
            conn.commit()
            
            # If email changed, send verification
            if new_email and new_email != current_email:
                # Generate verification token
                verification_token = str(uuid.uuid4())
                expires_at = datetime.utcnow() + timedelta(hours=24)

                expires_at_str = expires_at.strftime('%Y-%m-%d %H:%M:%S')
                
                # Save token
                cursor.execute(
                    """INSERT INTO verification_tokens (user_id, token, expires_at)
                    VALUES (?, ?, ?)""",
                    (user_id, verification_token, expires_at_str)
                )
                conn.commit()
                
                # Send verification email
                if not self.email_verification.send_verification_email(new_email, verification_token):
                    logger.warning(f"Failed to send verification email to {new_email}")
                    return True, "Profile updated but verification email failed to send. Please contact support."
                
                return True, "Profile updated successfully. Please check your email to verify your new address."
                
            return True, "Profile updated successfully"
            
        except Exception as e:
            logger.error(f"Error in update_profile: {str(e)}")
            return False, f"Failed to update profile: {str(e)}"
        finally:
            conn.close()