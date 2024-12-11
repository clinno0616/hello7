# Multi-Modal RAG Chat Application

An advanced chat application that integrates multiple AI models, document processing, image and video analysis capabilities.

## Features

### AI Integration
- Support for multiple AI providers:
  - OpenAI (GPT-4), Need API Key.
  - Anthropic (Claude) Need API Key.
  - Groq (llama-3.3-70b-versatile) Need API Key.
  - OLLAMA (Local models)
- Vision processing capabilities using LLaMA Vision (llama-3.2-90b-vision-preview models)

### Document Processing
- Support for multiple file formats:
  - PDF
  - DOCX
  - CSV
  - TXT
- ChromaDB integration for semantic search and document retrieval
- Document chunking and source referencing

### Image Processing
- Support for image/video analysis and processing
- Integration with llama-3.2-90b-vision-preview for image/video understanding
- Support for JPG, JPEG, PNG, mp4 formats

### User Management
- User authentication with local and AD (Active Directory) support
- Email verification system
- Profile management
- Password change functionality

### Chat Features
- Real-time chat interface
- Conversation history tracking
- File upload and management
- Source document reference tracking
- Multiple chat sessions support

## Installation

1. Clone the repository:
```bash
git clone https://github.com/clinno0616/hello7.git)
cd hello7
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables in `.env` file:
```plaintext
# API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GROQ_API_KEY=your_groq_key

# Email Configuration
SMTP_SERVER = 
SMTP_PORT = 
GMAIL_USER= 
GMAIL_APP_PASSWORD=

# Active Directory Configuration (Optional)
LDAP_SERVER=ldap://x.x.x.x:389
ROOT_DN='ou=User,ou=companyname,dc=domainname,dc=domainname,dc=domainname'
LDAP_PORT=389
BIND_DN=
BIND_PASSWORD=
```

5. Initialize the database:
```python
python
>>> from database import Database
>>> db = Database()
>>> db.init_database()
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Access the application through your web browser at `http://localhost:8501`

3. Register a new account or login with existing credentials

### Basic Usage Flow:
1. Login to the application
2. Select your preferred AI model
3. Upload documents or images if needed
4. Start chatting with the AI
5. View conversation history and referenced sources

### File Upload Guidelines:
- Maximum 5 files per upload
- Supported formats: PDF, DOCX, CSV, TXT
- For images: JPG, JPEG, PNG, MP4
- Files are automatically processed and indexed

## Configuration

### ChromaDB Configuration 
Need to create Chromdb and Embedding model in Docker. 
- Default host: localhost
- Chromdb Default port: 8000
- Embedding model: BAAI/bge-m3 ,  Default port: 8021
- Reranker model: BAAI/bge-reranker-v2-m3

### Vision Model Configuration
- Default model: Groq llama-3.2-90b-vision-preview (need ANTHROPIC_API_KEY in .env)
- Support for both image and video processing

## Security

- Passwords are hashed using SHA-256
- Email verification required for new accounts
- Support for Active Directory integration
- Secure file handling and storage

## Development

### Project Structure:
```plaintext
ai-chat-application/
├── app.py                 # Main Streamlit application
├── database.py           # Database operations
├── chromadb_operations.py # ChromaDB integration
├── image_operations.py   # Image processing
├── llama32_vision.py     # Vision operations
├── requirements.txt      # Dependencies
├── .env                 # Environment variables
└── uploads/             # Uploaded files directory
    ├── images/          # Image storage
    └── models/          # Model storage
```
## License

This project is licensed under the Apache License 2.0.
```
Copyright 2024 [clinno0616]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
For detailed license terms, please refer to the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) official website.

## User Interface
Default admin is 'admin' , password is 'admin123'
![image](https://github.com/user-attachments/assets/5a7251ac-9128-4ffc-a7c6-dc687a7760ea)
![image](https://github.com/user-attachments/assets/dfdd84a9-6859-41fa-ba6b-b029f9d26594)


