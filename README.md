# Sunjos ✿
### Your Cute Knowledge Q&A Assistant

A minimal, adorable RAG-based Q&A system that lets you chat with your documents! Upload PDFs, Word docs, or text files and ask questions in natural language~

![Sunjos Demo](./frontend/screenshot.png)

## ✨ Features

- **📄 Multi-format Support** - Upload PDF, DOCX, TXT, and Markdown files
- **🔍 Smart Retrieval** - TF-IDF based search (works offline!)
- **💬 Natural Answers** - Powered by Groq's Llama 3.3 (FREE tier)
- **📎 Source Citations** - See exactly where answers come from
- **🌸 Cute UI** - Pastel pink theme with animations
- **🌙 Dark Mode** - Easy on the eyes at night

## 🚀 Quick Start

### 1. Get a FREE Groq API Key
Visit [console.groq.com](https://console.groq.com) and create a free account

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Create .env file with your Groq key
echo "GROQ_API_KEY=your-key-here" > .env

# Start server
python main.py
```

### 3. Frontend
Open `frontend/index.html` in your browser!

## 📁 Project Structure

```
sunjos/
├── backend/
│   ├── main.py           # FastAPI server
│   ├── rag_engine.py     # RAG pipeline (TF-IDF + Groq)
│   ├── requirements.txt  # Python dependencies
│   └── .env.example      # Environment template
├── frontend/
│   ├── index.html        # Cute chat interface
│   ├── styles.css        # Pastel pink theme
│   └── app.js            # Frontend logic
└── README.md
```

## 🎨 Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | FastAPI (Python) |
| **LLM** | Groq (Llama 3.3 70B) - FREE |
| **Search** | TF-IDF (scikit-learn) - Offline |
| **Frontend** | Vanilla HTML/CSS/JS |
| **Font** | Quicksand |

## 💕 Why "Sunjos"?

Just a cute name for a cute assistant~ ✿

## 📝 License

MIT License - Use it however you like!

---

Made with 💕 by Pritam
