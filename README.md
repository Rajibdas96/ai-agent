# Study AI

A lightweight, privacy-friendly desktop study assistant that lets you search your PDF study materials, generate grounded answers, and use voice input/output.

Study AI is designed to work without downloading or running a large AI model.

## Features

- 📚 Search your PDF study materials
- 🔎 TF-IDF based document retrieval
- 📝 Automatic answer generation from PDF content
- 🎯 Different answer types
- 📖 Definition answers
- ✍️ Short answers
- 🧠 Exam-style answers
- ⚖️ Advantages and disadvantages
- 🔄 Difference / comparison answers
- 📋 Step-by-step answers
- 🎤 Voice questions
- 🔊 Text-to-speech answers
- 🖥️ Modern Windows desktop interface
- 🚫 No LLM/model download required
- 📦 Windows standalone executable

## How it works

Study AI uses traditional information-retrieval techniques instead of a generative AI model.

```text
PDF
 │
 ▼
PDF Text Extraction
 │
 ▼
Text Cleaning
 │
 ▼
Text Chunking
 │
 ▼
TF-IDF Search
 │
 ▼
Relevant Sentences
 │
 ▼
Answer Formatting
 │
 ▼
Study Answer
```

This approach keeps answers grounded in the user's PDF library and avoids requiring a large language model.

## Installation

### Option 1 — Windows executable

Download the latest Windows release from:

**GitHub Releases**

Download the `.exe` file and launch it.

No Python installation is required when using the packaged Windows application.

### Option 2 — Run from source

Clone the repository:

```bash
git clone https://github.com/Rajibdas96/ai-agent.git
cd ai-agent
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the application:

```powershell
python main.py
```

## Adding study materials

Place your PDF files inside:

```text
pdfs/
```

Example:

```text
pdfs/
├── physics.pdf
├── mathematics.pdf
└── indian_knowledge_system.pdf
```

Then open Study AI and reload the library.

## Voice

Study AI supports:

- Speech-to-text for questions
- Text-to-speech for answers

Internet access may be required for the speech services used by the application.

## Answer Types

Study AI supports several answer styles:

| Type | Purpose |
|---|---|
| Auto | Automatically select an appropriate answer format |
| Short | Quick answer |
| Definition | Definition-style answer |
| Brief | Concise explanation |
| Exam | Exam-oriented answer |
| Explain | Explanation |
| Advantages | Advantages |
| Disadvantages | Disadvantages |
| Difference | Compare two concepts |
| Compare | Comparison |
| Steps | Step-by-step explanation |

## Privacy

Study AI does not require an external LLM or API key to search your local PDF library.

Your PDF files are stored locally.

User PDF files are intentionally excluded from this Git repository.

## Limitations

Study AI is not a general-purpose generative AI assistant.

The answer engine primarily retrieves, ranks, selects, and formats information from the user's PDF documents.

Because it does not use a generative language model, it does not freely rewrite arbitrary textbook content like ChatGPT.

This is intentional: the goal is to keep study answers grounded in the user's source material.

## Project Structure

```text
ai-agent/
│
├── .github/
│   └── workflows/
│       └── build-windows.yml
│
├── pdfs/
│
├── tests/
│
├── answer_engine.py
├── main.py
├── study_engine.py
├── voice_engine.py
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── build_windows.bat
```

## Building for Windows

Install PyInstaller:

```powershell
python -m pip install pyinstaller
```

Then run:

```powershell
build_windows.bat
```

The Windows executable will be generated in:

```text
dist/
```

## License

MIT License

See `LICENSE` for details.

## Author

Rajib Das

---

⭐ If you find Study AI useful, consider starring the repository.
