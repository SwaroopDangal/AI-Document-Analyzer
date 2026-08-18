# 📄 AI Document Analyzer

An AI-powered document analysis application built with **Python and Streamlit**. The application allows users to upload PDF documents, generate concise English summaries, listen to summaries in multiple languages, and generate probable exam questions from the uploaded document.

## 🚀 Features

### 📤 PDF Upload & Text Extraction
- Upload PDF documents through the Streamlit interface.
- Extract text using **PyMuPDF**.
- Display the number of pages and extracted words.
- Detect PDFs where no readable text can be extracted.

### 🤖 AI Text Summarization
- Uses the pretrained **DistilBART** model from Hugging Face.
- Generates an English summary from the uploaded document.
- Uses beam search and repetition control.
- Loads the model locally after downloading it once.

### 📝 English Written Summary
- Displays the generated English summary.
- Helps users quickly understand lengthy documents.

### 🔊 Multilingual Audio Summary
- Converts the generated summary into speech.
- English is selected by default.
- Provides a language dropdown.
- Uses **Google Translator** for translation.
- Uses **gTTS (Google Text-to-Speech)** for audio generation.

Supported languages include:

- 🇬🇧 English
- 🇳🇵 Nepali
- 🇮🇳 Hindi
- 🇪🇸 Spanish
- 🇫🇷 French
- 🇩🇪 German
- 🇮🇹 Italian
- 🇵🇹 Portuguese
- 🇷🇺 Russian
- 🇨🇳 Chinese
- 🇯🇵 Japanese
- 🇰🇷 Korean
- 🇸🇦 Arabic
- 🇧🇩 Bengali
- 🇮🇩 Indonesian
- 🇹🇷 Turkish
- 🇳🇱 Dutch
- 🇻🇳 Vietnamese
- 🇹🇭 Thai
- 🇺🇦 Ukrainian

### 🎓 Probable Exam Question Generation
- Generates questions based on the uploaded PDF.
- Uses the pretrained **T5-small Question Generation model** from Hugging Face.
- Allows users to select the number of questions.
- Filters duplicate and poorly formed questions.

---

## 🧠 AI Models and Technologies

| Function | Technology |
|---|---|
| PDF Text Extraction | PyMuPDF |
| Text Summarization | DistilBART |
| Question Generation | T5-small |
| Translation | GoogleTranslator |
| Text-to-Speech | gTTS |
| User Interface | Streamlit |
| Programming Language | Python |

---

## 🏗️ Project Architecture

```text
                         ┌─────────────────┐
                         │    PDF Upload   │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │    PyMuPDF      │
                         │ Text Extraction │
                         └────────┬────────┘
                                  │
                 ┌────────────────┴────────────────┐
                 │                                 │
                 ▼                                 ▼
        ┌─────────────────┐              ┌──────────────────┐
        │   DistilBART    │              │   T5-small QG    │
        │   Summarizer    │              │ Question Generator│
        └────────┬────────┘              └─────────┬────────┘
                 │                                 │
                 ▼                                 ▼
        ┌─────────────────┐              ┌──────────────────┐
        │ English Written │              │ Probable Exam    │
        │     Summary     │              │    Questions     │
        └────────┬────────┘              └──────────────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Google Translator│
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │      gTTS       │
        │  Text-to-Speech │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │  Audio Summary  │
        └─────────────────┘
```

---

## 📁 Project Structure

```text
AI-Document-Analyzer/
│
├── app.py
│
├── models/
│   ├── distilbart-cnn-6-6/
│   │   └── ...
│   │
│   └── t5-small-qg-hl/
│       └── ...
│
├── download_summary_model.py
├── download_question_model.py
├── requirements.txt
└── README.md
```

> **Note:** Model files can be large. For GitHub, it is recommended to add the `models/` directory to `.gitignore` and provide download scripts instead of committing model weights directly.

---

## ⚙️ Requirements

Recommended:

```text
Python 3.10+
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Or:

```bash
pip install streamlit pymupdf gTTS deep-translator transformers torch huggingface-hub sentencepiece
```

---

## 📦 requirements.txt

```text
streamlit
pymupdf
gTTS
deep-translator
transformers
torch
huggingface-hub
sentencepiece
```

---

## 🤗 Download the AI Models

### 1. Download DistilBART

Create `download_summary_model.py`:

```python
from huggingface_hub import snapshot_download

print("Starting model download...")
print("Model: sshleifer/distilbart-cnn-6-6")
print()

path = snapshot_download(
    repo_id="sshleifer/distilbart-cnn-6-6",
    local_dir="./models/distilbart-cnn-6-6"
)

print()
print("===================================")
print("Download completed!")
print("Model location:")
print(path)
print("===================================")
```

Run:

```bash
python download_summary_model.py
```

### 2. Download T5 Question Generation Model

Create `download_question_model.py`:

```python
from huggingface_hub import snapshot_download

print("Starting model download...")
print("Model: valhalla/t5-small-qg-hl")
print()

path = snapshot_download(
    repo_id="valhalla/t5-small-qg-hl",
    local_dir="./models/t5-small-qg-hl"
)

print()
print("===================================")
print("Download completed!")
print("Model location:")
print(path)
print("===================================")
```

Run:

```bash
python download_question_model.py
```

---

## ▶️ Run the Application

After installing dependencies and downloading both models:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

## 🔄 Application Workflow

```text
1. Upload PDF
       ↓
2. Extract PDF text
       ↓
3. Generate English summary
       ↓
4. Display written summary
       ↓
5. Select audio language
       ↓
6. Translate summary
       ↓
7. Generate audio
       ↓
8. Select number of questions
       ↓
9. Generate probable exam questions
```

---

## 🎯 Use Cases

This application can be useful for:

- 📚 Students preparing for examinations
- 📄 Summarizing lecture notes
- 📖 Understanding lengthy study materials
- 🎧 Audio-based learning
- 🌐 Multilingual learning
- 🎓 Generating revision questions
- 📝 Quick document revision

---

## ⚠️ Limitations

- Summary quality depends on the quality and structure of the PDF text.
- Image-only or scanned PDFs may require OCR.
- Question generation may sometimes produce imperfect questions.
- Translation and gTTS require an internet connection.
- Large PDFs may take longer to process, especially on CPU.
- Generated questions are **probable questions based on document content**, not guaranteed predictions of actual examination questions.

---

## 🔮 Future Improvements

- 🔍 OCR support for scanned PDFs
- 🎓 Question difficulty classification
- ⭐ Probability ranking for exam questions
- 📝 MCQ generation
- ✅ Automatic answer generation
- 📊 Quiz mode
- 💾 Download summaries as PDF/DOCX
- 📑 Chapter-wise summarization
- 🧠 Fine-tuning using past exam papers
- 🔎 Important-topic detection
- 📈 Topic-wise analysis
- 🎤 Voice-based interaction

---

## 👨‍💻 Project Overview

**AI Document Analyzer** transforms a normal PDF into an interactive learning resource:

```text
PDF
 ↓
AI Summary
 ↓
Multilingual Audio
 ↓
Probable Exam Questions
```

Built with **Python, Streamlit, Hugging Face Transformers, PyMuPDF, Google Translator, and gTTS**.
