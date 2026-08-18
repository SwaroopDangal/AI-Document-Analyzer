import streamlit as st
import pymupdf
from gtts import gTTS
from deep_translator import GoogleTranslator
from io import BytesIO
import re
import os

from huggingface_hub import snapshot_download

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


# =========================================================
# Download Models for Deployment
# =========================================================

@st.cache_resource
def download_models():

    summary_path = "./models/distilbart-cnn-6-6"
    question_path = "./models/t5-small-qg-hl"


    # -----------------------------------------------------
    # Download Summary Model
    # -----------------------------------------------------

    if not os.path.exists(summary_path):

        snapshot_download(
            repo_id="sshleifer/distilbart-cnn-6-6",
            local_dir=summary_path
        )


    # -----------------------------------------------------
    # Download Question Generation Model
    # -----------------------------------------------------

    if not os.path.exists(question_path):

        snapshot_download(
            repo_id="valhalla/t5-small-qg-hl",
            local_dir=question_path
        )


    return summary_path, question_path


# Download models automatically on Streamlit deployment

summary_model_path, question_model_path = download_models()


# =========================================================
# Load Summary Model
# =========================================================

@st.cache_resource
def load_summary_model():

    model_path = summary_model_path

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        local_files_only=True
    )

    return tokenizer, model


# =========================================================
# Load Question Generation Model
# =========================================================

@st.cache_resource
def load_question_model():

    model_path = question_model_path

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        local_files_only=True
    )

    return tokenizer, model


# =========================================================
# Load Models
# =========================================================

summary_tokenizer, summary_model = load_summary_model()

question_tokenizer, question_model = load_question_model()


# =========================================================
# Generate Summary
# =========================================================

def generate_summary(text):

    inputs = summary_tokenizer(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )

    original_tokens = inputs["input_ids"].shape[1]

    target_summary_tokens = max(
        50,
        int(original_tokens * 0.30)
    )

    max_summary_tokens = max(
        80,
        int(original_tokens * 0.40)
    )

    if max_summary_tokens <= target_summary_tokens:

        max_summary_tokens = (
            target_summary_tokens + 20
        )

    outputs = summary_model.generate(
        **inputs,
        max_length=max_summary_tokens,
        min_length=target_summary_tokens,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    summary = summary_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return summary


# =========================================================
# Generate One Question
# =========================================================

def generate_question(context, answer):

    input_text = (
        "generate question: "
        + answer
        + " context: "
        + context
    )

    inputs = question_tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    outputs = question_model.generate(
        **inputs,
        max_length=64,
        min_length=8,
        num_beams=5,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    question = question_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()

    return question


# =========================================================
# Clean Question
# =========================================================

def clean_question(question):

    question = question.strip()

    # Remove common unwanted prefixes

    prefixes = [
        "question:",
        "question",
        "q:"
    ]

    for prefix in prefixes:

        if question.lower().startswith(prefix):

            question = question[
                len(prefix):
            ].strip()


    # Remove repeated consecutive words

    words = question.split()

    cleaned_words = []

    for word in words:

        if not cleaned_words:

            cleaned_words.append(word)

        elif word.lower() != cleaned_words[-1].lower():

            cleaned_words.append(word)


    question = " ".join(
        cleaned_words
    )


    # Remove multiple spaces

    question = re.sub(
        r"\s+",
        " ",
        question
    ).strip()


    # Add question mark

    if not question.endswith("?"):

        question += "?"


    return question


# =========================================================
# Check Question Quality
# =========================================================

def is_good_question(question, answer):

    question_lower = question.lower()

    answer_lower = answer.lower()


    # ---------------------------------------------
    # Must contain question mark
    # ---------------------------------------------

    if "?" not in question:

        return False


    # ---------------------------------------------
    # Minimum length
    # ---------------------------------------------

    if len(question.split()) < 4:

        return False


    # ---------------------------------------------
    # Maximum length
    # ---------------------------------------------

    if len(question.split()) > 30:

        return False


    # ---------------------------------------------
    # Question should not copy entire answer
    # ---------------------------------------------

    if answer_lower in question_lower:

        return False


    # ---------------------------------------------
    # Detect excessive repeated words
    # ---------------------------------------------

    words = question_lower.split()

    if len(words) >= 6:

        unique_words = set(words)

        ratio = len(unique_words) / len(words)

        if ratio < 0.50:

            return False


    # ---------------------------------------------
    # Reject if question starts with answer-like text
    # ---------------------------------------------

    bad_starts = [
        "deep learning deep learning",
        "machine learning machine learning",
        "gradient descent gradient descent"
    ]

    for bad_start in bad_starts:

        if question_lower.startswith(bad_start):

            return False


    return True


# =========================================================
# Generate Probable Questions
# =========================================================

def generate_questions(
    text,
    number_of_questions
):

    # ---------------------------------------------
    # Clean PDF text
    # ---------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()


    # ---------------------------------------------
    # Split into sentences
    # ---------------------------------------------

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )


    # ---------------------------------------------
    # Keep useful sentences
    # ---------------------------------------------

    sentences = [

        sentence.strip()

        for sentence in sentences

        if 10 <= len(sentence.split()) <= 60

    ]


    # ---------------------------------------------
    # Remove duplicate sentences
    # ---------------------------------------------

    unique_sentences = []

    seen = set()

    for sentence in sentences:

        key = sentence.lower()

        if key not in seen:

            seen.add(key)

            unique_sentences.append(sentence)


    questions = []


    # ---------------------------------------------
    # Generate questions
    # ---------------------------------------------

    for sentence in unique_sentences:

        if len(questions) >= number_of_questions:

            break


        context = sentence

        answer = sentence


        # -----------------------------------------
        # Generate question
        # -----------------------------------------

        question = generate_question(
            context,
            answer
        )


        # -----------------------------------------
        # Clean question
        # -----------------------------------------

        question = clean_question(
            question
        )


        # -----------------------------------------
        # Quality check
        # -----------------------------------------

        if not is_good_question(
            question,
            answer
        ):

            continue


        # -----------------------------------------
        # Duplicate check
        # -----------------------------------------

        duplicate = False

        for old_question in questions:

            if (
                question.lower()
                == old_question.lower()
            ):

                duplicate = True

                break


        if duplicate:

            continue


        questions.append(
            question
        )


    return questions


# =========================================================
# Translate Text
# =========================================================

def translate_text(
    text,
    target_language
):

    if target_language == "en":

        return text


    translated_text = GoogleTranslator(
        source="en",
        target=target_language
    ).translate(
        text
    )


    return translated_text


# =========================================================
# Generate Audio
# =========================================================

def generate_audio(
    text,
    language
):

    tts = gTTS(
        text=text,
        lang=language
    )


    audio_bytes = BytesIO()


    tts.write_to_fp(
        audio_bytes
    )


    audio_bytes.seek(0)


    return audio_bytes


# =========================================================
# Session State
# =========================================================

if "summary" not in st.session_state:

    st.session_state.summary = ""


if "questions" not in st.session_state:

    st.session_state.questions = []


# =========================================================
# Streamlit UI
# =========================================================

st.title(
    "📄 AI Document Analyzer"
)


st.write(
    "Upload a PDF and generate an AI-powered English summary."
)


# =========================================================
# Upload PDF
# =========================================================

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)


# =========================================================
# Process PDF
# =========================================================

if uploaded_file is not None:

    pdf_bytes = uploaded_file.read()


    doc = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )


    st.success(
        "PDF uploaded successfully!"
    )


    # =====================================================
    # Extract Text
    # =====================================================

    text = ""


    for page in doc:

        text += page.get_text() + "\n"


    # =====================================================
    # Check Text
    # =====================================================

    if not text.strip():

        st.error(
            "No text could be extracted from this PDF."
        )


    else:


        # =================================================
        # Generate Summary
        # =================================================

        if st.button(
            "🤖 Generate Summary"
        ):

            with st.spinner(
                "Generating summary..."
            ):

                st.session_state.summary = (
                    generate_summary(text)
                )


                # Clear old questions

                st.session_state.questions = []


            st.success(
                "✅ Summary generated successfully!"
            )


# =========================================================
# Display Results
# =========================================================

if st.session_state.summary:


    # =====================================================
    # 1. ENGLISH WRITTEN SUMMARY
    # =====================================================

    st.subheader(
        "📝 English Written Summary"
    )


    st.write(
        st.session_state.summary
    )


    st.divider()


    # =====================================================
    # 2. AUDIO SUMMARY
    # =====================================================

    st.subheader(
        "🔊 Audio Summary"
    )


    # =====================================================
    # Language Options
    # =====================================================

    language_options = {

        "🇬🇧 English": "en",

        "🇳🇵 Nepali": "ne",

        "🇮🇳 Hindi": "hi",

        "🇪🇸 Spanish": "es",

        "🇫🇷 French": "fr",

        "🇩🇪 German": "de",

        "🇮🇹 Italian": "it",

        "🇵🇹 Portuguese": "pt",

        "🇷🇺 Russian": "ru",

        "🇨🇳 Chinese": "zh-CN",

        "🇯🇵 Japanese": "ja",

        "🇰🇷 Korean": "ko",

        "🇸🇦 Arabic": "ar",

        "🇧🇩 Bengali": "bn",

        "🇮🇩 Indonesian": "id",

        "🇹🇷 Turkish": "tr",

        "🇳🇱 Dutch": "nl",

        "🇻🇳 Vietnamese": "vi",

        "🇹🇭 Thai": "th",

        "🇺🇦 Ukrainian": "uk"
    }


    # =====================================================
    # Audio Language Dropdown
    # =====================================================

    selected_language = st.selectbox(

        "Select Audio Language:",

        options=list(
            language_options.keys()
        ),

        index=0
    )


    language_code = language_options[
        selected_language
    ]


    # =====================================================
    # Generate Audio
    # =====================================================

    if st.button(
        "🔊 Generate Audio",
        key="generate_audio"
    ):

        with st.spinner(
            f"Generating {selected_language} audio..."
        ):


            # ---------------------------------------------
            # Translate Summary
            # ---------------------------------------------

            audio_text = translate_text(

                st.session_state.summary,

                language_code
            )


            # ---------------------------------------------
            # Generate Audio
            # ---------------------------------------------

            audio = generate_audio(

                audio_text,

                language_code
            )


        st.success(
            f"✅ {selected_language} audio generated!"
        )


        st.audio(
            audio.read(),
            format="audio/mp3"
        )


    st.divider()


    # =====================================================
    # 3. PROBABLE EXAM QUESTIONS
    # =====================================================

    st.subheader(
        "🎓 Probable Exam Questions"
    )


    st.write(
        "Generate questions that may be asked from the uploaded PDF."
    )


    # =====================================================
    # Number of Questions
    # =====================================================

    question_count = st.selectbox(

        "Number of Questions:",

        options=[
            5,
            10,
            15,
            20
        ],

        index=1
    )


    # =====================================================
    # Generate Questions
    # =====================================================

    if st.button(
        "🎓 Generate Exam Questions",
        key="generate_questions"
    ):

        with st.spinner(
            "Generating probable exam questions..."
        ):

            st.session_state.questions = (
                generate_questions(

                    text,

                    question_count
                )
            )


        if st.session_state.questions:

            st.success(
                "✅ Exam questions generated successfully!"
            )

        else:

            st.warning(
                "No good questions could be generated. "
                "Try generating again."
            )


    # =====================================================
    # Display Questions
    # =====================================================

    if st.session_state.questions:

        st.subheader(
            "📚 Probable Questions"
        )


        for index, question in enumerate(

            st.session_state.questions,

            start=1

        ):

            st.write(
                f"**{index}. {question}**"
            )