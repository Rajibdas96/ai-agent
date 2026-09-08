import os
import re
import asyncio

import pypdf
import edge_tts
import pygame
import speech_recognition as sr

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

PDF_FOLDER = "pdfs"

VOICE = "en-IN-NeerjaNeural"
VOICE_RATE = "-5%"

TOP_RESULTS = 5

DEFAULT_MAX_WORDS = 150
BRIEF_MAX_WORDS = 60
DETAILED_MAX_WORDS = 300

AUDIO_FILE = "ai_answer.mp3"

MIC_TIMEOUT = 5
PHRASE_TIME_LIMIT = 10


# ============================================================
# GLOBAL DATA
# ============================================================

documents = []
vectorizer = None
document_matrix = None

recognizer = sr.Recognizer()


# ============================================================
# PDF LOADING
# ============================================================

def load_pdfs():
    """
    Read all PDF files from the pdfs folder.

    Each page becomes a searchable document.
    """

    global documents

    documents = []

    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)

        print(f"\n📁 Created '{PDF_FOLDER}' folder.")
        print("Put your PDF files inside it and run 'reload'.")

        return

    pdf_files = [
        file
        for file in os.listdir(PDF_FOLDER)
        if file.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("\n⚠️ No PDF files found.")
        print(f"Put your PDFs inside the '{PDF_FOLDER}' folder.")

        return

    print("\n📚 Loading PDFs...")

    for filename in pdf_files:

        path = os.path.join(PDF_FOLDER, filename)

        try:

            reader = pypdf.PdfReader(path)

            print(f"   📖 {filename}")

            for page_number, page in enumerate(reader.pages, start=1):

                try:
                    text = page.extract_text()
                except Exception:
                    text = ""

                if not text:
                    continue

                text = clean_text(text)

                if len(text.strip()) < 20:
                    continue

                # Split page into smaller chunks.
                chunks = split_into_chunks(text)

                for chunk in chunks:

                    documents.append(
                        {
                            "text": chunk,
                            "filename": filename,
                            "page": page_number
                        }
                    )

        except Exception as error:

            print(
                f"   ❌ Could not read {filename}: {error}"
            )

    print(
        f"\n✅ Loaded {len(documents)} searchable sections."
    )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean extracted PDF text.
    """

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# CHUNKING
# ============================================================

def split_into_chunks(text, words_per_chunk=180):
    """
    Split large PDF pages into smaller searchable sections.
    """

    words = text.split()

    chunks = []

    for i in range(0, len(words), words_per_chunk):

        chunk = " ".join(
            words[i:i + words_per_chunk]
        )

        if chunk.strip():
            chunks.append(chunk)

    return chunks


# ============================================================
# BUILD SEARCH INDEX
# ============================================================

def build_index():

    global vectorizer
    global document_matrix

    if not documents:
        vectorizer = None
        document_matrix = None
        return

    print("\n🧠 Building search index...")

    texts = [
        document["text"]
        for document in documents
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    document_matrix = vectorizer.fit_transform(texts)

    print("✅ Search index ready.")


# ============================================================
# RELOAD PDFs
# ============================================================

def reload_pdfs():

    print("\n🔄 Reloading PDFs...")

    load_pdfs()
    build_index()

    print("\n✅ PDFs reloaded.")


# ============================================================
# QUESTION LENGTH
# ============================================================

def get_answer_length(question):

    question_lower = question.lower()

    brief_words = [
        "brief",
        "briefly",
        "short",
        "shortly",
        "in short",
        "one line"
    ]

    detailed_words = [
        "detailed",
        "detail",
        "deeply",
        "explain fully",
        "in detail"
    ]

    exam_words = [
        "exam",
        "marks",
        "5 marks",
        "10 marks",
        "answer"
    ]

    for phrase in brief_words:

        if phrase in question_lower:
            return BRIEF_MAX_WORDS

    for phrase in detailed_words:

        if phrase in question_lower:
            return DETAILED_MAX_WORDS

    for phrase in exam_words:

        if phrase in question_lower:
            return 250

    return DEFAULT_MAX_WORDS


# ============================================================
# WORD LIMIT
# ============================================================

def limit_words(text, max_words):

    words = text.split()

    if len(words) <= max_words:
        return text.strip()

    shortened = " ".join(
        words[:max_words]
    )

    # Try to finish naturally.
    last_period = shortened.rfind(".")
    last_question = shortened.rfind("?")
    last_exclamation = shortened.rfind("!")

    last_sentence = max(
        last_period,
        last_question,
        last_exclamation
    )

    if last_sentence > max_words * 3:
        shortened = shortened[:last_sentence + 1]

    else:
        shortened += "..."

    return shortened.strip()


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(text):

    text = clean_text(text)

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ============================================================
# REMOVE DUPLICATE SENTENCES
# ============================================================

def remove_duplicate_sentences(sentences):

    unique = []

    seen = set()

    for sentence in sentences:

        normalized = re.sub(
            r"\s+",
            " ",
            sentence.lower()
        ).strip()

        if normalized in seen:
            continue

        seen.add(normalized)
        unique.append(sentence)

    return unique


# ============================================================
# SEARCH PDF
# ============================================================

def search_pdfs(question):

    if not documents:
        return []

    if vectorizer is None or document_matrix is None:
        return []

    question_vector = vectorizer.transform(
        [question]
    )

    scores = cosine_similarity(
        question_vector,
        document_matrix
    ).flatten()

    ranked_indexes = scores.argsort()[::-1]

    results = []

    for index in ranked_indexes:

        score = scores[index]

        if score <= 0:
            continue

        document = documents[index]

        results.append(
            {
                "text": document["text"],
                "filename": document["filename"],
                "page": document["page"],
                "score": float(score)
            }
        )

        if len(results) >= TOP_RESULTS:
            break

    return results


# ============================================================
# EXTRACT BEST SENTENCES
# ============================================================

def extract_best_sentences(question, results):

    if not results:
        return []

    all_sentences = []

    for result in results:

        sentences = split_sentences(
            result["text"]
        )

        for sentence in sentences:

            if len(sentence.split()) < 4:
                continue

            all_sentences.append(
                {
                    "text": sentence,
                    "filename": result["filename"],
                    "page": result["page"]
                }
            )

    if not all_sentences:
        return []

    sentence_texts = [
        item["text"]
        for item in all_sentences
    ]

    try:

        sentence_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )

        sentence_matrix = sentence_vectorizer.fit_transform(
            sentence_texts + [question]
        )

        question_vector = sentence_matrix[-1]

        sentence_scores = cosine_similarity(
            question_vector,
            sentence_matrix[:-1]
        ).flatten()

    except Exception:

        sentence_scores = [
            0
            for _ in all_sentences
        ]

    ranked = []

    for i, item in enumerate(all_sentences):

        ranked.append(
            {
                "text": item["text"],
                "filename": item["filename"],
                "page": item["page"],
                "score": float(sentence_scores[i])
            }
        )

    ranked.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return ranked


# ============================================================
# BUILD ANSWER FROM PDF
# ============================================================

def create_answer(question):

    results = search_pdfs(question)

    if not results:
        return None, []

    sentences = extract_best_sentences(
        question,
        results
    )

    if not sentences:
        return None, results

    max_words = get_answer_length(question)

    selected_sentences = []

    current_words = 0

    for sentence in sentences:

        sentence_words = len(
            sentence["text"].split()
        )

        if current_words + sentence_words > max_words:
            continue

        selected_sentences.append(sentence)

        current_words += sentence_words

        if current_words >= max_words:
            break

        # Usually 3-6 sentences is enough.
        if len(selected_sentences) >= 6:
            break

    if not selected_sentences:

        selected_sentences = sentences[:1]

    answer_sentences = [
        item["text"]
        for item in selected_sentences
    ]

    answer_sentences = remove_duplicate_sentences(
        answer_sentences
    )

    answer = " ".join(
        answer_sentences
    )

    answer = limit_words(
        answer,
        max_words
    )

    # Sources used by selected sentences.
    sources = []

    for item in selected_sentences:

        source = (
            item["filename"],
            item["page"]
        )

        if source not in sources:
            sources.append(source)

    return answer, sources


# ============================================================
# SPEECH CLEANING
# ============================================================

def clean_for_speech(text):

    text = text.replace("*", "")
    text = text.replace("_", "")

    text = text.replace("•", "")
    text = text.replace("●", "")
    text = text.replace("▪", "")

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

    text = text.replace(";", ",")
    text = text.replace(":", ",")

    text = text.replace("?", ".")
    text = text.replace("!", ".")

    text = text.replace("(", " ")
    text = text.replace(")", " ")

    text = text.replace("[", " ")
    text = text.replace("]", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = re.sub(
        r"\s+([.,])",
        r"\1",
        text
    )

    text = re.sub(
        r"([.,])(?=\S)",
        r"\1 ",
        text
    )

    text = re.sub(
        r",{2,}",
        ",",
        text
    )

    text = re.sub(
        r"\.{2,}",
        ".",
        text
    )

    return text.strip()


# ============================================================
# GENERATE VOICE
# ============================================================

async def generate_voice(text, audio_file):

    communicate = edge_tts.Communicate(
        text=text,
        voice=VOICE,
        rate=VOICE_RATE
    )

    await communicate.save(
        audio_file
    )


# ============================================================
# SPEAK
# ============================================================

def speak(text):

    speech_text = clean_for_speech(
        text
    )

    if not speech_text:
        return

    try:

        if os.path.exists(AUDIO_FILE):

            try:
                os.remove(AUDIO_FILE)

            except Exception:
                pass

        print("\n🔊 Generating voice...")

        asyncio.run(
            generate_voice(
                speech_text,
                AUDIO_FILE
            )
        )

        if not os.path.exists(AUDIO_FILE):

            print(
                "❌ Voice file was not created."
            )

            return

        if os.path.getsize(AUDIO_FILE) == 0:

            print(
                "❌ Voice file is empty."
            )

            os.remove(AUDIO_FILE)

            return

        print("🔊 Playing answer...")

        pygame.mixer.init()

        pygame.mixer.music.load(
            AUDIO_FILE
        )

        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():

            pygame.time.Clock().tick(10)

        pygame.mixer.music.unload()

        pygame.mixer.quit()

        if os.path.exists(AUDIO_FILE):
            os.remove(AUDIO_FILE)

    except Exception as error:

        print(
            f"\n❌ Voice error: {error}"
        )

        try:
            pygame.mixer.quit()
        except Exception:
            pass


# ============================================================
# MICROPHONE
# ============================================================

def setup_microphone():

    print("\n🎤 Preparing microphone...")

    try:

        with sr.Microphone() as source:

            print(
                "🎤 Calibrating for background noise..."
            )

            recognizer.adjust_for_ambient_noise(
                source,
                duration=1
            )

        # Make speech detection responsive.
        recognizer.pause_threshold = 0.6
        recognizer.phrase_threshold = 0.2
        recognizer.non_speaking_duration = 0.3

        print("✅ Microphone ready.")

    except Exception as error:

        print(
            f"⚠️ Microphone setup failed: {error}"
        )


# ============================================================
# LISTEN
# ============================================================

def listen():

    try:

        with sr.Microphone() as source:

            print("\n🎤 Listening...")

            audio = recognizer.listen(
                source,
                timeout=MIC_TIMEOUT,
                phrase_time_limit=PHRASE_TIME_LIMIT
            )

        print(
            "🧠 Converting speech to text..."
        )

        text = recognizer.recognize_google(
            audio
        )

        text = text.strip()

        if text:

            print(
                f"\nYou: {text}"
            )

        return text

    except sr.WaitTimeoutError:

        print(
            "⏱️ No speech detected."
        )

        return ""

    except sr.UnknownValueError:

        print(
            "❌ I couldn't understand that."
        )

        return ""

    except sr.RequestError as error:

        print(
            f"❌ Speech recognition error: {error}"
        )

        return ""

    except Exception as error:

        print(
            f"❌ Microphone error: {error}"
        )

        return ""


# ============================================================
# ASK STUDY AI
# ============================================================

def ask_ai(question):

    print("\n🔎 Searching your PDFs...")

    answer, sources = create_answer(
        question
    )

    if not answer:

        print(
            "\n🤖 Study AI:"
        )

        print(
            "I could not find relevant information "
            "in your PDF files."
        )

        print(
            "\n💡 Try asking the question using "
            "different words."
        )

        speak(
            "I could not find relevant information "
            "in your PDF files."
        )

        return

    print("\n🤖 Study AI:\n")

    print(answer)

    if sources:

        print("\n📚 Sources:")

        for filename, page in sources:

            print(
                f"   • {filename} "
                f"(page {page})"
            )

    print("\n🔊 Speaking...")

    speak(answer)


# ============================================================
# SHOW PDF INFORMATION
# ============================================================

def show_pdf_info():

    if not documents:

        print(
            "\n📚 No PDF content loaded."
        )

        return

    files = {}

    for document in documents:

        filename = document["filename"]

        if filename not in files:
            files[filename] = set()

        files[filename].add(
            document["page"]
        )

    print("\n📚 Your study material:")

    for filename, pages in files.items():

        print(
            f"   • {filename} "
            f"({len(pages)} pages)"
        )

    print(
        f"\n📖 Searchable sections: "
        f"{len(documents)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "📚 STUDY AI"
    )

    print(
        "PDF Search + Voice Assistant"
    )

    print("=" * 60)

    print(
        "\n🚫 No LLM"
    )

    print(
        "📖 Answers come directly from your PDFs."
    )

    # Load PDFs.
    load_pdfs()

    # Build search index.
    build_index()

    # Prepare microphone.
    setup_microphone()

    while True:

        print("\n" + "=" * 60)

        print("Choose input:")

        print("1 → 🎤 Speak")

        print("2 → ⌨️ Type")

        print("3 → 📚 Show PDFs")

        print("4 → 🔄 Reload PDFs")

        print("5 → 🚪 Exit")

        choice = input(
            "\nChoice: "
        ).strip()

        # ----------------------------------------------------
        # VOICE
        # ----------------------------------------------------

        if choice == "1":

            question = listen()

            if not question:
                continue

        # ----------------------------------------------------
        # TEXT
        # ----------------------------------------------------

        elif choice == "2":

            question = input(
                "\nYou: "
            ).strip()

            if not question:
                continue

        # ----------------------------------------------------
        # SHOW PDF INFO
        # ----------------------------------------------------

        elif choice == "3":

            show_pdf_info()

            continue

        # ----------------------------------------------------
        # RELOAD
        # ----------------------------------------------------

        elif choice == "4":

            reload_pdfs()

            continue

        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        elif choice == "5":

            print(
                "\n👋 Goodbye. Keep studying!"
            )

            break

        else:

            print(
                "\n❌ Please choose 1, 2, 3, 4, or 5."
            )

            continue

        # ----------------------------------------------------
        # SPECIAL COMMAND
        # ----------------------------------------------------

        if question.lower() == "reload":

            reload_pdfs()

            continue

        # ----------------------------------------------------
        # ASK
        # ----------------------------------------------------

        ask_ai(question)


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    main()