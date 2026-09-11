from pathlib import Path
import hashlib
import random
import re
from collections import Counter

import pypdf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# ANSWER-STYLE TUNING (no LLM anywhere below - pure TF-IDF + rules)

DEFAULT_MAX_WORDS = 150
BRIEF_MAX_WORDS = 60
DETAILED_MAX_WORDS = 300
EXAM_MAX_WORDS = 250

# rough rule of thumb for Indian exam answers: ~40 words per mark
WORDS_PER_MARK = 40
MIN_EXAM_WORDS = 50
MAX_EXAM_WORDS = 400

CANDIDATE_POOL = 40
MAX_SELECTED_SENTENCES = 8
MMR_LAMBDA = 0.7
SYNONYM_REPLACEMENT_RATE = 0.85

# Textbook narration that doesn't belong in an answer.
FILLER_PATTERNS = [
    r"^in this (chapter|section|unit|topic|lesson),?\s*",
    r"^as (discussed|mentioned|explained|stated) (earlier|above|previously|below)?,?\s*",
    r"^according to (the )?(text|passage|author|book|chapter),?\s*",
    r"^we (will|shall) (now )?(discuss|explore|look at|study)\s*",
    r"^let us (now )?(discuss|explore|understand|study)\s*",
    r"^it (is|was) (stated|mentioned|said) that\s*",
    r"^as (we|you) (can|could) see,?\s*",
    r"^note that\s*",
]

# Deliberately generic/connective vocabulary only - never subject-matter
# nouns - so technical terms from the PDF are never altered, only the
# phrasing around them.
SYNONYM_MAP = {
    "important": ["significant", "key", "essential"],
    "shows": ["indicates", "demonstrates"],
    "show": ["indicate", "demonstrate"],
    "because": ["since", "as"],
    "also": ["additionally", "furthermore"],
    "many": ["numerous", "several"],
    "used": ["utilized", "employed"],
    "helps": ["aids", "assists"],
    "helps in": ["aids in", "assists in"],
    "makes": ["creates", "produces"],
    "big": ["large", "substantial"],
    "small": ["minor", "limited"],
    "different": ["distinct", "varied"],
    "however": ["nevertheless", "on the other hand"],
    "therefore": ["thus", "hence", "as a result"],
    "example": ["instance", "illustration"],
    "process": ["procedure", "mechanism"],
    "result": ["outcome", "consequence"],
    "problem": ["issue", "challenge"],
    "method": ["approach", "technique"],
    "increase": ["rise", "growth"],
    "decrease": ["decline", "reduction"],
    "main": ["primary", "principal"],
    "study": ["research", "analysis"],
    "known as": ["referred to as", "called"],
    "in order to": ["so as to"],
    "such as": ["like"],
    "typically": ["generally", "usually"],
    "provide": ["offer", "supply"],
    "provides": ["offers", "supplies"],
    "consider": ["regard", "view"],
    "focus on": ["concentrate on"],
    "leads to": ["results in"],
    "part of": ["component of"],
    "type of": ["kind of", "form of"],
    "occurs": ["takes place", "happens"],
    "allows": ["enables", "permits"],
    "several": ["various", "multiple"],
    "similar": ["comparable"],
    "difference": ["distinction", "variation"],
    "requires": ["needs", "demands"],
    "contains": ["includes", "comprises"],
    "produces": ["generates"],
    "changes": ["alters", "modifies"],
    "affects": ["influences", "impacts"],
    "responsible for": ["the cause of"],
}


class StudyEngine:
    """
    PDF retrieval + answer engine for the Study AI Agent.

    Responsibilities:
    - Read PDFs
    - Clean extracted PDF text
    - Remove repeated headers/footers
    - Split text into useful chunks
    - Search using TF-IDF, with page-level diversity
    - Turn the retrieved material into a written, exam-ready answer -
      entirely locally, no LLM/external API call anywhere in this file.
    """

    def __init__(self, pdf_folder="pdfs"):
        self.pdf_folder = Path(pdf_folder)

        self.documents = []
        self.chunks = []

        self.vectorizer = None
        self.matrix = None

        self.loaded = False

    
    # TEXT CLEANING
    

    @staticmethod
    def clean_text(text):
        if not text:
            return ""

        text = text.replace("\x00", " ")
        text = text.replace("\r", "\n")

        # Fix common PDF line-break problems.
        # Example:
        # "photo-
        # synthesis"
        # becomes "photosynthesis"
        text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

        # Replace remaining line breaks with spaces.
        text = re.sub(r"\s*\n\s*", " ", text)

        # Remove repeated whitespace.
        text = re.sub(r"[ \t]+", " ", text)

        # Remove spaces before punctuation.
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)

        # Remove spaces after opening brackets.
        text = re.sub(r"([(\[])\s+", r"\1", text)

        # Remove spaces before closing brackets.
        text = re.sub(r"\s+([)\]])", r"\1", text)

        # Clean strange bullet characters.
        text = re.sub(r"[•●▪◦]", " ", text)

        # Normalize repeated punctuation.
        text = re.sub(r"\.{3,}", "...", text)
        text = re.sub(r"[|]{2,}", " ", text)

        return text.strip()

    @staticmethod
    def normalize_for_comparison(text):
        """
        Used for detecting duplicate/repeated PDF lines.
        """
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-z0-9 ]", "", text)
        return text.strip()

    
    # PDF HEADER / FOOTER CLEANING
    

    def remove_repeated_lines(self, pages):
        """
        Removes lines that occur on many pages.

        This helps with PDFs containing repeated:
        - chapter names
        - book titles
        - page headers
        - footer text
        """

        if len(pages) < 3:
            return pages

        line_counter = Counter()

        processed_pages = []

        for page in pages:
            lines = []

            for raw_line in page.split("\n"):
                line = raw_line.strip()

                if not line:
                    continue

                normalized = self.normalize_for_comparison(line)

                if 3 <= len(normalized) <= 100:
                    line_counter[normalized] += 1

                lines.append(line)

            processed_pages.append(lines)

        threshold = max(3, int(len(pages) * 0.45))

        repeated = {
            line
            for line, count in line_counter.items()
            if count >= threshold
        }

        result = []

        for lines in processed_pages:
            cleaned_lines = []

            for line in lines:
                normalized = self.normalize_for_comparison(line)

                if normalized in repeated:
                    continue

                # Remove isolated page numbers.
                if re.fullmatch(r"(page\s*)?\d{1,4}", normalized):
                    continue

                cleaned_lines.append(line)

            result.append("\n".join(cleaned_lines))

        return result

    
    # CHUNKING
    

    @staticmethod
    def split_into_chunks(text, chunk_size=180, overlap=35):
        """
        Splits text into overlapping word chunks.

        Overlap helps prevent important information from being
        split between two chunks.
        """

        words = text.split()

        if not words:
            return []

        # Guard against a bad overlap/chunk_size combo looping forever.
        overlap = min(overlap, chunk_size - 1) if chunk_size > 0 else 0

        chunks = []

        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))

            chunk = " ".join(words[start:end]).strip()

            if chunk:
                chunks.append(chunk)

            if end >= len(words):
                break

            start = end - overlap

            if start < 0:
                start = 0

        return chunks

    
    # SENTENCE SPLITTING
    

    @staticmethod
    def split_sentences(text):
        if not text:
            return []

        text = text.strip()

        # Protect common abbreviations.
        protected = {
            "e.g.": "eg<prd>",
            "i.e.": "ie<prd>",
            "etc.": "etc<prd>",
            "Mr.": "Mr<prd>",
            "Mrs.": "Mrs<prd>",
            "Dr.": "Dr<prd>",
            "Prof.": "Prof<prd>",
        }

        for old, new in protected.items():
            text = text.replace(old, new)

        # Split sentences.
        parts = re.split(
            r"(?<=[.!?])\s+(?=[A-Z0-9])",
            text
        )

        restored = []

        for sentence in parts:
            for old, new in protected.items():
                sentence = sentence.replace(new, old)

            sentence = re.sub(r"\s+", " ", sentence).strip()

            if sentence:
                restored.append(sentence)

        return restored

    
    # LOAD PDFS
    

    def load_pdfs(self):
        self.documents = []
        self.chunks = []

        self.pdf_folder.mkdir(parents=True, exist_ok=True)

        pdf_files = sorted(self.pdf_folder.glob("*.pdf"))

        for pdf_path in pdf_files:

            try:
                reader = pypdf.PdfReader(str(pdf_path))

            except Exception as error:
                print(f"Could not open {pdf_path.name}: {error}")
                continue

            raw_pages = []

            for page in reader.pages:

                try:
                    raw_text = page.extract_text() or ""

                except Exception:
                    raw_text = ""

                raw_pages.append(raw_text)

            # Remove repeated headers/footers before cleaning.
            cleaned_pages = self.remove_repeated_lines(raw_pages)

            for page_number, page_text in enumerate(
                cleaned_pages,
                start=1
            ):

                text = self.clean_text(page_text)

                if not text:
                    continue

                self.documents.append(
                    {
                        "filename": pdf_path.name,
                        "page": page_number,
                        "text": text,
                    }
                )

                chunks = self.split_into_chunks(
                    text,
                    chunk_size=180,
                    overlap=35
                )

                for chunk in chunks:

                    self.chunks.append(
                        {
                            "text": chunk,
                            "filename": pdf_path.name,
                            "page": page_number,
                        }
                    )

        self.loaded = True

        self.build_index()

        return self.get_library()

    
    # BUILD TF-IDF INDEX
    

    def build_index(self):

        if not self.chunks:
            self.vectorizer = None
            self.matrix = None
            return

        texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=50000
        )

        self.matrix = self.vectorizer.fit_transform(texts)

    
    # RELOAD
    

    def reload(self):

        self.documents = []
        self.chunks = []
        self.vectorizer = None
        self.matrix = None

        return self.load_pdfs()

    
    # SEARCH
    

    def search(self, question, top_k=8):

        if not question:
            return []

        if not self.chunks:
            return []

        if self.vectorizer is None or self.matrix is None:
            self.build_index()

        try:
            question_vector = self.vectorizer.transform(
                [question]
            )

            similarities = cosine_similarity(
                question_vector,
                self.matrix
            ).flatten()

        except Exception:
            return []

        ranked_indexes = similarities.argsort()[::-1]

        results = []

        for index in ranked_indexes:

            score = float(similarities[index])

            if score <= 0:
                continue

            result = dict(self.chunks[index])

            result["score"] = score

            results.append(result)

            if len(results) >= top_k:
                break

        return results

    
    # SEARCH WITH DIVERSITY
    

    def search_diverse(self, question, top_k=8):

        results = self.search(
            question,
            top_k=max(top_k * 3, 15)
        )

        if not results:
            return []

        selected = []

        page_seen = set()

        for result in results:

            key = (
                result["filename"],
                result["page"]
            )

            # Prefer different pages so the answer doesn't
            # copy one long section from a single page.
            if key not in page_seen:

                selected.append(result)
                page_seen.add(key)

            if len(selected) >= top_k:
                break

        # If there aren't enough unique pages, fill remaining slots.
        if len(selected) < top_k:

            for result in results:

                if result not in selected:
                    selected.append(result)

                if len(selected) >= top_k:
                    break

        return selected

    
    # ANSWER STYLE (brief / detailed / exam / default)
    

    def detect_marks(self, question):
        """'5 marks' / '10 marks' -> a target word count. None if absent."""
        match = re.search(r"(\d+)\s*marks?\b", question.lower())

        if not match:
            return None

        marks = int(match.group(1))
        words = marks * WORDS_PER_MARK
        return max(MIN_EXAM_WORDS, min(words, MAX_EXAM_WORDS))

    def get_answer_style(self, question):
        question_lower = question.lower()

        marks_words = self.detect_marks(question)
        if marks_words:
            return {"style": "exam", "max_words": marks_words}

        brief_words = ["brief", "briefly", "short", "shortly", "in short", "one line"]
        detailed_words = ["detailed", "detail", "deeply", "explain fully", "in detail"]
        exam_words = ["exam", "marks", "answer"]

        for phrase in brief_words:
            if phrase in question_lower:
                return {"style": "brief", "max_words": BRIEF_MAX_WORDS}

        for phrase in detailed_words:
            if phrase in question_lower:
                return {"style": "detailed", "max_words": DETAILED_MAX_WORDS}

        for phrase in exam_words:
            if phrase in question_lower:
                return {"style": "exam", "max_words": EXAM_MAX_WORDS}

        return {"style": "default", "max_words": DEFAULT_MAX_WORDS}

    
    # WORD LIMIT
    

    @staticmethod
    def limit_words(text, max_words):
        words = text.split()

        if len(words) <= max_words:
            return text.strip()

        shortened = " ".join(words[:max_words])

        last_period = shortened.rfind(".")
        last_question = shortened.rfind("?")
        last_exclamation = shortened.rfind("!")
        last_sentence = max(last_period, last_question, last_exclamation)

        if last_sentence > max_words * 3:
            shortened = shortened[:last_sentence + 1]
        else:
            shortened += "..."

        return shortened.strip()

    
    # FILLER STRIPPING + LOCAL PARAPHRASING (no LLM)
    

    @staticmethod
    def strip_filler(sentence):
        cleaned = sentence
        for pattern in FILLER_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()

        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]

        return cleaned or sentence

    @staticmethod
    def _match_case(original, replacement):
        if original.isupper():
            return replacement.upper()
        if original[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement

    def paraphrase_sentence(self, sentence, rng):
        """Swap generic connective/descriptive words/phrases for a
        synonym. SYNONYM_MAP only contains generic vocabulary, never
        subject-matter nouns, so facts/technical terms are never touched -
        only the phrasing around them changes."""
        result = sentence

        for phrase, synonyms in SYNONYM_MAP.items():
            pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)

            def replace(match, synonyms=synonyms):
                if rng.random() > SYNONYM_REPLACEMENT_RATE:
                    return match.group(0)
                choice = rng.choice(synonyms)
                return self._match_case(match.group(0), choice)

            result = pattern.sub(replace, result)

        return result

    
    # DIVERSE SENTENCE SELECTION (Maximal Marginal Relevance)
    # search_diverse() already spreads results across different pages;
    # this adds a second, semantic layer of diversity at the sentence
    # level, since two different pages can still repeat the same
    # sentence (e.g. a definition restated in two chapters).
    

    def select_diverse_sentences(self, candidates, candidate_vectors, max_words):
        selected_idx = []
        remaining = list(range(len(candidates)))
        word_count = 0

        while remaining and word_count < max_words and len(selected_idx) < MAX_SELECTED_SENTENCES:
            best_pos = None
            best_score = None

            for pos in remaining:
                relevance = candidates[pos]["score"]

                if selected_idx:
                    sims = cosine_similarity(
                        candidate_vectors[pos], candidate_vectors[selected_idx]
                    ).flatten()
                    redundancy = float(sims.max())
                else:
                    redundancy = 0.0

                mmr_score = MMR_LAMBDA * relevance - (1 - MMR_LAMBDA) * redundancy

                if best_score is None or mmr_score > best_score:
                    best_score = mmr_score
                    best_pos = pos

            words_in_sentence = len(candidates[best_pos]["text"].split())

            if selected_idx and word_count + words_in_sentence > max_words:
                remaining.remove(best_pos)
                continue

            selected_idx.append(best_pos)
            word_count += words_in_sentence
            remaining.remove(best_pos)

        return [candidates[pos] for pos in selected_idx]

    @staticmethod
    def format_answer(style, display_texts):
        if not display_texts:
            return ""

        if style == "exam":
            return "\n".join(f"{i + 1}. {text}" for i, text in enumerate(display_texts))

        if len(display_texts) > 4:
            midpoint = (len(display_texts) + 1) // 2
            first_half = " ".join(display_texts[:midpoint])
            second_half = " ".join(display_texts[midpoint:])
            return first_half + "\n\n" + second_half

        return " ".join(display_texts)

    
    # CREATE ANSWER - the missing piece: turns retrieved chunks into
    # an actual written answer. No LLM/API call anywhere in here.
    

    def create_answer(self, question):
        style_info = self.get_answer_style(question)
        style = style_info["style"]
        max_words = style_info["max_words"]

        results = self.search_diverse(question, top_k=8)

        if not results:
            return None, []

        all_sentences = []

        for result in results:
            for sentence in self.split_sentences(result["text"]):
                if len(sentence.split()) < 4:
                    continue

                all_sentences.append({
                    "text": sentence,
                    "filename": result["filename"],
                    "page": result["page"],
                })

        if not all_sentences:
            return None, []

        # drop exact/near-identical raw sentences before scoring
        seen_norm = set()
        deduped = []
        for item in all_sentences:
            normalized = self.normalize_for_comparison(item["text"])
            if normalized in seen_norm:
                continue
            seen_norm.add(normalized)
            deduped.append(item)
        all_sentences = deduped

        sentence_texts = [item["text"] for item in all_sentences]

        try:
            sentence_vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True,
            )
            sentence_matrix = sentence_vectorizer.fit_transform(sentence_texts + [question])
            question_vector = sentence_matrix[-1]
            doc_vectors = sentence_matrix[:-1]
            scores = cosine_similarity(question_vector, doc_vectors).flatten()
        except Exception:
            scores = [0.0 for _ in all_sentences]
            doc_vectors = None

        for i, item in enumerate(all_sentences):
            item["score"] = float(scores[i])
            item["order"] = i

        candidates = sorted(
            [item for item in all_sentences if item["score"] > 0],
            key=lambda x: x["score"],
            reverse=True,
        )[:CANDIDATE_POOL]

        if not candidates:
            candidates = sorted(all_sentences, key=lambda x: x["score"], reverse=True)[:1]

        if doc_vectors is not None and len(candidates) > 1:
            candidate_positions = [item["order"] for item in candidates]
            candidate_vectors = doc_vectors[candidate_positions]
            selected = self.select_diverse_sentences(candidates, candidate_vectors, max_words)
        else:
            selected = candidates[:1]

        if not selected:
            selected = candidates[:1]

        # restore original document order so the answer reads naturally
        selected.sort(key=lambda x: x["order"])

        # deterministic per-question seed: same question -> same wording
        seed = int(hashlib.md5(question.encode("utf-8")).hexdigest(), 16) % (2 ** 32)
        rng = random.Random(seed)

        display_texts = []
        for item in selected:
            text = self.strip_filler(item["text"])
            text = self.paraphrase_sentence(text, rng)
            display_texts.append(text)

        answer = self.format_answer(style, display_texts)
        answer = self.limit_words(answer, max_words + 20)

        sources = []
        for item in selected:
            key = (item["filename"], item["page"])
            if key not in sources:
                sources.append(key)

        return answer, sources

    
    # LIBRARY
    

    def get_library(self):

        files = {}

        for document in self.documents:

            filename = document["filename"]

            files[filename] = files.get(
                filename,
                0
            ) + 1

        return [
            {
                "filename": filename,
                "pages": page_count
            }
            for filename, page_count in sorted(
                files.items()
            )
        ]

    
    # DELETE PDF
    

    def delete_pdf(self, filename):

        pdf_path = self.pdf_folder / filename

        if pdf_path.exists():

            try:
                pdf_path.unlink()

            except Exception as error:
                raise RuntimeError(
                    f"Could not delete PDF: {error}"
                )

        self.reload()

    
    # ADD PDF
    

    def add_pdf(self, source_path):

        source = Path(source_path)

        if not source.exists():
            raise FileNotFoundError(
                f"File not found: {source}"
            )

        if source.suffix.lower() != ".pdf":
            raise ValueError(
                "Only PDF files are supported."
            )

        self.pdf_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        destination = self.pdf_folder / source.name

        # Avoid overwriting unexpectedly.
        if destination.exists():

            stem = destination.stem
            suffix = destination.suffix

            counter = 2

            while destination.exists():

                destination = (
                    self.pdf_folder
                    / f"{stem}_{counter}{suffix}"
                )

                counter += 1

        destination.write_bytes(
            source.read_bytes()
        )

        self.reload()

        return destination.name