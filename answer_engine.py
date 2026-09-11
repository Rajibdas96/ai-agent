import hashlib
import random
import re
from collections import OrderedDict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Textbook narration that doesn't belong in a written answer.
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
SYNONYM_REPLACEMENT_RATE = 0.85


class AnswerEngine:
    """
    Model-free exam answer generator.

    It does NOT generate new factual information.

    Instead it:
    1. Retrieves relevant PDF information.
    2. Extracts useful sentences.
    3. Removes duplicates.
    4. Identifies the question type.
    5. Organizes information into an exam-friendly structure.
    6. Lightly rewords connective language so it doesn't read as a
       straight paste of PDF sentences (no LLM involved - see
       strip_filler / paraphrase_sentence below).
    """

    ANSWER_TYPES = [
        "Auto",
        "Short",
        "Definition",
        "Brief",
        "Exam",
        "Explain",
        "Advantages",
        "Disadvantages",
        "Difference",
        "Compare",
        "Steps",
    ]

    MARK_OPTIONS = [
        "Auto",
        "2 Marks",
        "3 Marks",
        "5 Marks",
        "10 Marks",
        "15 Marks",
    ]

    def __init__(self):

        self.lengths = {
            "2 Marks": 60,
            "3 Marks": 100,
            "5 Marks": 180,
            "10 Marks": 300,
            "15 Marks": 450,
        }

    
    # BASIC CLEANING
    

    @staticmethod
    def clean_sentence(sentence):

        if not sentence:
            return ""

        sentence = sentence.strip()

        # Remove accidental bullets.
        sentence = re.sub(
            r"^[•●▪◦\-*]+\s*",
            "",
            sentence
        )

        # Remove excessive whitespace.
        sentence = re.sub(
            r"\s+",
            " ",
            sentence
        )

        # Fix spaces before punctuation.
        sentence = re.sub(
            r"\s+([,.;:!?])",
            r"\1",
            sentence
        )

        return sentence.strip()

    
    # SENTENCE SPLITTING
    

    @staticmethod
    def split_sentences(text):

        if not text:
            return []

        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        parts = re.split(
            r"(?<=[.!?])\s+(?=[A-Z0-9])",
            text
        )

        result = []

        for part in parts:

            part = AnswerEngine.clean_sentence(part)

            if not part:
                continue

            # Avoid useless tiny fragments.
            if len(part.split()) < 4:
                continue

            result.append(part)

        return result

    
    # DUPLICATE REMOVAL
    

    @staticmethod
    def normalize(sentence):

        sentence = sentence.lower()

        sentence = re.sub(
            r"[^a-z0-9\s]",
            "",
            sentence
        )

        sentence = re.sub(
            r"\s+",
            " ",
            sentence
        )

        return sentence.strip()

    def remove_duplicates(self, sentences):

        unique = OrderedDict()

        for sentence in sentences:

            normalized = self.normalize(sentence)

            if not normalized:
                continue

            # Exact duplicate.
            if normalized in unique:
                continue

            unique[normalized] = sentence

        return list(unique.values())

    
    # LOCAL PARAPHRASING (no LLM)
    # Applied only at the point text is written into the final
    # answer - rank_sentences / find_definition always see the raw
    # PDF text, so indicator-phrase matching stays accurate.
    

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

    def _polish(self, sentence, rng):
        cleaned = self.strip_filler(sentence)
        cleaned = self.paraphrase_sentence(cleaned, rng)
        return cleaned

    
    # QUESTION TYPE DETECTION
    

    def detect_answer_type(self, question):

        q = question.lower().strip()

        # Difference / comparison
        if any(
            phrase in q
            for phrase in [
                "difference between",
                "differentiate between",
                "differentiate",
                "distinguish between",
                "distinguish",
                "compare",
                "comparison between",
            ]
        ):
            return "Difference"

        # Advantages
        if any(
            word in q
            for word in [
                "advantages",
                "benefits",
                "merits",
                "importance",
            ]
        ):
            return "Advantages"

        # Disadvantages
        if any(
            word in q
            for word in [
                "disadvantages",
                "limitations",
                "demerits",
                "drawbacks",
            ]
        ):
            return "Disadvantages"

        # Steps / process
        if any(
            phrase in q
            for phrase in [
                "steps",
                "step by step",
                "procedure",
                "process of",
                "how does",
                "how do",
                "method of",
                "working of",
            ]
        ):
            return "Steps"

        # Definition
        if any(
            phrase in q
            for phrase in [
                "what is",
                "what are",
                "define",
                "definition of",
                "meaning of",
                "meaning",
            ]
        ):
            return "Definition"

        # Explain
        if any(
            phrase in q
            for phrase in [
                "explain",
                "describe",
                "discuss",
                "elaborate",
            ]
        ):
            return "Explain"

        return "Exam"

    
    # DETERMINE WORD LIMIT
    
    def get_word_limit(self, answer_type, marks):

        if marks in self.lengths:
            return self.lengths[marks]

        defaults = {
            "Short": 70,
            "Definition": 90,
            "Brief": 120,
            "Explain": 180,
            "Advantages": 180,
            "Disadvantages": 180,
            "Difference": 220,
            "Compare": 220,
            "Steps": 220,
            "Exam": 220,
        }

        return defaults.get(
            answer_type,
            180
        )

    
    # REMOVE QUESTION-LIKE / BAD SENTENCES
    

    @staticmethod
    def is_good_sentence(sentence):

        if not sentence:
            return False

        words = sentence.split()

        if len(words) < 5:
            return False

        if len(words) > 80:
            return False

        # Ignore obvious PDF navigation.
        bad_patterns = [
            r"^chapter\s+\d+$",
            r"^unit\s+\d+$",
            r"^page\s+\d+$",
            r"^contents$",
            r"^references$",
            r"^bibliography$",
        ]

        lowered = sentence.lower().strip()

        for pattern in bad_patterns:

            if re.fullmatch(pattern, lowered):
                return False

        return True

    
    # RANK SENTENCES
    

    def rank_sentences(self, question, results):

        candidates = []

        for result in results:

            sentences = self.split_sentences(
                result.get("text", "")
            )

            for sentence in sentences:

                if not self.is_good_sentence(sentence):
                    continue

                candidates.append(
                    {
                        "text": sentence,
                        "filename": result.get(
                            "filename",
                            "Unknown"
                        ),
                        "page": result.get(
                            "page",
                            "?"
                        ),
                        "chunk_score": result.get(
                            "score",
                            0
                        ),
                    }
                )

        if not candidates:
            return []

        # Remove exact duplicates before ranking.
        seen = set()
        filtered = []

        for candidate in candidates:

            key = self.normalize(
                candidate["text"]
            )

            if key in seen:
                continue

            seen.add(key)
            filtered.append(candidate)

        candidates = filtered

        texts = [
            item["text"]
            for item in candidates
        ]

        try:

            vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True
            )

            matrix = vectorizer.fit_transform(
                [question] + texts
            )

            similarities = cosine_similarity(
                matrix[0:1],
                matrix[1:]
            ).flatten()

        except Exception:

            similarities = [
                item["chunk_score"]
                for item in candidates
            ]

        for index, item in enumerate(candidates):

            similarity = float(
                similarities[index]
            )

            # Small bonus for information-rich sentences.
            word_count = len(
                item["text"].split()
            )

            length_bonus = min(
                word_count / 30,
                1.0
            )

            item["sentence_score"] = (
                similarity * 0.80
                + item["chunk_score"] * 0.15
                + length_bonus * 0.05
            )

        candidates.sort(
            key=lambda x: x["sentence_score"],
            reverse=True
        )

        return candidates

    
    # WORD LIMIT
    

    @staticmethod
    def limit_words(text, max_words):

        words = text.split()

        if len(words) <= max_words:
            return text

        shortened = " ".join(
            words[:max_words]
        )

        # Don't end halfway through a sentence if possible.
        last_stop = max(
            shortened.rfind("."),
            shortened.rfind("!"),
            shortened.rfind("?")
        )

        if last_stop > len(shortened) * 0.60:
            shortened = shortened[
                :last_stop + 1
            ]

        else:
            shortened += "..."

        return shortened.strip()

    
    # EXTRACT KEY SENTENCES
    

    def select_sentences(
        self,
        ranked,
        max_words,
        max_sentences=8
    ):

        selected = []

        used_words = 0

        for item in ranked:

            sentence = item["text"]

            words = len(
                sentence.split()
            )

            if used_words + words > max_words:
                continue

            # Avoid sentences that are almost duplicates.
            duplicate = False

            current_normalized = set(
                self.normalize(sentence).split()
            )

            for existing in selected:

                existing_normalized = set(
                    self.normalize(existing["text"]).split()
                )

                if not current_normalized:
                    continue

                overlap = (
                    len(
                        current_normalized
                        & existing_normalized
                    )
                    / len(current_normalized)
                )

                if overlap >= 0.75:
                    duplicate = True
                    break

            if duplicate:
                continue

            selected.append(item)

            used_words += words

            if len(selected) >= max_sentences:
                break

        return selected

    
    # FIND DEFINITION SENTENCE
    

    def find_definition(self, question, ranked):
        """Pick the most definition-like sentence out of `ranked`.

        Strong indicators ("is defined as", "refers to"...) count no
        matter where they appear. Weak indicators ("is a", "is an",
        "means") only count near the start of the sentence - those
        phrases show up constantly in ordinary sentences that aren't
        definitions at all, so counting them anywhere was flagging
        almost anything as a "definition".
        """

        question_words = set(
            re.findall(
                r"[a-zA-Z]{3,}",
                question.lower()
            )
        )

        strong_phrases = [
            "is defined as",
            "can be defined as",
            "is the process",
            "refers to",
            "is known as",
            "is called",
        ]

        weak_phrases = [
            "is a",
            "is an",
            "means",
        ]

        candidates = []

        for item in ranked:

            sentence = item["text"]

            lowered = sentence.lower()

            score = item["sentence_score"]

            matched_strong = False

            for phrase in strong_phrases:

                if phrase in lowered:
                    score += 0.35
                    matched_strong = True
                    break

            if not matched_strong:

                lead_words = " ".join(lowered.split()[:6])

                for phrase in weak_phrases:

                    if phrase in lead_words:
                        score += 0.15
                        break

            # Reward question terms.
            sentence_words = set(
                re.findall(
                    r"[a-zA-Z]{3,}",
                    lowered
                )
            )

            overlap = len(
                question_words
                & sentence_words
            )

            score += min(
                overlap * 0.03,
                0.20
            )

            candidates.append(
                (score, item)
            )

        candidates.sort(
            key=lambda x: x[0],
            reverse=True
        )

        if candidates:
            return candidates[0][1]

        return None

    
    # TITLE
    

    def make_title(self, question):

        q = question.strip()

        # Remove common question prefixes.
        q = re.sub(
            r"^(what is|what are|define|explain|describe|"
            r"discuss|write about|tell me about)\s+",
            "",
            q,
            flags=re.IGNORECASE
        )

        # Strip a trailing marks annotation so it doesn't leak into
        # the heading - "...for 5 marks" / "...(5 Marks)".
        q = re.sub(
            r"[\(\[]?\s*(?:for\s+)?\d+\s*marks?\s*[\)\]]?\s*$",
            "",
            q,
            flags=re.IGNORECASE
        )

        q = q.rstrip("?").strip()

        if not q:
            return "Answer"

        return q[0].upper() + q[1:]

    
    # FORMAT BULLETS
    

    @staticmethod
    def bullet_list(sentences):

        return "\n".join(
            f"• {sentence}"
            for sentence in sentences
        )

    
    # BUILD DEFINITION ANSWER
    

    def build_definition(
        self,
        question,
        ranked,
        max_words,
        rng
    ):

        selected = self.select_sentences(
            ranked,
            max_words,
            max_sentences=5
        )

        if not selected:
            return ""

        definition = self.find_definition(
            question,
            selected
        )

        if not definition:

            body = " ".join(
                self._polish(item["text"], rng)
                for item in selected[:3]
            )

            return self.limit_words(
                body,
                max_words
            )

        answer = self._polish(definition["text"], rng)

        additional = []

        for item in selected:

            if item is definition:
                continue

            additional.append(
                self._polish(item["text"], rng)
            )

            if len(additional) >= 2:
                break

        if additional:
            answer += "\n\n" + (
                " ".join(additional)
            )

        return self.limit_words(
            answer,
            max_words
        )

    
    # BUILD SHORT ANSWER
    

    def build_short(
        self,
        question,
        ranked,
        max_words,
        rng
    ):

        selected = self.select_sentences(
            ranked,
            max_words,
            max_sentences=3
        )

        answer = " ".join(
            self._polish(item["text"], rng)
            for item in selected
        )

        return self.limit_words(
            answer,
            max_words
        )

    
    # BUILD EXAM ANSWER
    

    def build_exam(
        self,
        question,
        ranked,
        max_words,
        marks,
        rng
    ):

        if not ranked:
            return ""

        title = self.make_title(question)

        selected = self.select_sentences(
            ranked,
            max_words,
            max_sentences=8
        )

        if not selected:
            return ""

        # Look for the definition within the already-budgeted
        # selection, not the full ranked list - otherwise the intro
        # could come from a sentence that was never actually
        # budgeted for, pushing the real body content out when the
        # final word limit is applied.
        definition = self.find_definition(
            question,
            selected
        )

        body_items = []

        for item in selected:

            if definition is not None:
                if item["text"] == definition["text"]:
                    continue

            body_items.append(item)

        answer_parts = []

        answer_parts.append(
            title
        )

        # ---------------- INTRODUCTION ----------------

        if definition:

            intro = self._polish(definition["text"], rng)

            answer_parts.append(
                f"Introduction\n{intro}"
            )

        else:

            intro = self._polish(selected[0]["text"], rng)

            answer_parts.append(
                f"Introduction\n{intro}"
            )

            body_items = selected[1:]

        # ---------------- MAIN POINTS ----------------

        if body_items:

            points = []

            for item in body_items[:6]:

                points.append(
                    self._polish(item["text"], rng)
                )

            answer_parts.append(
                "Key Points\n"
                + self.bullet_list(points)
            )

        # ---------------- CONCLUSION ----------------

        if marks in [
            "5 Marks",
            "10 Marks",
            "15 Marks"
        ]:

            conclusion_source = None

            if len(selected) >= 3:

                candidate_conclusion = selected[-1]["text"]

                already_shown = {definition["text"]} if definition else set()
                already_shown.update(item["text"] for item in body_items[:6])

                if candidate_conclusion not in already_shown:
                    conclusion_source = candidate_conclusion

            if conclusion_source:

                answer_parts.append(
                    "Conclusion\n"
                    + self._polish(conclusion_source, rng)
                )

        answer = "\n\n".join(
            answer_parts
        )

        return self.limit_formatted_words(
            answer,
            max_words
        )

    
    # BUILD ADVANTAGES / DISADVANTAGES
    

    def build_points_answer(
        self,
        question,
        ranked,
        max_words,
        heading,
        rng
    ):

        title = self.make_title(question)

        selected = self.select_sentences(
            ranked,
            max_words,
            max_sentences=8
        )

        if not selected:
            return ""

        points = [
            self._polish(item["text"], rng)
            for item in selected
        ]

        answer = (
            f"{title}\n\n"
            f"Introduction\n"
            f"{points[0]}\n\n"
            f"{heading}\n"
            f"{self.bullet_list(points[1:] or points)}"
        )

        return self.limit_formatted_words(
            answer,
            max_words
        )

    
    # BUILD STEPS ANSWER
    

    def build_steps(
        self,
        question,
        ranked,
        max_words,
        rng
    ):

        title = self.make_title(question)

        selected = self.select_sentences(
            ranked,
            max_words,
            max_sentences=8
        )

        if not selected:
            return ""

        steps = []

        for item in selected:

            steps.append(
                self._polish(item["text"], rng)
            )

        numbered = "\n".join(
            f"{index + 1}. {sentence}"
            for index, sentence
            in enumerate(steps)
        )

        answer = (
            f"{title}\n\n"
            f"Process / Steps\n"
            f"{numbered}"
        )

        return self.limit_formatted_words(
            answer,
            max_words
        )

    
    # BUILD DIFFERENCE ANSWER
    

    def build_difference(
        self,
        question,
        ranked,
        max_words,
        rng
    ):

        title = self.make_title(question)

        selected = self.select_sentences(
            ranked,
            max_words,
            max_sentences=8
        )

        if not selected:
            return ""

        answer_lines = [
            f"{title}",
            "",
            "Key Differences"
        ]

        for index, item in enumerate(selected):

            answer_lines.append(
                f"{index + 1}. {self._polish(item['text'], rng)}"
            )

        answer = "\n".join(
            answer_lines
        )

        return self.limit_formatted_words(
            answer,
            max_words
        )

    
    # FORMATTED WORD LIMIT
    

    @staticmethod
    def limit_formatted_words(
        text,
        max_words
    ):
        """Trim to `max_words`, but only by dropping whole lines.

        The old version counted words but sliced mid-line, so a
        bulleted "Key Points" section could get cut off in the
        middle of a bullet. Keeping whole lines means a bullet or
        numbered point is either fully there or not shown at all -
        and a section heading left with nothing under it gets
        dropped too, instead of ending on a bare "Key Points" label.
        """

        lines = text.split("\n")

        result_lines = []
        count = 0

        for line in lines:

            words_in_line = line.split()

            if words_in_line and words_in_line[0] == "•":
                words_in_line = words_in_line[1:]

            line_word_count = len(words_in_line)

            if result_lines and count + line_word_count > max_words:
                break

            result_lines.append(line)
            count += line_word_count

        def is_dangling_heading(line):
            stripped = line.strip()

            if not stripped:
                return False

            if stripped.startswith(("•", "-", "*")):
                return False

            if re.match(r"^\d+\.\s", stripped):
                return False

            if stripped[-1] in ".!?":
                return False

            return len(stripped.split()) <= 4

        while result_lines and is_dangling_heading(result_lines[-1]):

            result_lines.pop()

            while result_lines and not result_lines[-1].strip():
                result_lines.pop()

        output = "\n".join(result_lines).strip()

        # Clean accidental trailing punctuation issues.
        output = re.sub(
            r"\s+([.,!?])",
            r"\1",
            output
        )

        return output

    
    # SOURCES
    

    @staticmethod
    def get_sources(ranked):

        sources = []

        seen = set()

        for item in ranked:

            filename = item.get(
                "filename",
                "Unknown"
            )

            page = item.get(
                "page",
                "?"
            )

            key = (
                filename,
                page
            )

            if key in seen:
                continue

            seen.add(key)

            sources.append(
                {
                    "filename": filename,
                    "page": page
                }
            )

            if len(sources) >= 5:
                break

        return sources

    
    # MAIN ANSWER FUNCTION
    

    def create_answer(
        self,
        question,
        search_results,
        answer_type="Auto",
        marks="Auto"
    ):

        if not question:
            return {
                "answer": "",
                "sources": [],
                "answer_type": "Auto",
                "marks": marks,
            }

        if not search_results:
            return {
                "answer": (
                    "I couldn't find enough relevant "
                    "information in your PDFs."
                ),
                "sources": [],
                "answer_type": answer_type,
                "marks": marks,
            }

        # Automatically detect answer type.
        if answer_type == "Auto":
            detected_type = self.detect_answer_type(
                question
            )
        else:
            detected_type = answer_type

        # Marks can override general answer length.
        max_words = self.get_word_limit(
            detected_type,
            marks
        )

        ranked = self.rank_sentences(
            question,
            search_results
        )

        if not ranked:
            return {
                "answer": (
                    "I found the PDF material, "
                    "but couldn't extract a useful answer."
                ),
                "sources": [],
                "answer_type": detected_type,
                "marks": marks,
            }

        # Deterministic per-question seed for the local paraphrasing
        # step, so re-running the same question gives the same
        # wording without needing real randomness or any API call.
        seed = int(
            hashlib.md5(question.encode("utf-8")).hexdigest(), 16
        ) % (2 ** 32)

        rng = random.Random(seed)

        
        # ANSWER TYPE ROUTING
        
        if detected_type == "Definition":

            answer = self.build_definition(
                question,
                ranked,
                max_words,
                rng
            )

        elif detected_type == "Short":

            answer = self.build_short(
                question,
                ranked,
                max_words,
                rng
            )

        elif detected_type == "Brief":

            answer = self.build_short(
                question,
                ranked,
                max_words,
                rng
            )

        elif detected_type == "Explain":

            answer = self.build_exam(
                question,
                ranked,
                max_words,
                marks,
                rng
            )

        elif detected_type == "Advantages":

            answer = self.build_points_answer(
                question,
                ranked,
                max_words,
                "Advantages",
                rng
            )

        elif detected_type == "Disadvantages":

            answer = self.build_points_answer(
                question,
                ranked,
                max_words,
                "Disadvantages",
                rng
            )

        elif detected_type in [
            "Difference",
            "Compare"
        ]:

            answer = self.build_difference(
                question,
                ranked,
                max_words,
                rng
            )

        elif detected_type == "Steps":

            answer = self.build_steps(
                question,
                ranked,
                max_words,
                rng
            )

        else:

            answer = self.build_exam(
                question,
                ranked,
                max_words,
                marks,
                rng
            )

        sources = self.get_sources(
            ranked
        )

        return {
            "answer": answer.strip(),
            "sources": sources,
            "answer_type": detected_type,
            "marks": marks,
        }