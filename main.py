import sys
import re
from pathlib import Path

from PySide6.QtCore import (
    Qt,
    QObject,
    QRunnable,
    QThreadPool,
    Signal,
)
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from study_engi import StudyEngine
from answer_engine import AnswerEngine

try:
    from voice_engine import VoiceEngine
    VOICE_AVAILABLE = True
except Exception:
    VoiceEngine = None
    VOICE_AVAILABLE = False


# ================================================================
# PATHS
# ================================================================

BASE_DIR = Path(__file__).resolve().parent
PDF_FOLDER = BASE_DIR / "pdfs"


# ================================================================
# WORKER SIGNALS
# ================================================================

class WorkerSignals(QObject):

    finished = Signal(object)
    error = Signal(str)


# ================================================================
# BACKGROUND WORKER
# ================================================================

class Worker(QRunnable):

    def __init__(self, function):

        super().__init__()

        self.function = function
        self.signals = WorkerSignals()

    def run(self):

        try:

            result = self.function()

            self.signals.finished.emit(
                result
            )

        except Exception as error:

            self.signals.error.emit(
                str(error)
            )


# ================================================================
# MAIN WINDOW
# ================================================================

class StudyAI(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "Study AI"
        )

        self.resize(
            1250,
            780
        )

        self.thread_pool = QThreadPool.globalInstance()

        self.engine = StudyEngine(
            PDF_FOLDER
        )

        self.answer_engine = AnswerEngine()

        self.voice = None
        self.voice_available = False
        self.voice_speaking = False

        # --------------------------------------------------------
        # VOICE
        # --------------------------------------------------------

        if VOICE_AVAILABLE:

            try:

                self.voice = VoiceEngine()

                self.voice_available = True

            except Exception as error:

                print(
                    f"Voice unavailable: {error}"
                )

                self.voice_available = False

        # --------------------------------------------------------
        # BUILD UI
        # --------------------------------------------------------

        self.build_ui()
        self.apply_styles()

        self.set_status(
            "Loading your PDF library..."
        )

        self.load_library_async()

    # ============================================================
    # UI
    # ============================================================

    def build_ui(self):

        central = QWidget()

        self.setCentralWidget(
            central
        )

        main_layout = QHBoxLayout(
            central
        )

        main_layout.setContentsMargins(
            0,
            0,
            0,
            0
        )

        main_layout.setSpacing(
            0
        )

        # ========================================================
        # SIDEBAR
        # ========================================================

        sidebar = QFrame()

        sidebar.setObjectName(
            "sidebar"
        )

        sidebar.setFixedWidth(
            290
        )

        sidebar_layout = QVBoxLayout(
            sidebar
        )

        sidebar_layout.setContentsMargins(
            20,
            25,
            20,
            20
        )

        # Logo
        logo = QLabel(
            "Study AI"
        )

        logo.setObjectName(
            "logo"
        )

        sidebar_layout.addWidget(
            logo
        )

        subtitle = QLabel(
            "Your PDF Study Assistant"
        )

        subtitle.setObjectName(
            "sidebarSubtitle"
        )

        sidebar_layout.addWidget(
            subtitle
        )

        sidebar_layout.addSpacing(
            25
        )

        # Library heading
        library_title = QLabel(
            "YOUR LIBRARY"
        )

        library_title.setObjectName(
            "sectionTitle"
        )

        sidebar_layout.addWidget(
            library_title
        )

        # PDF list
        self.library_list = QListWidget()

        self.library_list.setObjectName(
            "libraryList"
        )

        sidebar_layout.addWidget(
            self.library_list,
            1
        )

        # Buttons
        self.add_button = QPushButton(
            "+  Add PDF"
        )

        self.add_button.clicked.connect(
            self.add_pdf
        )

        sidebar_layout.addWidget(
            self.add_button
        )

        self.delete_button = QPushButton(
            "Delete Selected"
        )

        self.delete_button.clicked.connect(
            self.delete_selected_pdf
        )

        sidebar_layout.addWidget(
            self.delete_button
        )

        self.reload_button = QPushButton(
            "↻  Reload Library"
        )

        self.reload_button.clicked.connect(
            self.reload_library
        )

        sidebar_layout.addWidget(
            self.reload_button
        )

        main_layout.addWidget(
            sidebar
        )

        # ========================================================
        # CONTENT AREA
        # ========================================================

        content = QWidget()

        content_layout = QVBoxLayout(
            content
        )

        content_layout.setContentsMargins(
            35,
            30,
            35,
            25
        )

        content_layout.setSpacing(
            18
        )

        # --------------------------------------------------------
        # HEADER
        # --------------------------------------------------------

        header_layout = QHBoxLayout()

        title = QLabel(
            "Ask your study material"
        )

        title.setObjectName(
            "pageTitle"
        )

        header_layout.addWidget(
            title
        )

        header_layout.addStretch()

        self.status_label = QLabel(
            "Ready"
        )

        self.status_label.setObjectName(
            "statusLabel"
        )

        header_layout.addWidget(
            self.status_label
        )

        content_layout.addLayout(
            header_layout
        )

        # ========================================================
        # QUESTION CARD
        # ========================================================

        question_card = QFrame()

        question_card.setObjectName(
            "card"
        )

        question_layout = QVBoxLayout(
            question_card
        )

        question_layout.setContentsMargins(
            22,
            20,
            22,
            20
        )

        # Question heading
        question_label = QLabel(
            "Question"
        )

        question_label.setObjectName(
            "cardTitle"
        )

        question_layout.addWidget(
            question_label
        )

        # Input row
        input_row = QHBoxLayout()

        self.question_input = QTextEdit()

        self.question_input.setPlaceholderText(
            "Ask something from your PDFs..."
        )

        self.question_input.setFixedHeight(
            95
        )

        input_row.addWidget(
            self.question_input,
            1
        )

        # Voice
        self.voice_button = QPushButton(
            "🎤"
        )

        self.voice_button.setObjectName(
            "voiceButton"
        )

        self.voice_button.setFixedSize(
            55,
            55
        )

        self.voice_button.setToolTip(
            "Ask using your microphone"
        )

        self.voice_button.clicked.connect(
            self.start_voice_question
        )

        input_row.addWidget(
            self.voice_button,
            alignment=Qt.AlignBottom
        )

        question_layout.addLayout(
            input_row
        )

        # --------------------------------------------------------
        # OPTIONS
        # --------------------------------------------------------

        options_row = QHBoxLayout()

        # Answer type
        answer_type_label = QLabel(
            "Answer Type"
        )

        answer_type_label.setObjectName(
            "optionLabel"
        )

        options_row.addWidget(
            answer_type_label
        )

        self.answer_type_combo = QComboBox()

        self.answer_type_combo.addItems(
            self.answer_engine.ANSWER_TYPES
        )

        self.answer_type_combo.setCurrentText(
            "Auto"
        )

        options_row.addWidget(
            self.answer_type_combo
        )

        options_row.addSpacing(
            20
        )

        # Marks
        marks_label = QLabel(
            "Marks"
        )

        marks_label.setObjectName(
            "optionLabel"
        )

        options_row.addWidget(
            marks_label
        )

        self.marks_combo = QComboBox()

        self.marks_combo.addItems(
            self.answer_engine.MARK_OPTIONS
        )

        self.marks_combo.setCurrentText(
            "Auto"
        )

        options_row.addWidget(
            self.marks_combo
        )

        options_row.addStretch()

        # Ask button
        self.ask_button = QPushButton(
            "Ask"
        )

        self.ask_button.setObjectName(
            "askButton"
        )

        self.ask_button.setFixedWidth(
            130
        )

        self.ask_button.clicked.connect(
            self.ask_question
        )

        options_row.addWidget(
            self.ask_button
        )

        question_layout.addLayout(
            options_row
        )

        content_layout.addWidget(
            question_card
        )

        # ========================================================
        # ANSWER CARD
        # ========================================================

        answer_card = QFrame()

        answer_card.setObjectName(
            "card"
        )

        answer_layout = QVBoxLayout(
            answer_card
        )

        answer_layout.setContentsMargins(
            22,
            20,
            22,
            20
        )

        # Header
        answer_header = QHBoxLayout()

        answer_title = QLabel(
            "Answer"
        )

        answer_title.setObjectName(
            "cardTitle"
        )

        answer_header.addWidget(
            answer_title
        )

        answer_header.addStretch()

        self.answer_type_label = QLabel(
            ""
        )

        self.answer_type_label.setObjectName(
            "answerBadge"
        )

        answer_header.addWidget(
            self.answer_type_label
        )

        self.speak_button = QPushButton(
            "🔊 Speak"
        )

        self.speak_button.clicked.connect(
            self.speak_answer
        )

        answer_header.addWidget(
            self.speak_button
        )

        self.stop_button = QPushButton(
            "⏹ Stop"
        )

        self.stop_button.clicked.connect(
            self.stop_speaking
        )

        answer_header.addWidget(
            self.stop_button
        )

        answer_layout.addLayout(
            answer_header
        )

        # Answer box
        self.answer_box = QTextEdit()

        self.answer_box.setReadOnly(
            True
        )

        self.answer_box.setPlaceholderText(
            "Your exam-ready answer will appear here..."
        )

        answer_layout.addWidget(
            self.answer_box,
            1
        )

        # Sources
        source_title = QLabel(
            "Sources"
        )

        source_title.setObjectName(
            "sourceTitle"
        )

        answer_layout.addWidget(
            source_title
        )

        self.sources_label = QLabel(
            "No sources yet."
        )

        self.sources_label.setObjectName(
            "sourcesLabel"
        )

        self.sources_label.setWordWrap(
            True
        )

        answer_layout.addWidget(
            self.sources_label
        )

        content_layout.addWidget(
            answer_card,
            1
        )

        # ========================================================
        # STATUS
        # ========================================================

        footer = QLabel(
            "Local PDF search • No AI model required"
        )

        footer.setObjectName(
            "footer"
        )

        content_layout.addWidget(
            footer
        )

        main_layout.addWidget(
            content,
            1
        )

    # ============================================================
    # STYLES
    # ============================================================

    def apply_styles(self):

        self.setStyleSheet(
            """
            * {
                font-family: "Segoe UI";
            }

            QMainWindow {
                background: #f5f7f5;
            }

            #sidebar {
                background: #142016;
            }

            #logo {
                color: white;
                font-size: 27px;
                font-weight: 700;
            }

            #sidebarSubtitle {
                color: #aebcaf;
                font-size: 13px;
            }

            #sectionTitle {
                color: #8fa28f;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 1px;
            }

            #libraryList {
                background: transparent;
                border: none;
                color: #e7eee7;
                font-size: 14px;
                outline: none;
            }

            #libraryList::item {
                padding: 12px;
                border-radius: 8px;
                margin-bottom: 3px;
            }

            #libraryList::item:selected {
                background: #28532d;
            }

            QPushButton {
                background: #e7ece7;
                border: none;
                border-radius: 9px;
                padding: 10px 14px;
                font-size: 13px;
                font-weight: 600;
            }

            QPushButton:hover {
                background: #d9e2d9;
            }

            #sidebar QPushButton {
                background: #243726;
                color: white;
            }

            #sidebar QPushButton:hover {
                background: #304a32;
            }

            #pageTitle {
                font-size: 26px;
                font-weight: 700;
                color: #172117;
            }

            #statusLabel {
                color: #657365;
                font-size: 13px;
            }

            #card {
                background: white;
                border: 1px solid #e0e5e0;
                border-radius: 16px;
            }

            #cardTitle {
                font-size: 17px;
                font-weight: 700;
                color: #1c281c;
            }

            QTextEdit {
                background: #fafcf9;
                border: 1px solid #dce3dc;
                border-radius: 10px;
                padding: 12px;
                color: #1b241b;
                font-size: 14px;
                selection-background-color: #b8d6b9;
            }

            #answer_box {
                line-height: 1.5;
            }

            #voiceButton {
                font-size: 21px;
                border-radius: 27px;
                background: #eaf3ea;
            }

            #voiceButton:hover {
                background: #dcebdc;
            }

            QComboBox {
                background: #fafcf9;
                border: 1px solid #dce3dc;
                border-radius: 8px;
                padding: 9px 12px;
                min-width: 120px;
            }

            #optionLabel {
                color: #536153;
                font-size: 13px;
                font-weight: 600;
            }

            #askButton {
                background: #1f6b2a;
                color: white;
                font-size: 14px;
                padding: 11px 20px;
                border-radius: 9px;
            }

            #askButton:hover {
                background: #185722;
            }

            #answerBadge {
                background: #e8f2e8;
                color: #28632f;
                border-radius: 8px;
                padding: 7px 10px;
                font-size: 12px;
                font-weight: 600;
            }

            #sourceTitle {
                color: #3d4b3d;
                font-size: 13px;
                font-weight: 700;
            }

            #sourcesLabel {
                color: #6a756a;
                font-size: 12px;
            }

            #footer {
                color: #8a948a;
                font-size: 11px;
            }
            """
        )

    # ============================================================
    # STATUS
    # ============================================================

    def set_status(self, text):

        self.status_label.setText(
            text
        )

    # ============================================================
    # LIBRARY LOADING
    # ============================================================

    def load_library_async(self):

        self.set_status(
            "Indexing PDFs..."
        )

        self.set_controls_enabled(
            False
        )

        worker = Worker(
            self.engine.reload
        )

        worker.signals.finished.connect(
            self.library_loaded
        )

        worker.signals.error.connect(
            self.worker_error
        )

        self.thread_pool.start(
            worker
        )

    def library_loaded(self, library):

        self.library_list.clear()

        for file_info in library:

            filename = file_info[
                "filename"
            ]

            pages = file_info[
                "pages"
            ]

            item = QListWidgetItem(
                f"📄 {filename}\n"
                f"   {pages} page(s)"
            )

            item.setData(
                Qt.UserRole,
                filename
            )

            self.library_list.addItem(
                item
            )

        self.set_controls_enabled(
            True
        )

        self.set_status(
            f"{len(library)} PDF(s) loaded"
        )

    # ============================================================
    # ADD PDF
    # ============================================================

    def add_pdf(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Add PDF",
            "",
            "PDF Files (*.pdf)"
        )

        if not file_path:
            return

        self.set_status(
            "Adding PDF..."
        )

        self.set_controls_enabled(
            False
        )

        worker = Worker(
            lambda: self.engine.add_pdf(
                file_path
            )
        )

        worker.signals.finished.connect(
            self.pdf_added
        )

        worker.signals.error.connect(
            self.worker_error
        )

        self.thread_pool.start(
            worker
        )

    def pdf_added(self, filename):

        self.load_library_async()

        self.set_status(
            f"Added {filename}"
        )

    # ============================================================
    # DELETE PDF
    # ============================================================

    def delete_selected_pdf(self):

        item = self.library_list.currentItem()

        if item is None:

            QMessageBox.information(
                self,
                "Delete PDF",
                "Please select a PDF first."
            )

            return

        filename = item.data(
            Qt.UserRole
        )

        answer = QMessageBox.question(
            self,
            "Delete PDF",
            f"Delete '{filename}'?",
            QMessageBox.Yes
            | QMessageBox.No
        )

        if answer != QMessageBox.Yes:
            return

        self.set_status(
            "Deleting PDF..."
        )

        self.set_controls_enabled(
            False
        )

        worker = Worker(
            lambda: self.engine.delete_pdf(
                filename
            )
        )

        worker.signals.finished.connect(
            lambda _: self.library_loaded(
                self.engine.get_library()
            )
        )

        worker.signals.error.connect(
            self.worker_error
        )

        self.thread_pool.start(
            worker
        )

    # ============================================================
    # RELOAD
    # ============================================================

    def reload_library(self):

        self.load_library_async()

    # ============================================================
    # ASK QUESTION
    # ============================================================

    def ask_question(self):

        question = (
            self.question_input
            .toPlainText()
            .strip()
        )

        if not question:

            QMessageBox.information(
                self,
                "Question",
                "Please enter a question."
            )

            return

        if not self.engine.chunks:

            QMessageBox.information(
                self,
                "No PDFs",
                "Please add a PDF first."
            )

            return

        answer_type = (
            self.answer_type_combo
            .currentText()
        )

        marks = (
            self.marks_combo
            .currentText()
        )

        self.set_status(
            "Finding the best information..."
        )

        self.ask_button.setEnabled(
            False
        )

        self.voice_button.setEnabled(
            False
        )

        self.answer_box.clear()

        self.sources_label.setText(
            "Searching..."
        )

        worker = Worker(
            lambda: self.generate_answer(
                question,
                answer_type,
                marks
            )
        )

        worker.signals.finished.connect(
            self.answer_ready
        )

        worker.signals.error.connect(
            self.worker_error
        )

        self.thread_pool.start(
            worker
        )

    def generate_answer(
        self,
        question,
        answer_type,
        marks
    ):

        results = self.engine.search_diverse(
            question,
            top_k=10
        )

        return self.answer_engine.create_answer(
            question,
            results,
            answer_type,
            marks
        )

    # ============================================================
    # ANSWER READY
    # ============================================================

    def answer_ready(self, result):

        answer = result.get(
            "answer",
            ""
        )

        answer_type = result.get(
            "answer_type",
            "Auto"
        )

        marks = result.get(
            "marks",
            "Auto"
        )

        sources = result.get(
            "sources",
            []
        )

        self.answer_box.setPlainText(
            answer
        )

        badge = answer_type

        if marks != "Auto":
            badge += f" • {marks}"

        self.answer_type_label.setText(
            badge
        )

        # --------------------------------------------------------
        # SOURCES
        # --------------------------------------------------------

        if sources:

            source_lines = []

            for source in sources:

                source_lines.append(
                    f"📄 {source['filename']} "
                    f"— Page {source['page']}"
                )

            self.sources_label.setText(
                "\n".join(source_lines)
            )

        else:

            self.sources_label.setText(
                "No source information available."
            )

        self.set_status(
            "Answer ready"
        )

        self.ask_button.setEnabled(
            True
        )

        self.voice_button.setEnabled(
            self.voice_available
        )

    # ============================================================
    # VOICE QUESTION
    # ============================================================

    def start_voice_question(self):

        if not self.voice_available:

            QMessageBox.warning(
                self,
                "Voice unavailable",
                "Voice features are not available."
            )

            return

        self.set_status(
            "Listening..."
        )

        self.voice_button.setEnabled(
            False
        )

        self.ask_button.setEnabled(
            False
        )

        worker = Worker(
            self.voice.listen
        )

        worker.signals.finished.connect(
            self.voice_question_received
        )

        worker.signals.error.connect(
            self.worker_error
        )

        self.thread_pool.start(
            worker
        )

    def voice_question_received(
        self,
        question
    ):

        if question:

            self.question_input.setPlainText(
                question
            )

            self.set_status(
                "Voice question received"
            )

            self.ask_button.setEnabled(
                True
            )

            self.ask_question()

        else:

            self.set_status(
                "I couldn't understand you."
            )

            self.ask_button.setEnabled(
                True
            )

            self.voice_button.setEnabled(
                self.voice_available
            )

    # ============================================================
    # TEXT TO SPEECH
    # ============================================================

    def speak_answer(self):

        if not self.voice_available:
            return

        answer = (
            self.answer_box
            .toPlainText()
            .strip()
        )

        if not answer:

            QMessageBox.information(
                self,
                "Speak",
                "There is no answer to speak."
            )

            return

        self.set_status(
            "Speaking..."
        )

        self.voice_speaking = True

        self.speak_button.setEnabled(
            False
        )

        worker = Worker(
            lambda: self.voice.speak(
                self.clean_for_speech(
                    answer
                )
            )
        )

        worker.signals.finished.connect(
            self.speech_finished
        )

        worker.signals.error.connect(
            self.worker_error
        )

        self.thread_pool.start(
            worker
        )

    def clean_for_speech(self, text):

        # Remove visual formatting.
        text = re.sub(
            r"[*_#]",
            "",
            text
        )

        text = text.replace(
            "•",
            ""
        )

        text = re.sub(
            r"\n+",
            ". ",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    def speech_finished(self, _):

        self.voice_speaking = False

        self.speak_button.setEnabled(
            self.voice_available
        )

        self.set_status(
            "Ready"
        )

    def stop_speaking(self):

        if self.voice_available:

            try:
                self.voice.stop()

            except Exception:
                pass

            self.voice_speaking = False

            self.speak_button.setEnabled(
                True
            )

            self.set_status(
                "Speech stopped"
            )

    # ============================================================
    # ENABLE / DISABLE CONTROLS
    # ============================================================

    def set_controls_enabled(self, enabled):

        self.add_button.setEnabled(
            enabled
        )

        self.delete_button.setEnabled(
            enabled
        )

        self.reload_button.setEnabled(
            enabled
        )

        self.ask_button.setEnabled(
            enabled
        )

        self.voice_button.setEnabled(
            enabled
            and self.voice_available
        )

    # ============================================================
    # ERROR
    # ============================================================

    def worker_error(self, message):

        self.set_controls_enabled(
            True
        )

        self.set_status(
            "Something went wrong"
        )

        QMessageBox.critical(
            self,
            "Error",
            message
        )

    # ============================================================
    # CLOSE
    # ============================================================

    def closeEvent(self, event):

        if self.voice_available:

            try:
                self.voice.cleanup()

            except Exception:
                pass

        event.accept()


# ================================================================
# MAIN
# ================================================================

def main():

    app = QApplication(
        sys.argv
    )

    app.setApplicationName(
        "Study AI"
    )

    window = StudyAI()

    window.show()

    sys.exit(
        app.exec()
    )


if __name__ == "__main__":
    main()