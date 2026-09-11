import asyncio
import os
import tempfile

import edge_tts
import pygame
import speech_recognition as sr


VOICE = "en-IN-NeerjaNeural"
VOICE_RATE = "-5%"


class VoiceEngine:

    def __init__(self):

        self.recognizer = sr.Recognizer()

        pygame.mixer.init()

        self.audio_file = os.path.join(
            tempfile.gettempdir(),
            "study_ai_voice.mp3"
        )


    
    # SPEECH TO TEXT
    

    def listen(self):

        with sr.Microphone() as source:

            print("Listening...")

            self.recognizer.adjust_for_ambient_noise(
                source,
                duration=0.5
            )

            audio = self.recognizer.listen(
                source,
                timeout=5,
                phrase_time_limit=10
            )

        try:

            text = self.recognizer.recognize_google(
                audio
            )

            return text

        except sr.UnknownValueError:

            return None

        except sr.RequestError as error:

            raise RuntimeError(
                f"Speech recognition service error: {error}"
            )


    
    # TEXT TO SPEECH
    

    async def _generate_audio(self, text):

        communicate = edge_tts.Communicate(
            text,
            VOICE,
            rate=VOICE_RATE
        )

        await communicate.save(
            self.audio_file
        )


    def speak(self, text):

        if not text:
            return

        self.stop()

        asyncio.run(
            self._generate_audio(text)
        )

        pygame.mixer.music.load(
            self.audio_file
        )

        pygame.mixer.music.play()


        while pygame.mixer.music.get_busy():

            pygame.time.Clock().tick(10)


    
    # STOP
    

    def stop(self):

        if pygame.mixer.music.get_busy():

            pygame.mixer.music.stop()


    
    # CLEANUP
    

    def cleanup(self):

        self.stop()

        try:

            pygame.mixer.quit()

        except Exception:

            pass