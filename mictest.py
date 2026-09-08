import speech_recognition as sr

recognizer = sr.Recognizer()

print("🎤 Microphone test")
print("Speak something...")

with sr.Microphone() as source:

    recognizer.adjust_for_ambient_noise(
        source,
        duration=1
    )

    print("🎤 Listening...")

    audio = recognizer.listen(
        source,
        timeout=5,
        phrase_time_limit=10
    )

try:

    print("🧠 Converting speech to text...")

    text = recognizer.recognize_google(
        audio
    )

    print("\nYou said:")
    print(text)

except sr.UnknownValueError:

    print(
        "\n❌ I couldn't understand your speech."
    )

except sr.RequestError as error:

    print(
        f"\n❌ Speech recognition service error: {error}"
    )

except Exception as error:

    print(
        f"\n❌ Error: {error}"
    )