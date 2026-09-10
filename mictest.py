from voice_engine import VoiceEngine


voice = VoiceEngine()


print("================================")
print("      STUDY AI VOICE TEST")
print("================================")

print()
print("Testing microphone...")
print("Say something after 'Listening...'")
print()


try:

    text = voice.listen()

    if text:

        print()
        print("You said:")
        print(text)

        print()
        print("Testing AI voice...")

        voice.speak(
            f"You said: {text}"
        )

        print()
        print("Voice test successful.")

    else:

        print()
        print("I couldn't understand you.")


except Exception as error:

    print()
    print("VOICE ERROR:")
    print(error)


finally:

    voice.cleanup()