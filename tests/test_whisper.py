import whisper

try:
    print("Loading model...")
    model = whisper.load_model("small")
    print("Transcribing test.wav...")
    result = model.transcribe("test.wav")
    print("Result:", result["text"][:100])
except Exception as e:
    import traceback
    traceback.print_exc()
