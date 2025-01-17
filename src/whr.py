# import whisper

# # Załaduj model (np. 'base', 'small', 'medium', 'large')
# model = whisper.load_model("base")

# # Ścieżka do pliku audio
# audio_path = "example_audio.mp3"

# # Transkrypcja pliku audio
# result = model.transcribe(audio_path)
# # result = model.transcribe(audio_path, language="pl")
# # result = model.transcribe(audio_path, task="translate")
# # result = model.transcribe(audio_path, no_speech_threshold=0.6, logprob_threshold=-1.0)


# # Wyświetl transkrypcję
# print(result["text"])


import whisper

def load_model(model_name="base"):
    return whisper.load_model(model_name)

def transcribe_audio(model, audio_path):
    result = model.transcribe(audio_path)
    return result["text"]