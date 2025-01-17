import tkinter as tk
from tkinter import filedialog, messagebox
import whisper
import os

# Load Whisper model
model = whisper.load_model("medium")

def transcribe_audio(file_path):
    result = model.transcribe(file_path)
    return result["text"]

def translate_text(file_path):
    result = model.transcribe(file_path, task="translate")
    return result["text"]

def upload_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3;*.wav")])
    if file_path:
        try:
            transcription = transcribe_audio(file_path)
            translation = translate_text(file_path)
            result_text.delete(1.0, tk.END)  # Clear previous text
            result_text.insert(tk.END, f"Transcription:\n{transcription}\n\nTranslation:\n{translation}")  # Display new transcription and translation
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

# Initialize GUI
app = tk.Tk()
app.title("SneakyEars")

upload_button = tk.Button(app, text="Upload Audio File", command=upload_file)
upload_button.pack(pady=20)

result_text = tk.Text(app, wrap=tk.WORD, height=15, width=50)
result_text.pack(pady=20)

app.mainloop()