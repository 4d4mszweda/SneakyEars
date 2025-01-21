import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from transformers import BartForConditionalGeneration, BartTokenizer
import whisper
import os
from langdetect import detect
import matplotlib.pyplot as plt
from collections import Counter
import threading
from textblob import TextBlob
import pyaudio
import numpy as np
from faster_whisper import WhisperModel
import threading
import time


# Load Whisper and Faster Whisper models
model = whisper.load_model("medium")
# faster_model = WhisperModel("medium", device="cpu", compute_type="int8")
faster_model = WhisperModel("medium", device="cuda", compute_type="float16")  # Użyj "cpu" dla CPU

bart_model_name = "facebook/bart-large-cnn"
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)

# Audio configuration
CHUNK = 16000  # Liczba próbek w jednym fragmencie (1 sekunda dla 16 kHz)
FORMAT = pyaudio.paInt16  # Format danych audio
CHANNELS = 1  # Mono
RATE = 16000  # Częstotliwość próbkowania

live_transcription_running = False
transcription_thread = None
indicator_running = False

def start_live_transcription():
    global live_transcription_running, transcription_thread, indicator_running
    live_transcription_running = True
    indicator_running = True
    transcription_thread = threading.Thread(target=live_transcription, daemon=True)
    transcription_thread.start()
    threading.Thread(target=animate_indicator, daemon=True).start()

def stop_live_transcription():
    global live_transcription_running, indicator_running
    live_transcription_running = False
    indicator_running = False
    listening_label.config(text="")  # Clear the indicator

def animate_indicator():
    states = ["Listening.", "Listening..", "Listening..."]
    idx = 0
    start_time = time.time()
    while indicator_running:
        elapsed_time = int(time.time() - start_time)
        minutes, seconds = divmod(elapsed_time, 60)
        time_str = f"{minutes:02}:{seconds:02}"
        listening_label.config(text=f"{states[idx]} {time_str}")
        idx = (idx + 1) % len(states)
        listening_label.update()
        app.after(500) 

def live_transcription():
    global live_transcription_running
    audio = pyaudio.PyAudio()

    # Configure audio stream
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    transcription_text.delete(1.0, tk.END)  
    buffer = []

    try:
        while live_transcription_running:
            # Read audio data from the microphone
            data = stream.read(CHUNK, exception_on_overflow=False)
            buffer.append(np.frombuffer(data, np.int16).astype(np.float32) / 32768.0)  # Normalize audio

            if len(buffer) >= 30:
                audio_data = np.concatenate(buffer, axis=0)  # Dodaj kontekst

                buffer = []

                segments, _ = faster_model.transcribe(audio_data, beam_size=5, word_timestamps=False)
                for segment in segments:
                    transcription_text.insert(tk.END, segment.text + " ")
                    transcription_text.see(tk.END)
    except Exception as e:
        print(f"Error during live transcription: {e}")
    finally:
        if buffer:
            audio_data = np.concatenate(buffer, axis=0)
            segments, _ = faster_model.transcribe(audio_data, beam_size=5, word_timestamps=False)
            for segment in segments:
                transcription_text.insert(tk.END, segment.text + " ")
                transcription_text.see(tk.END)
        stream.stop_stream()
        stream.close()
        audio.terminate()

def transcribe_audio(file_path):
    result = model.transcribe(file_path)
    return result["text"]

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    sentiment_label.config(text=f"Sentiment: {sentiment.polarity:.2f} (subjectivity: {sentiment.subjectivity:.2f})")

def calculate_statistics(text):
    words = text.split()
    sentences = text.split('.')
    paragraphs = text.split('\n\n')
    unique_words = set(words)
    word_count = len(words)
    sentence_count = len(sentences)
    paragraph_count = len(paragraphs)
    avg_sentence_length = word_count / sentence_count if sentence_count else 0
    stats_label.config(text=f"Words: {word_count}, Sentences: {sentence_count}, Paragraphs: {paragraph_count}, Avg. Sentence Length: {avg_sentence_length:.2f}, Unique Words: {len(unique_words)}")

def translate_text(file_path):
    result = model.transcribe(file_path, task="translate")
    return result["text"]

def process_file(file_path):
    try:
        # Show loading indicator
        loading_label.config(text="Processing...")

        # Perform the operations
        transcription = transcribe_audio(file_path)
        translation = translate_text(file_path)

        # Generate summary
        inputs = bart_tokenizer.encode("summarize: " + translation, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = bart_model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Update UI with results
        transcription_text.delete(1.0, tk.END)
        transcription_text.insert(tk.END, transcription)

        translation_text.delete(1.0, tk.END)
        translation_text.insert(tk.END, translation)

        summary_text.delete(1.0, tk.END)
        summary_text.insert(tk.END, summary)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
    finally:
        # Hide loading indicator
        loading_label.config(text="")

def upload_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3;*.wav;*.flac;*.ogg")])
    if file_path:
        # Run processing in a separate thread
        threading.Thread(target=process_file, args=(file_path,), daemon=True).start()

def save_results():
    file_path = filedialog.asksaveasfilename(defaultextension=".txt", 
                                             filetypes=[("Text Files", "*.txt")])
    if file_path:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("Transcription:\n")
                f.write(transcription_text.get(1.0, tk.END).strip() + "\n\n")
                f.write("Translation:\n")
                f.write(translation_text.get(1.0, tk.END).strip() + "\n\n")
                f.write("Summary:\n")
                f.write(summary_text.get(1.0, tk.END).strip() + "\n")
            messagebox.showinfo("Success", "Results saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

def highlight_keywords(text_widget):
    text_widget.tag_configure("highlight", background="yellow")
    content = text_widget.get(1.0, tk.END).lower()
    words = content.split()
    keywords = [word for word in set(words) if words.count(word) > 2]  # Simple algorithm
    for keyword in keywords:
        start_idx = "1.0"
        while True:
            start_idx = text_widget.search(keyword, start_idx, tk.END)
            if not start_idx:
                break
            end_idx = f"{start_idx}+{len(keyword)}c"
            text_widget.tag_add("highlight", start_idx, end_idx)
            start_idx = end_idx

def detect_language(text):
    try:
        lang = detect(text)
        messagebox.showinfo("Detected Language", f"Detected language: {lang}")
    except Exception as e:
        messagebox.showerror("Error", f"Could not detect language: {e}")

def plot_word_frequency():
    transcription = transcription_text.get(1.0, tk.END).strip()
    words = transcription.split()
    word_counts = Counter(words)
    most_common = word_counts.most_common(10)

    labels, values = zip(*most_common)
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values, color="skyblue")
    plt.title("Top 10 Words in Transcription")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def change_language(lang):
    if lang == "pl":
        upload_button.config(text="Prześlij plik audio")
        save_button.config(text="Zapisz wyniki")
        transcription_label.config(text="Transkrypcja:")
        translation_label.config(text="Tłumaczenie:")
        summary_label.config(text="Podsumowanie:")
        highlight_button.config(text="Podświetl słowa kluczowe")
        detect_lang_button.config(text="Wykryj język")
        plot_button.config(text="Pokaż wykres")
    else:
        upload_button.config(text="Upload Audio File")
        save_button.config(text="Save Results")
        transcription_label.config(text="Transcription:")
        translation_label.config(text="Translation:")
        summary_label.config(text="Summary:")
        highlight_button.config(text="Highlight Keywords")
        detect_lang_button.config(text="Detect Language")
        plot_button.config(text="Show Plot")

# Initialize GUI
app = tk.Tk()
app.title("SneakyEars")
app.geometry("800x1100")

# Style
style = ttk.Style()
style.configure("TButton", font=("Arial", 12))
style.configure("TLabel", font=("Arial", 12))

# Language Menu
language_options = {"English": "en", "Polish": "pl"}
lang_menu = ttk.Combobox(app, values=list(language_options.keys()))
lang_menu.set("English")
lang_menu.pack(pady=10)
lang_menu.bind("<<ComboboxSelected>>", lambda e: change_language(language_options[lang_menu.get()]))

# Upload Button
upload_button = ttk.Button(app, text="Upload Audio File", command=upload_file)
upload_button.pack(pady=10)

start_button = ttk.Button(app, text="Start Live Transcription", command=start_live_transcription)
start_button.pack(pady=10)

stop_button = ttk.Button(app, text="Stop Live Transcription", command=stop_live_transcription)
stop_button.pack(pady=10)

# Save Button
save_button = ttk.Button(app, text="Save Results", command=save_results)
save_button.pack(pady=10)

listening_label = ttk.Label(app, text="", font=("Arial", 14), foreground="green")
listening_label.pack(pady=10)

# Loading Indicator
loading_label = ttk.Label(app, text="", font=("Arial", 12), foreground="red")
loading_label.pack(pady=10)

# Transcription Section
transcription_label = ttk.Label(app, text="Transcription:")
transcription_label.pack(anchor="w", padx=20)
transcription_text = tk.Text(app, wrap=tk.WORD, height=8, width=70)
transcription_text.pack(padx=20, pady=10)

# Translation Section
translation_label = ttk.Label(app, text="Translation:")
translation_label.pack(anchor="w", padx=20)
translation_text = tk.Text(app, wrap=tk.WORD, height=8, width=70)
translation_text.pack(padx=20, pady=10)

# Summary Section
summary_label = ttk.Label(app, text="Summary:")
summary_label.pack(anchor="w", padx=20)
summary_text = tk.Text(app, wrap=tk.WORD, height=8, width=70)
summary_text.pack(padx=20, pady=10)

# Highlight Button
highlight_button = ttk.Button(app, text="Highlight Keywords", 
                              command=lambda: highlight_keywords(translation_text))
highlight_button.pack(pady=10)

# Detect Language Button
detect_lang_button = ttk.Button(app, text="Detect Language", 
                                command=lambda: detect_language(transcription_text.get(1.0, tk.END)))
detect_lang_button.pack(pady=10)

# Plot Button
plot_button = ttk.Button(app, text="Show Plot", command=plot_word_frequency)
plot_button.pack(pady=10)

sentiment_button = ttk.Button(app, text="Analyze Sentiment", 
                              command=lambda: analyze_sentiment(transcription_text.get(1.0, tk.END)))
sentiment_button.pack(pady=10)

stats_button = ttk.Button(app, text="Calculate Statistics", 
                          command=lambda: calculate_statistics(transcription_text.get(1.0, tk.END)))
stats_button.pack(pady=10)

# Sentiment Label
sentiment_label = ttk.Label(app, text="Sentiment:")
sentiment_label.pack(anchor="w", padx=20)

# Statistics Label
stats_label = ttk.Label(app, text="Statistics:")
stats_label.pack(anchor="w", padx=20)


app.mainloop()
