import whisper
import time
from jiwer import wer
from langdetect import detect

# Lista modeli Whisper
models = ["tiny", "base", "small", "medium", "large"]

# Funkcja do testowania modelu Whisper
def test_model(model_name, audio_files, ground_truth_texts):
    model = whisper.load_model(model_name)
    total_wer = 0
    total_language_detection = 0
    total_translation_success = 0
    total_time = 0

    for i, audio_file in enumerate(audio_files):
        ground_truth = ground_truth_texts[i]

        # Start pomiaru czasu
        start_time = time.time()

        # Transkrypcja
        result = model.transcribe(audio_file)
        transcription = result["text"]
        detected_language = result["language"]

        # Zatrzymanie pomiaru czasu
        end_time = time.time()
        total_time += (end_time - start_time)

        # Obliczenie WER (Word Error Rate)
        total_wer += wer(ground_truth, transcription)

        # Wykrywanie języka
        if detected_language == detect(ground_truth):
            total_language_detection += 1

        # Tłumaczenie (zakładamy poprawność tłumaczenia, jeśli jest to angielski)
        if "translate" in result and detect(result["translate"]) == "en":
            total_translation_success += 1

    # Obliczenie średnich wyników
    num_files = len(audio_files)
    avg_wer = total_wer / num_files * 100  # WER jako procent
    avg_time = total_time / num_files
    lang_detection_rate = (total_language_detection / num_files) * 100
    translation_success_rate = (total_translation_success / num_files) * 100

    return {
        "model": model_name,
        "avg_wer": avg_wer,
        "avg_time": avg_time,
        "lang_detection_rate": lang_detection_rate,
        "translation_success_rate": translation_success_rate
    }

# Przygotowanie danych testowych
# Lista plików audio i odpowiadające im prawdziwe transkrypcje (ground truth)
audio_files = ["sample1.wav", "sample2.wav", "sample3.wav", "..."]  # Dodaj 100 plików
ground_truth_texts = ["ground truth text 1", "ground truth text 2", "ground truth text 3", "..."]

# Przeprowadzenie testów dla każdego modelu
results = []
for model_name in models:
    print(f"Testing model: {model_name}...")
    result = test_model(model_name, audio_files, ground_truth_texts)
    results.append(result)
    print(f"Completed model: {model_name}")

# Wyświetlenie wyników w formacie tabelarycznym
print("\nAnalysis Results:\n")
print(f"{'Model':<10}{'WER (%)':<10}{'Time (s)':<10}{'Lang Detect (%)':<15}{'Translation (%)':<15}")
for result in results:
    print(f"{result['model']:<10}{result['avg_wer']:<10.2f}{result['avg_time']:<10.2f}{result['lang_detection_rate']:<15.2f}{result['translation_success_rate']:<15.2f}")

# Zapis wyników do pliku
with open("whisper_analysis_results.txt", "w") as f:
    f.write("Model\tWER (%)\tTime (s)\tLang Detect (%)\tTranslation (%)\n")
    for result in results:
        f.write(f"{result['model']}\t{result['avg_wer']:.2f}\t{result['avg_time']:.2f}\t{result['lang_detection_rate']:.2f}\t{result['translation_success_rate']:.2f}\n")
