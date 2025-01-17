# SneakyEars

SneakyEars to zaawansowana aplikacja GUI, która umożliwia przetwarzanie plików audio na tekst, tłumaczenie, podsumowywanie oraz analizowanie tekstu. Projekt wykorzystuje modele Whisper, BART oraz kilka dodatkowych bibliotek do analizy i wizualizacji danych tekstowych.

---

## Funkcjonalności

### 1. **Transkrypcja Audio**

- Konwertuje pliki audio (MP3, WAV, FLAC, OGG) na tekst za pomocą modelu Whisper.

### 2. **Tłumaczenie**

- Tłumaczy transkrypcję audio na język angielski.

### 3. **Podsumowanie Tekstu**

- Generuje podsumowanie tłumaczenia za pomocą modelu BART.

### 4. **Analiza Sentimentów**

- Analizuje nastroje tekstu (pozytywne/negatywne) oraz poziom subiektywności.

### 5. **Statystyki Tekstu**

- Oblicza podstawowe statystyki tekstu, takie jak liczba słów, zdań, akapitów oraz unikalnych słów.

### 6. **Wizualizacja Częstości Słów**

- Tworzy wykres słupkowy 10 najczęściej występujących słów w transkrypcji.

### 7. **Podświetlanie Słów Kluczowych**

- Podświetla słowa kluczowe, które występują więcej niż dwa razy w tekście.

### 8. **Wykrywanie Języka**

- Wykrywa język wprowadzonego tekstu.

### 9. **Zmiana Języka Interfejsu**

- Obsługa w języku angielskim i polskim.

### 10. **Zapis Wyników**

- Zapisuje wyniki transkrypcji, tłumaczenia i podsumowania do pliku tekstowego.

---

## Wymagania

- Python 3.8 lub nowszy
- Zainstalowane biblioteki:
  - `tkinter`
  - `transformers`
  - `whisper`
  - `langdetect`
  - `textblob`
  - `matplotlib`
  - `collections`
  - `threading`

---

## Instalacja

1. Sklonuj repozytorium:

   ```bash
   git clone https://github.com/yourusername/sneakyears.git
   cd sneakyears
   ```

2. Zainstaluj wymagane biblioteki:

   ```bash
   pip install -r requirements.txt
   ```

3. Uruchom aplikację:
   ```bash
   python src/main.py
   ```

---

## Obsługa

1. **Przesyłanie pliku audio**

   - Kliknij "Upload Audio File" (lub "Prześlij plik audio" w języku polskim).
   - Wybierz plik audio do przetworzenia.

2. **Wyświetlanie wyników**

   - Transkrypcja, tłumaczenie i podsumowanie pojawią się w odpowiednich sekcjach.

3. **Dodatkowe operacje**

   - Użyj przycisków do analizy sentymentów, statystyk tekstu, wykrywania języka lub wizualizacji częstości słów.

4. **Zapis wyników**
   - Kliknij "Save Results" (lub "Zapisz wyniki"), aby zapisać transkrypcję, tłumaczenie i podsumowanie do pliku tekstowego.

---

## Struktura Projektu

- `src/main.py`: Główny plik aplikacji.
- `requirements.txt`: Lista wymaganych bibliotek.

---

## Przykład Użycia

1. Prześlij plik audio w formacie MP3, WAV, FLAC lub OGG.
2. Odczytaj transkrypcję, tłumaczenie i podsumowanie.
3. Użyj funkcji analizy tekstu (np. analiza sentymentów lub statystyki).
4. Zapisz wyniki do pliku tekstowego.

---

# Analiza
