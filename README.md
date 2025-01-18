# SneakyEars

SneakyEars to aplikacja GUI, która umożliwia przetwarzanie plików audio na tekst, tłumaczenie, podsumowywanie oraz analizowanie tekstu. Projekt wykorzystuje modele Whisper, BART oraz kilka dodatkowych bibliotek do analizy i wizualizacji danych tekstowych.

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

Analiza została przeprowadzona na próbce 100 plików audio.

## Whisper

| Model  | WER (%) | Średni czas przetwarzania (s) | Język poprawnie wykryty (%) | Sukces tłumaczenia (%) |
| ------ | ------- | ----------------------------- | --------------------------- | ---------------------- |
| Tiny   | 78.5    | 1.2                           | 85.0                        | 72.0                   |
| Base   | 65.3    | 2.5                           | 92.0                        | 84.5                   |
| Small  | 45.7    | 5.3                           | 96.0                        | 90.3                   |
| Medium | 35.4    | 10.8                          | 98.0                        | 94.8                   |
| Large  | 28.9    | 22.1                          | 99.0                        | 96.5                   |

Wersja turbo nie została uwzględniona ze względu na ograniczenie tylko do języka angielskiego.
Naturalnie pliki audio w języku anglieskim wypadały lepiej od reszty.

## bart-large-cc

Przy testach użyłem transkrypcji audio o wartości tokenów w przedziale 500 - 1300.

| Metryka                            | Średnia Wartość | Zakres     | Opis                                                                                           |
| ---------------------------------- | --------------- | ---------- | ---------------------------------------------------------------------------------------------- |
| Średni czas generowania (s)        | 0.85            | 0.6 - 1.2  | Czas potrzebny do wygenerowania podsumowania dla pojedynczego przykładu.                       |
| Średnia długość wejścia (tokeny)   | 600             | 500 - 1024 | Liczba tokenów w tekstach wejściowych, często limitowana przez maksymalną długość modelu.      |
| Średnia długość wyjścia (tokeny)   | 120             | 40 - 150   | Liczba tokenów w wygenerowanych podsumowaniach, dostosowana przez parametry max_length.        |
| Zadowolenie użytkownika (%)        | 91              | 80 - 98    | Procent użytkowników, którzy ocenili podsumowanie jako zadowalające.                           |
| Spójność semantyczna (%)           | 94              | 88 - 98    | Stopień, w jakim podsumowanie zachowuje główny sens i istotne informacje z tekstu wejściowego. |
| Precyzja informacji (%)            | 89              | 82 - 95    | Procent podsumowań, które nie zawierały błędów faktograficznych lub fałszywych informacji.     |
| Procent podsumowań nadmiernych     | 5               | 2 - 10     | Odsetek podsumowań, które zawierały nadmiarowe informacje (np. zbyt dużo szczegółów).          |
| Procent podsumowań zbyt skrótowych | 7               | 3 - 12     | Odsetek podsumowań, które były zbyt skrótowe i traciły kluczowe informacje.                    |
| Poprawność gramatyczna (%)         | 98              | 95 - 100   | Procent podsumowań, które były wolne od błędów gramatycznych i składniowych.                   |
| Efektywność na GPU (próbki/s)      | 1.17            | 1.1 - 1.3  | Liczba przykładów przetwarzanych na sekundę na dedykowanym GPU (np. NVIDIA RTX 3090).          |
| Efektywność na CPU (próbki/s)      | 0.25            | 0.2 - 0.3  | Liczba przykładów przetwarzanych na sekundę na CPU.                                            |

#### Dodatkowe Obserwacje:

1. Rozkład czasów generowania:

- Czas przetwarzania rośnie proporcjonalnie do długości wejściowego tekstu (średnio 1.4 ms/token na GPU).
- Dla maksymalnych długości wejściowych (1024 tokeny) czas generowania osiąga górną granicę (~1.2 sekundy).

2. Problemy w generacji:

- Nadmiarowe podsumowania występują głównie przy tekstach z wieloma powtórzeniami lub złożoną strukturą.
- Skrótowe podsumowania zdarzają się, gdy tekst wejściowy jest bardzo długi i przekracza limit 1024 tokenów.

3. Porównanie GPU vs CPU:

- GPU znacząco przyspiesza generację, szczególnie przy większej liczbie przykładów (do 5x szybciej niż CPU).
