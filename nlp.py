import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from gensim.models import Phrases
from gensim.models.phrases import Phraser

# === 1. Inisialisasi ===
# Load stopword tanpa header, kolom langsung diasumsikan 0
stopword_df = pd.read_csv('stopwordbahasa.csv', header=None)
stopwords = set(stopword_df[0].tolist())

# Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# === 2. Fungsi Cleaning + Tokenization + Stopword Removal + Stemming ===
def preprocess_basic(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # cleaning
    text = text.lower()  # lowercasing
    tokens = text.split()  # tokenization
    tokens = [t for t in tokens if t not in stopwords]  # stopword removal
    tokens = [stemmer.stem(t) for t in tokens]  # stemming
    return tokens

# === 3. Fungsi untuk phrase detection ===
def detect_phrases(corpus_tokens):
    bigram = Phrases(corpus_tokens, min_count=1, threshold=1)
    trigram = Phrases(bigram[corpus_tokens], threshold=1)

    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)

    return [trigram_mod[bigram_mod[doc]] for doc in corpus_tokens]

# === 4. Main program interaktif ===
def main():
    print("Masukkan kalimat yang akan diproses (ketik 'exit' untuk keluar):")
    corpus_tokens = []

    while True:
        text = input("> ")
        if text.lower() == 'exit':
            break
        tokens = preprocess_basic(text)
        corpus_tokens.append(tokens)
        print("Hasil preprocessing:", tokens)

    if corpus_tokens:
        print("\n=== Hasil Phrase Detection ===")
        final_corpus = detect_phrases(corpus_tokens)
        for i, doc in enumerate(final_corpus):
            print(f"Kalimat {i+1}: {' '.join(doc)}")
    else:
        print("Tidak ada kalimat yang diproses.")

if __name__ == "__main__":
    main()
