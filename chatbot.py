import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from gensim.models import Phrases
from gensim.models.phrases import Phraser

# === 1. Load stopwords dari file CSV tanpa header ===
stopword_df = pd.read_csv('stopwordbahasa.csv', header=None)
stopwords = set(stopword_df[0].tolist())

# === 2. Inisialisasi stemmer ===
stemmer = StemmerFactory().create_stemmer()

# === 3. Dataset contoh untuk phrase detection ===
example_texts = [
    "menanam cabai di lahan sempit",
    "mengatasi hama tanaman secara alami",
    "penyiraman tanaman setiap hari",
    "penggunaan pupuk organik untuk sayur"
]

# === 4. Fungsi tokenize sederhana ===
def tokenize(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    return tokens

# === 5. Buat phrase model (bigram + trigram) dari dataset contoh ===
base_tokens = [tokenize(txt) for txt in example_texts]
bigram = Phrases(base_tokens, min_count=1, threshold=1)
trigram = Phrases(bigram[base_tokens], threshold=1)
bigram_mod = Phraser(bigram)
trigram_mod = Phraser(trigram)

# === 6. Manual phrase merge setelah stemming (contoh: tanam + cabai => tanam_cabai) ===
def manual_phrase_merge(tokens):
    merged_tokens = []
    i = 0
    while i < len(tokens):
        # contoh manual merge sederhana
        if i + 1 < len(tokens):
            if tokens[i] == "tanam" and tokens[i+1] == "cabai":
                merged_tokens.append("tanam_cabai")
                i += 2
                continue
            if tokens[i] == "hama" and tokens[i+1] == "tanam":
                merged_tokens.append("hama_tanaman")
                i += 2
                continue
            if tokens[i] == "pupuk" and tokens[i+1] == "organik":
                merged_tokens.append("pupuk_organik")
                i += 2
                continue
            if tokens[i] == "penyiram" and tokens[i+1] == "tanam":
                merged_tokens.append("penyiram_tanaman")
                i += 2
                continue
        # jika tidak merge, masukkan token biasa
        merged_tokens.append(tokens[i])
        i += 1
    return merged_tokens

# === 7. Fungsi proses input user: phrase detection dulu, baru stemming, manual merge, stopword removal ===
def detect_phrases_and_preprocess(text, bigram_mod, trigram_mod, stopwords, stemmer):
    tokens = tokenize(text)
    tokens_with_phrases = trigram_mod[bigram_mod[tokens]]

    # Stemming dulu sebelum manual merge
    tokens_stemmed = [stemmer.stem(t) for t in tokens_with_phrases]

    # Manual merge setelah stemming
    tokens_merged = manual_phrase_merge(tokens_stemmed)

    # Stopword removal setelah merge
    tokens_cleaned = [t for t in tokens_merged if t not in stopwords]

    return tokens_cleaned

# === 8. Knowledge base chatbot (key adalah phrase/token hasil preprocessing) ===
knowledge_base = {
    "tanam_cabai": "Untuk menanam cabai, pastikan mendapat sinar matahari cukup dan drainase baik.",
    "hama_tanaman": "Hama tanaman bisa dikendalikan dengan insektisida alami seperti bawang putih.",
    "pupuk_organik": "Pupuk organik seperti kompos membantu menyuburkan tanah secara alami.",
    "penyiram_tanaman": "Sebaiknya tanaman disiram pagi atau sore hari agar tidak cepat menguap.",
    "tanam": "Tanaman butuh sinar matahari cukup dan air yang memadai.",
    "cabai": "Cabai sebaiknya ditanam di tempat yang mendapat sinar matahari cukup.",
    "hama": "Gunakan insektisida nabati atau perangkap hama alami.",
    "pupuk": "Gunakan pupuk organik seperti kompos atau pupuk kandang.",
    "alami": "Metode alami lebih ramah lingkungan dan aman untuk tanaman.",
    "penyiram": "Penyiraman harus dilakukan secara rutin terutama pada pagi dan sore hari."
}

# === 9. Fungsi chatbot interaktif ===
def chatbot():
    print("🌱 Chatbot Perkebunan Siap! (Ketik 'exit' untuk keluar)")
    while True:
        user_input = input("👩‍🌾 Kamu: ")
        if user_input.lower() == 'exit':
            break

        tokens = detect_phrases_and_preprocess(user_input, bigram_mod, trigram_mod, stopwords, stemmer)
        print("🔍 Token + Phrase:", tokens)

        responses = []
        for phrase in tokens:
            if phrase in knowledge_base:
                responses.append(knowledge_base[phrase])

        if responses:
            for resp in responses:
                print("🤖 Bot:", resp)
        else:
            print("🤖 Bot: Maaf, saya belum tahu jawabannya. Coba tanyakan hal lain.")

if __name__ == "__main__":
    chatbot()
