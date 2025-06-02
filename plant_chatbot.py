import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from gensim.models import Phrases
from gensim.models.phrases import Phraser

# Load stopwords
stopword_df = pd.read_csv('stopwordbahasa.csv', header=None)
stopwords = set(stopword_df[0].tolist())

stemmer = StemmerFactory().create_stemmer()

example_texts = [
    "menanam cabai di lahan sempit",
    "mengatasi hama tanaman secara alami",
    "penyiraman tanaman setiap hari",
    "penggunaan pupuk organik untuk sayur"
]

def tokenize(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    return tokens

base_tokens = [tokenize(txt) for txt in example_texts]
bigram = Phrases(base_tokens, min_count=1, threshold=1)
trigram = Phrases(bigram[base_tokens], threshold=1)
bigram_mod = Phraser(bigram)
trigram_mod = Phraser(trigram)

def manual_phrase_merge(tokens):
    merged_tokens = []
    i = 0
    while i < len(tokens):
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
        merged_tokens.append(tokens[i])
        i += 1
    return merged_tokens

def detect_phrases_and_preprocess(text):
    tokens = tokenize(text)
    tokens_with_phrases = trigram_mod[bigram_mod[tokens]]
    tokens_stemmed = [stemmer.stem(t) for t in tokens_with_phrases]
    tokens_merged = manual_phrase_merge(tokens_stemmed)
    tokens_cleaned = [t for t in tokens_merged if t not in stopwords]
    return tokens_cleaned

# Rule-based inference function
def rule_based_inference(facts):
    conclusions = {}

    # Rule 1
    if facts.get('pertumbuhan') == 'lambat' and facts.get('ukuran') == 'kerdil':
        conclusions['penyebab'] = 'nutrisi_rendah'

    # Rule 2
    if conclusions.get('penyebab') == 'nutrisi_rendah' and facts.get('jenis_pupuk') == 'kandang':
        conclusions['frekuensi_pemupukan'] = '3_bulan'

    # Rule 3
    if facts.get('pH_tanah') == 'asam':
        conclusions['tambahkan'] = 'kapur_dolomit'

    return conclusions

# Fungsi ekstraksi fakta sederhana dari token hasil preprocessing
def extract_facts(tokens):
    facts = {}

    # Cek kata kunci untuk pertumbuhan
    if 'lambat' in tokens:
        facts['pertumbuhan'] = 'lambat'
    elif 'cepat' in tokens:
        facts['pertumbuhan'] = 'cepat'

    # Cek kata kunci untuk ukuran
    if 'kerdil' in tokens:
        facts['ukuran'] = 'kerdil'
    elif 'besar' in tokens or 'subur' in tokens:
        facts['ukuran'] = 'besar'

    # Cek jenis pupuk
    if 'kandang' in tokens:
        facts['jenis_pupuk'] = 'kandang'
    elif 'organik' in tokens:
        facts['jenis_pupuk'] = 'organik'

    # Cek pH tanah
    if 'asam' in tokens:
        facts['pH_tanah'] = 'asam'
    elif 'basa' in tokens:
        facts['pH_tanah'] = 'basa'

    return facts

# Chatbot interaktif
def chatbot():
    print("🌱 Chatbot Perkebunan Siap! (Ketik 'exit' untuk keluar)")
    while True:
        user_input = input("👩‍🌾 Kamu: ")
        if user_input.lower() == 'exit':
            break

        tokens = detect_phrases_and_preprocess(user_input)
        print("🔍 Token + Phrase:", tokens)

        facts = extract_facts(tokens)
        print("📝 Fakta yang terdeteksi:", facts)

        if not facts:
            print("🤖 Bot: Maaf, saya belum bisa mengerti kondisi tanaman kamu. Coba jelaskan dengan kata lain ya.")
            continue

        conclusions = rule_based_inference(facts)

        if conclusions:
            print("🤖 Bot: Berikut hasil analisis dan rekomendasi:")
            for k, v in conclusions.items():
                if k == 'penyebab':
                    print(f"- Penyebab kemungkinan: {v.replace('_', ' ')}")
                elif k == 'frekuensi_pemupukan':
                    print(f"- Frekuensi pemupukan yang disarankan: {v.replace('_', ' ')}")
                elif k == 'tambahkan':
                    print(f"- Disarankan menambahkan: {v.replace('_', ' ')}")
        else:
            print("🤖 Bot: Saya tidak menemukan masalah berdasarkan input kamu. Silakan coba info lain.")

if __name__ == "__main__":
    chatbot()
