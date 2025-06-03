import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from gensim.models import Phrases
from gensim.models.phrases import Phraser

# Load stopwords
stopword_df = pd.read_csv('stopwordbahasa.csv', header=None)
stopwords = set(stopword_df[0].tolist())

# Inisialisasi stemmer
stemmer = StemmerFactory().create_stemmer()

# Contoh kalimat untuk membentuk model bigram & trigram
example_texts = [
    "penyimpanan optimal dalam suhu 20-30 derajat",
    "media tanam hidroponik dengan garam khusus",
    "penggunaan pot kulit padat di lahan sempit",
    "kemasan berlubang kecil menjaga ventilasi",
    "waktu semprot pagi sore dan tidak hujan",
    "daun bercak kecil akibat hama atau penyakit"
]

def tokenize(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    return text.split()

# Latih bigram dan trigram
base_tokens = [tokenize(txt) for txt in example_texts]
bigram = Phrases(base_tokens, min_count=1, threshold=1)
trigram = Phrases(bigram[base_tokens], threshold=1)
bigram_mod = Phraser(bigram)
trigram_mod = Phraser(trigram)

# NLP Preprocessing
def detect_phrases_and_preprocess(text):
    tokens = tokenize(text)
    tokens = trigram_mod[bigram_mod[tokens]]
    tokens = [stemmer.stem(t) for t in tokens]
    tokens = [t for t in tokens if t not in stopwords]
    return tokens

# Ekstraksi fakta dari token
def extract_facts(tokens):
    facts = {}

    # Kondisi pertumbuhan
    if 'lambat' in tokens:
        facts['pertumbuhan'] = 'lambat'
    if 'kerdil' in tokens:
        facts['ukuran'] = 'kerdil'
    if 'cepat' in tokens:
        facts['pertumbuhan'] = 'cepat'
    if 'besar' in tokens or 'subur' in tokens:
        facts['ukuran'] = 'besar'
    if 'kuning' in tokens and 'daun' in tokens:
        facts['daun'] = 'menguning'
    if 'bercak' in tokens:
        facts['daun'] = 'bercak'
    if 'kecil' in tokens and 'daun' in tokens:
        facts['daun'] = 'kecil'
    if 'melintir' in tokens:
        facts['daun'] = 'melintir'
    if 'tidak' in tokens and 'sehat' in tokens:
        facts['tanaman'] = 'tidak_sehat'

    # Tanah dan pupuk
    if 'asam' in tokens:
        facts['pH_tanah'] = 'asam'
    if 'basa' in tokens:
        facts['pH_tanah'] = 'basa'
    if 'rendah' in tokens and 'ph' in tokens:
        facts['pH_tanah'] = 'rendah'
    if 'kandang' in tokens:
        facts['jenis_pupuk'] = 'kandang'
    if 'organik' in tokens:
        facts['jenis_pupuk'] = 'organik'

    # Media tanam dan gulma
    if 'gulma' in tokens:
        facts['media_tanam'] = 'gulma'
    if 'cacing' in tokens and 'akar' in tokens:
        facts['akar'] = 'cacing'

    # Budidaya
    if 'sepit' in tokens or 'sempit' in tokens:
        facts['lahan'] = 'sempit'
    if 'mudah' in tokens and 'rawat' in tokens:
        facts['perawatan'] = 'mudah'
    if 'tinggi' in tokens and 'rawat' in tokens:
        facts['perawatan'] = 'tinggi'
    if 'hidroponik' in tokens:
        facts['metode_budidaya'] = 'hidroponik'
    if 'optimal' in tokens:
        facts['hasil_optimal'] = 'diinginkan'

    # Penyimpanan
    if '2030c' in tokens or '2030' in tokens:
        facts['suhu'] = '20_30C'
    if 'tutup' in tokens and 'ventilasi' in tokens:
        facts['ventilasi'] = 'tertutup'
    if 'kemas' in tokens and 'lubang' in tokens:
        facts['kemasan'] = 'berlubang_kecil'
    if 'lama' in tokens and 'simpan' in tokens:
        facts['waktu_simpan'] = 'lama'

    # Lain-lain
    if 'ulang' in tokens and 'hama' in tokens:
        facts['serangan_hama'] = 'berulang'
    if 'hujan' in tokens and 'tinggi' in tokens:
        facts['curah_hujan'] = 'tinggi'
    if 'cukup' in tokens and 'hujan' in tokens:
        facts['hujan'] = 'cukup'
    if 'pagi' in tokens and 'sore' in tokens:
        facts['waktu_semprot'] = 'pagi_sore'
    if 'tidak' in tokens and 'hujan' in tokens:
        facts['cuaca'] = 'tidak_hujan'

    return facts

# Inference rules
def rule_based_inference(facts):
    conclusions = {}

    # Penyebab umum
    if facts.get('pertumbuhan') == 'lambat' and facts.get('ukuran') == 'kerdil':
        conclusions['penyebab'] = 'nutrisi_rendah'
    if facts.get('daun') == 'menguning' and facts.get('pertumbuhan') == 'lambat':
        conclusions['penyebab'] = 'nutrisi_rendah / sinar_matahari_kurang / gulma_ada'
    if facts.get('daun') == 'bercak':
        conclusions['penyebab'] = 'hama / penyakit'
    if facts.get('daun') == 'kecil':
        conclusions['nutrisi'] = 'kurang'
    if facts.get('penyebab') == 'hama':
        conclusions['tindakan'] = 'pestisida_alami'
    if facts.get('tanaman') == 'tidak_sehat':
        conclusions['lakukan'] = 'pemangkasan'
        conclusions['ganti_media_tanam'] = 'ya'
    if facts.get('media_tanam') == 'gulma':
        conclusions['lakukan'] = 'pembersihan'
    if facts.get('akar') == 'cacing':
        conclusions['gunakan'] = 'furadan'
    if facts.get('serangan_hama') == 'berulang':
        conclusions['gunakan'] = 'predator_alami'

    # pH dan pupuk
    if facts.get('pH_tanah') == 'asam' or facts.get('pH_tanah') == 'rendah':
        conclusions['tambahkan'] = 'kapur_dolomit'
    if facts.get('jenis_pupuk') == 'kandang' and facts.get('penyebab') == 'nutrisi_rendah':
        conclusions['frekuensi_pemupukan'] = '3_bulan'

    # Budidaya dan metode
    if facts.get('lahan') == 'sempit' and facts.get('perawatan') == 'mudah':
        conclusions['metode_budidaya'] = 'pot_kulit_padat'
    if facts.get('hasil_optimal') == 'diinginkan' and facts.get('perawatan') == 'tinggi':
        conclusions['metode_budidaya'] = 'hidroponik'
    if facts.get('metode_budidaya') == 'hidroponik':
        conclusions['tambahkan'] = 'garam_khusus'

    # Penyimpanan
    if facts.get('suhu') == '20_30C' and facts.get('ventilasi') == 'tertutup' and facts.get('kemasan') == 'berlubang_kecil':
        conclusions['penyimpanan'] = 'optimal'
    if facts.get('waktu_simpan') == 'lama':
        conclusions['gunakan'] = 'pengeringan / pendinginan'

    # Cuaca
    if facts.get('curah_hujan') == 'tinggi':
        conclusions['jumlah_hama'] = 'banyak'
    if facts.get('waktu_semprot') == 'pagi_sore' and facts.get('cuaca') == 'tidak_hujan':
        conclusions['efektivitas_semprot'] = 'tinggi'
    if facts.get('hujan') == 'cukup':
        conclusions['penyiraman'] = 'tidak_perlu'

    return conclusions

# Fungsi tambahan: penjelasan setiap kesimpulan
def explain_conclusions(conclusions):
    explanation_map = {
        'penyebab': {
            'nutrisi_rendah': "Tanaman kemungkinan kekurangan nutrisi yang diperlukan.",
            'nutrisi_rendah / sinar_matahari_kurang / gulma_ada': "Pertumbuhan tanaman lambat bisa karena nutrisi kurang, kurang sinar matahari, atau adanya gulma.",
            'hama / penyakit': "Daun bercak kemungkinan disebabkan oleh serangan hama atau penyakit."
        },
        'nutrisi': {
            'kurang': "Nutrisi tanaman kurang sehingga menyebabkan daun kecil."
        },
        'tindakan': {
            'pestisida_alami': "Disarankan menggunakan pestisida alami untuk mengendalikan hama."
        },
        'lakukan': {
            'pemangkasan': "Lakukan pemangkasan untuk tanaman yang tidak sehat.",
            'pembersihan': "Bersihkan gulma yang ada pada media tanam."
        },
        'ganti_media_tanam': {
            'ya': "Disarankan mengganti media tanam untuk kesehatan tanaman."
        },
        'gunakan': {
            'furadan': "Gunakan Furadan untuk mengatasi cacing pada akar.",
            'predator_alami': "Gunakan predator alami untuk mengendalikan serangan hama yang berulang.",
            'pengeringan / pendinginan': "Gunakan teknik pengeringan atau pendinginan untuk penyimpanan yang lama."
        },
        'tambahkan': {
            'kapur_dolomit': "Tambahkan kapur dolomit untuk menetralkan pH tanah yang asam atau rendah.",
            'garam_khusus': "Tambahkan garam khusus untuk metode budidaya hidroponik."
        },
        'frekuensi_pemupukan': {
            '3_bulan': "Pemupukan dilakukan setiap 3 bulan untuk pupuk kandang."
        },
        'metode_budidaya': {
            'pot_kulit_padat': "Metode budidaya menggunakan pot kulit padat cocok untuk lahan sempit dan perawatan mudah.",
            'hidroponik': "Metode budidaya hidroponik memerlukan perawatan yang lebih tinggi dan hasil optimal."
        },
        'penyimpanan': {
            'optimal': "Penyimpanan sudah optimal dengan suhu dan ventilasi yang sesuai."
        },
        'jumlah_hama': {
            'banyak': "Curah hujan tinggi dapat meningkatkan jumlah hama."
        },
        'efektivitas_semprot': {
            'tinggi': "Penyemprotan pada pagi dan sore saat tidak hujan memiliki efektivitas tinggi."
        },
        'penyiraman': {
            'tidak_perlu': "Penyiraman tidak diperlukan jika hujan sudah cukup."
        }
    }

    explanations = []
    for key, value in conclusions.items():
        if key in explanation_map and value in explanation_map[key]:
            explanations.append(explanation_map[key][value])
        else:
            explanations.append(f"{key.replace('_', ' ').capitalize()}: {value.replace('_', ' ')}")
    return explanations


# Chatbot interaktif
def chatbot():
    print("🌱 Chatbot Pertanian Siap Membantu! (Ketik 'exit' untuk keluar)\n")
    while True:
        user_input = input("👩‍🌾 Kamu: ")
        if user_input.lower() == 'exit':
            print("Terima kasih sudah menggunakan chatbot. Sampai jumpa!")
            break

        tokens = detect_phrases_and_preprocess(user_input)
        facts = extract_facts(tokens)
        conclusions = rule_based_inference(facts)

        if not conclusions:
            print("🤖 Bot: Maaf, saya belum bisa memberikan kesimpulan dari kalimat tersebut.\n")
            continue

        print("🤖 Bot: Berdasarkan input kamu, berikut kesimpulan saya:")
        for k, v in conclusions.items():
            print(f" - {k.replace('_', ' ').capitalize()}: {v.replace('_', ' ')}")

        # Tampilkan penjelasan
        print("\nPenjelasan:")
        explanations = explain_conclusions(conclusions)
        for exp in explanations:
            print(f" - {exp}")
        print()

if __name__ == "__main__":
    chatbot()
