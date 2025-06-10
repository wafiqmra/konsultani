import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from gensim.models import Phrases
from gensim.models.phrases import Phraser

# Load stopwords
try:
    stopword_df = pd.read_csv('stopwordbahasa.csv', header=None)
    stopwords = set(stopword_df[0].tolist())
except:
    # Fallback stopwords jika file tidak ada
    stopwords = {'dan', 'atau', 'ini', 'itu', 'yang', 'untuk', 'dengan', 'dari', 'ke', 'di', 'pada', 'dalam', 'adalah', 'akan', 'sudah', 'telah', 'dapat', 'bisa', 'juga', 'saya', 'aku', 'kamu', 'dia', 'mereka', 'kita'}

# Inisialisasi stemmer
stemmer = StemmerFactory().create_stemmer()

# Contoh kalimat untuk membentuk model bigram & trigram
example_texts = [
    "penyimpanan optimal dalam suhu 20-30 derajat",
    "media tanam hidroponik dengan garam khusus",
    "penggunaan pot kulit padat di lahan sempit",
    "kemasan berlubang kecil menjaga ventilasi",
    "waktu semprot pagi sore dan tidak hujan",
    "daun bercak kecil akibat hama atau penyakit",
    "daun menguning karena kekurangan nutrisi",
    "pertumbuhan lambat dan tanaman kerdil",
    "serangan hama berulang pada tanaman",
    "tanah asam perlu kapur dolomit"
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
    tokens = [t for t in tokens if t not in stopwords and len(t) > 1]
    return tokens

# Fungsi untuk deteksi angka suhu
def extract_temperature(text):
    # Cari pola angka diikuti derajat atau celsius
    temp_pattern = r'(\d+)[-\s]*(\d+)?\s*(?:derajat|celsius|c°|°c)'
    match = re.search(temp_pattern, text.lower())
    if match:
        temp1 = int(match.group(1))
        if match.group(2):
            temp2 = int(match.group(2))
            if 20 <= temp1 <= 30 or 20 <= temp2 <= 30:
                return '20_30C'
        elif 20 <= temp1 <= 30:
            return '20_30C'
    
    # Cari angka standalone
    numbers = re.findall(r'\d+', text)
    for num in numbers:
        temp = int(num)
        if 20 <= temp <= 30:
            return '20_30C'
    return None

# Ekstraksi fakta dari token dan teks asli
def extract_facts(tokens, original_text):
    facts = {}
    text_lower = original_text.lower()
    
    # Deteksi suhu
    suhu_detected = extract_temperature(original_text)
    if suhu_detected:
        facts['suhu'] = suhu_detected

    # Kondisi daun (lebih fleksibel)
    daun_conditions = {
        'menguning': ['kuning', 'menguning', 'kekuningan', 'pucat'],
        'bercak': ['bercak', 'flek', 'spot', 'noda'],
        'kecil': ['kecil', 'mengecil', 'menyusut'],
        'melintir': ['melintir', 'keriting', 'menggulung', 'bengkok'],
        'rontok': ['rontok', 'gugur', 'jatuh', 'copot'],
        'layu': ['layu', 'lemas', 'lemah', 'tidak segar']
    }
    
    for condition, keywords in daun_conditions.items():
        if any(keyword in text_lower for keyword in keywords):
            if 'daun' in text_lower or 'leaf' in text_lower:
                facts['daun'] = condition
                break

    # Kondisi pertumbuhan
    growth_conditions = {
        'lambat': ['lambat', 'terhambat', 'tidak berkembang', 'stagnan'],
        'cepat': ['cepat', 'pesat', 'subur', 'berkembang baik'],
        'terhenti': ['terhenti', 'berhenti', 'mandeg', 'stuck']
    }
    
    for condition, keywords in growth_conditions.items():
        if any(keyword in text_lower for keyword in keywords):
            facts['pertumbuhan'] = condition
            break

    # Ukuran tanaman
    size_conditions = {
        'kerdil': ['kerdil', 'pendek', 'kecil', 'tidak besar'],
        'besar': ['besar', 'tinggi', 'subur', 'rimbun'],
        'normal': ['normal', 'biasa', 'standar']
    }
    
    for size, keywords in size_conditions.items():
        if any(keyword in text_lower for keyword in keywords):
            facts['ukuran'] = size
            break

    # Kondisi tanaman umum
    if any(word in text_lower for word in ['tidak sehat', 'sakit', 'bermasalah', 'stress']):
        facts['tanaman'] = 'tidak_sehat'
    elif any(word in text_lower for word in ['sehat', 'baik', 'normal']):
        facts['tanaman'] = 'sehat'

    # pH tanah
    ph_conditions = {
        'asam': ['asam', 'rendah', 'ph rendah', 'acidic'],
        'basa': ['basa', 'tinggi', 'ph tinggi', 'alkaline'],
        'netral': ['netral', 'normal', 'seimbang']
    }
    
    for ph, keywords in ph_conditions.items():
        if any(keyword in text_lower for keyword in keywords):
            if 'tanah' in text_lower or 'ph' in text_lower:
                facts['pH_tanah'] = ph
                break

    # Jenis pupuk
    pupuk_types = {
        'kandang': ['kandang', 'kompos', 'organik'],
        'kimia': ['kimia', 'npk', 'urea', 'sintesis'],
        'cair': ['cair', 'liquid'],
        'granul': ['granul', 'butiran']
    }
    
    for pupuk, keywords in pupuk_types.items():
        if any(keyword in text_lower for keyword in keywords):
            if 'pupuk' in text_lower:
                facts['jenis_pupuk'] = pupuk
                break

    # Media tanam
    if any(word in text_lower for word in ['gulma', 'rumput liar', 'tanaman pengganggu']):
        facts['media_tanam'] = 'gulma'
    if any(word in text_lower for word in ['hidroponik', 'hidro', 'air']):
        facts['metode_budidaya'] = 'hidroponik'

    # Hama dan penyakit
    if any(word in text_lower for word in ['hama', 'serangga', 'ulat', 'kutu']):
        facts['masalah'] = 'hama'
    if any(word in text_lower for word in ['penyakit', 'jamur', 'bakteri', 'virus']):
        facts['masalah'] = 'penyakit'
    if any(word in text_lower for word in ['berulang', 'terus menerus', 'sering']):
        facts['serangan'] = 'berulang'

    # Akar
    if any(word in text_lower for word in ['cacing', 'nematoda', 'belatung']):
        if 'akar' in text_lower:
            facts['akar'] = 'cacing'

    # Lahan
    if any(word in text_lower for word in ['sempit', 'kecil', 'terbatas']):
        facts['lahan'] = 'sempit'
    elif any(word in text_lower for word in ['luas', 'besar', 'lebar']):
        facts['lahan'] = 'luas'

    # Perawatan
    if any(word in text_lower for word in ['mudah', 'simple', 'sederhana']):
        facts['perawatan'] = 'mudah'
    elif any(word in text_lower for word in ['sulit', 'rumit', 'ribet', 'intensif']):
        facts['perawatan'] = 'tinggi'

    # Cuaca
    if any(word in text_lower for word in ['hujan', 'basah', 'lembab']):
        facts['cuaca'] = 'hujan'
    elif any(word in text_lower for word in ['kering', 'panas', 'tidak hujan']):
        facts['cuaca'] = 'kering'

    # Waktu
    if any(word in text_lower for word in ['pagi', 'sore', 'morning', 'evening']):
        facts['waktu'] = 'pagi_sore'

    # Hapus fakta 'tanaman': 'sehat' jika itu satu-satunya fakta dan input tidak mengandung kata 'sehat', 'baik', atau 'normal'
    if list(facts.keys()) == ['tanaman'] and facts['tanaman'] == 'sehat' and not any(word in text_lower for word in ['sehat', 'baik', 'normal']):
        facts = {}
        
    return facts

# Inference rules yang lebih komprehensif
def rule_based_inference(facts):
    conclusions = {}
    recommendations = []

    # Daun menguning - berbagai penyebab dan solusi
    if facts.get('daun') == 'menguning':
        conclusions['kemungkinan_penyebab'] = 'kekurangan_nitrogen / overwatering / underwatering / penyakit'
        recommendations.extend([
            "Periksa kelembaban tanah - jangan terlalu basah atau kering",
            "Berikan pupuk nitrogen (urea atau pupuk kandang)",
            "Pastikan drainase tanah baik",
            "Periksa adanya hama atau penyakit pada akar",
            "Berikan pupuk daun dengan kandungan nitrogen tinggi"
        ])

    # Daun bercak
    if facts.get('daun') == 'bercak':
        conclusions['kemungkinan_penyebab'] = 'penyakit_jamur / bakteri / virus'
        recommendations.extend([
            "Semprotkan fungisida organik (baking soda + sabun)",
            "Potong dan buang daun yang terinfeksi",
            "Perbaiki sirkulasi udara di sekitar tanaman",
            "Hindari menyiram daun, siram bagian akar saja",
            "Gunakan bakterisida alami jika perlu"
        ])

    # Pertumbuhan lambat
    if facts.get('pertumbuhan') == 'lambat':
        conclusions['kemungkinan_penyebab'] = 'kekurangan_nutrisi / cahaya_kurang / akar_bermasalah'
        recommendations.extend([
            "Pindahkan ke lokasi dengan cahaya lebih banyak",
            "Berikan pupuk NPK seimbang",
            "Periksa kondisi akar - mungkin perlu repotting",
            "Pastikan pH tanah optimal (6.0-7.0)",
            "Berikan pupuk cair seminggu sekali"
        ])

    # Tanaman kerdil
    if facts.get('ukuran') == 'kerdil':
        conclusions['kemungkinan_penyebab'] = 'pot_terlalu_kecil / nutrisi_kurang / genetik'
        recommendations.extend([
            "Pindahkan ke pot yang lebih besar",
            "Berikan pupuk dengan kandungan fosfor tinggi",
            "Pastikan akar tidak terikat dalam pot",
            "Berikan ruang yang cukup untuk pertumbuhan"
        ])

    # Tanaman tidak sehat
    if facts.get('tanaman') == 'tidak_sehat':
        recommendations.extend([
            "Lakukan pemangkasan bagian yang mati/sakit",
            "Ganti sebagian media tanam dengan yang baru",
            "Periksa drainase dan aerasi tanah",
            "Berikan nutrisi tambahan secara bertahap",
            "Isolasi dari tanaman lain jika perlu"
        ])

    # Masalah hama
    if facts.get('masalah') == 'hama':
        recommendations.extend([
            "Semprotkan pestisida organik (neem oil)",
            "Gunakan perangkap kuning untuk serangga terbang",
            "Tanam tanaman pengusir hama (kemangi, lavender)",
            "Bersihkan area sekitar tanaman",
            "Gunakan predator alami jika memungkinkan"
        ])

    # Masalah penyakit
    if facts.get('masalah') == 'penyakit':
        recommendations.extend([
            "Isolasi tanaman yang terinfeksi",
            "Gunakan fungisida organik",
            "Perbaiki ventilasi udara",
            "Kurangi kelembaban berlebih",
            "Sterilkan alat berkebun setelah digunakan"
        ])

    # pH tanah asam
    if facts.get('pH_tanah') == 'asam':
        recommendations.extend([
            "Tambahkan kapur dolomit ke tanah",
            "Gunakan abu sekam untuk menaikkan pH",
            "Berikan kompos matang untuk buffer pH",
            "Tunggu 2-4 minggu setelah aplikasi kapur sebelum menanam"
        ])

    # Lahan sempit
    if facts.get('lahan') == 'sempit':
        recommendations.extend([
            "Gunakan sistem vertikultur (bertingkat)",
            "Pilih varietas tanaman yang compact",
            "Gunakan pot gantung untuk maksimalkan ruang",
            "Pertimbangkan sistem hidroponik sederhana"
        ])

    # Metode hidroponik
    if facts.get('metode_budidaya') == 'hidroponik':
        recommendations.extend([
            "Gunakan nutrisi AB mix khusus hidroponik",
            "Periksa pH larutan nutrisi (5.5-6.5)",
            "Ganti larutan nutrisi setiap 1-2 minggu",
            "Pastikan aerasi/oksigen dalam larutan cukup",
            "Monitor EC/TDS larutan nutrisi"
        ])

    # Cuaca hujan
    if facts.get('cuaca') == 'hujan':
        recommendations.extend([
            "Pastikan drainase tanah baik",
            "Hindari penyiraman berlebih",
            "Berikan naungan jika hujan terlalu deras",
            "Waspada terhadap penyakit jamur",
            "Periksa tanaman lebih sering"
        ])

    # Rekomendasi umum jika tidak ada masalah spesifik
    if not recommendations:
        recommendations.extend([
            "Pastikan tanaman mendapat cahaya cukup (6-8 jam/hari)",
            "Siram secara teratur tapi jangan berlebihan",
            "Berikan pupuk seimbang setiap 2-4 minggu",
            "Periksa tanaman secara rutin untuk deteksi dini masalah",
            "Jaga kebersihan area sekitar tanaman"
        ])

    conclusions['rekomendasi'] = recommendations
    return conclusions

# Fungsi untuk memberikan solusi berdasarkan keyword
def get_general_advice(text):
    advice = []
    text_lower = text.lower()
    
    # Panduan umum berdasarkan kata kunci
    if any(word in text_lower for word in ['cara', 'bagaimana', 'how']):
        if 'menanam' in text_lower:
            advice.append("Pilih benih berkualitas, siapkan media tanam yang gembur, tanam dengan kedalaman 2-3x ukuran benih")
        if 'merawat' in text_lower:
            advice.append("Siram teratur, beri pupuk sesuai kebutuhan, pangkas bagian mati, kontrol hama penyakit")
        if 'panen' in text_lower:
            advice.append("Panen saat buah/sayuran sudah matang optimal, gunakan alat bersih, simpan di tempat yang tepat")
    
    if any(word in text_lower for word in ['pupuk', 'nutrisi', 'fertilizer']):
        advice.append("Gunakan pupuk NPK seimbang, kombinasikan pupuk organik dan anorganik, sesuaikan dengan fase pertumbuhan tanaman")
    
    if any(word in text_lower for word in ['air', 'siram', 'water']):
        advice.append("Siram pada pagi atau sore hari, hindari genangan air, sesuaikan frekuensi dengan cuaca dan jenis tanaman")
    
    return advice

# Chatbot interaktif yang lebih responsif
def chatbot():
    print("🌱 Chatbot Pertanian Siap Membantu! (Ketik 'exit' untuk keluar)")
    print("💡 Contoh pertanyaan: 'daun saya menguning', 'tanaman tumbuh lambat', 'cara merawat tomat'")
    print("=" * 60)
    
    while True:
        user_input = input("\n👩‍🌾 Anda: ")
        if user_input.lower() == 'exit':
            print("👋 Terima kasih telah menggunakan chatbot pertanian! Selamat berkebun!")
            break

        if len(user_input.strip()) < 3:
            print("🤖 Bot: Mohon berikan pertanyaan yang lebih lengkap.")
            continue

        # Proses input
        tokens = detect_phrases_and_preprocess(user_input)
        print(f"🔍 Kata kunci terdeteksi: {', '.join(tokens)}")

        facts = extract_facts(tokens, user_input)
        print(f"📝 Fakta yang teranalisis: {facts}")

        conclusions = rule_based_inference(facts)
        general_advice = get_general_advice(user_input)

        # Tampilkan hasil
        print("\n🤖 Bot: Berikut analisis dan rekomendasi saya:")
        print("-" * 50)
        
        # Tampilkan penyebab jika ada
        if 'kemungkinan_penyebab' in conclusions:
            print(f"🔍 Kemungkinan penyebab: {conclusions['kemungkinan_penyebab'].replace('_', ' ')}")
        
        # Tampilkan rekomendasi
        if 'rekomendasi' in conclusions:
            print("💡 Rekomendasi tindakan:")
            for i, rec in enumerate(conclusions['rekomendasi'][:5], 1):  # Tampilkan max 5 rekomendasi
                print(f"   {i}. {rec}")
        
        # Tampilkan saran umum jika ada
        if general_advice:
            print("📚 Saran umum:")
            for advice in general_advice:
                print(f"   • {advice}")
        
        # Jika tidak ada hasil spesifik
        if not conclusions.get('rekomendasi') and not general_advice:
            print("🤔 Maaf, saya perlu informasi lebih spesifik untuk memberikan saran yang tepat.")
            print("💭 Coba deskripsikan masalah dengan lebih detail, misalnya:")
            print("   • Kondisi daun (warna, bentuk, dll)")
            print("   • Pertumbuhan tanaman")
            print("   • Jenis tanaman yang ditanam")
            print("   • Kondisi lingkungan")

# Run chatbot
if __name__ == "__main__":
    chatbot()