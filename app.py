from flask import Flask, request, jsonify, render_template
from plant_chatbot import detect_phrases_and_preprocess, extract_facts, rule_based_inference, get_general_advice

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    if len(user_input.strip()) < 3:
        return jsonify({'response': 'Mohon berikan pertanyaan yang lebih lengkap.'})

    tokens = detect_phrases_and_preprocess(user_input)
    facts = extract_facts(tokens, user_input)
    conclusions = rule_based_inference(facts)
    general_advice = get_general_advice(user_input)

    response_parts = []

    # Tampilkan hasil NLP dan prosesnya
    response_parts.append('--- Hasil NLP & Proses ---')
    response_parts.append(f"Token hasil NLP: <span class='nlp-token'>{tokens}</span>")
    response_parts.append(f"Fakta yang diekstrak: <span class='nlp-fakta'>{facts}</span>")
    response_parts.append('--------------------------')

    if 'kemungkinan_penyebab' in conclusions:
        response_parts.append(f"🔍 Kemungkinan penyebab: {conclusions['kemungkinan_penyebab'].replace('_', ' ')}")

    if conclusions.get('rekomendasi'):
        response_parts.append("💡 Rekomendasi:")
        for i, r in enumerate(conclusions['rekomendasi'][:5], 1):
            response_parts.append(f"{i}. {r}")

    if general_advice:
        response_parts.append("📚 Saran umum:")
        for a in general_advice:
            response_parts.append(f"• {a}")

    if not response_parts:
        response_parts.append("🤔 Saya perlu informasi lebih spesifik. Jelaskan lebih detail ya!")

    return jsonify({'response': '\n'.join(response_parts)})

if __name__ == '__main__':
    app.run(debug=True)
