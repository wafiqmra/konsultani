from flask import Flask, render_template, request, jsonify
import chatbot

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    # Use chatbot logic to get response
    tokens = chatbot.detect_phrases_and_preprocess(user_input, chatbot.bigram_mod, chatbot.trigram_mod, chatbot.stopwords, chatbot.stemmer)
    responses = []
    for phrase in tokens:
        if phrase in chatbot.knowledge_base:
            responses.append(chatbot.knowledge_base[phrase])
    if responses:
        bot_reply = ' '.join(responses)
    else:
        bot_reply = 'Maaf, saya belum tahu jawabannya. Coba tanyakan hal lain.'
    return jsonify({'reply': bot_reply, 'tokens': tokens})

if __name__ == '__main__':
    app.run(debug=True)
