from flask import Flask, render_template, request, jsonify
from stock_chatbot import StockAssistantChatbot
from flask_cors import CORS  # Enable cross-origin requests

app = Flask(__name__)
CORS(app)  # Allow requests from frontend if needed
print("starting....")
chatbot = StockAssistantChatbot()  # No duplicate app initialization

@app.route('/')
def index():
    return render_template('chatweb1.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        if not request.is_json:  # Ensure request contains JSON
            return jsonify({'error': 'Invalid request. JSON data expected.'}), 400

        user_message = request.json.get('message', '').strip()
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        response = chatbot.get_response(user_message)
        return jsonify({'response': response})

    except Exception as e:
        return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)

