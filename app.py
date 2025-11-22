from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import os, uuid
from rag_chain import load_or_create_vectorstore, get_rag_chain

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

vectorstore = load_or_create_vectorstore()
rag_chain = get_rag_chain(vectorstore)

sessions = {}

@app.route("/")
def home():
    return "RAG WebSocket Server is running!"

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('chat')
def handle_chat(data):
    user_message = data.get("message")
    session_id = data.get("session_id") or str(uuid.uuid4())

    if not user_message:
        emit('response', {"error": "Message is required."})
        return

    try:
        response = rag_chain.invoke(
            {"input": user_message},
            config={"configurable": {"session_id": session_id}}
        )

        emit('response', {
            "session_id": session_id,
            "response": response.get("answer", "No answer generated.")
        })

    except Exception as e:
        emit('response', {"error": str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)
