
import eventlet
eventlet.monkey_patch()


from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import os, uuid
from rag_chain import load_or_create_vectorstore, get_rag_chain

app = Flask(__name__)
# allow cross origin requests from the frontend
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

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
    """
    Handle incoming WebSocket chat messages.
    Expects:
    {
        "session_id": "optional-session-id",
        "message": "User question"
    }
    """
    user_message = data.get("message")
    session_id = data.get("session_id")

    if not session_id:
        session_id = str(uuid.uuid4())



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
    socketio.run(app, host='0.0.0.0', port=8000, debug=True)

