from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import os
import uuid
from werkzeug.utils import secure_filename
from rag_chain import load_or_create_vectorstore, create_vector_embedding, get_rag_chain

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = "research_papers"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

UPLOAD_FOLDER = "research_papers"
vectorstore = load_or_create_vectorstore() 
rag_chain = get_rag_chain(vectorstore)

sessions = {}


@app.route("/")
def home():
    return "RAG WebSocket Server is running!"

@app.route("/upload", methods=["POST"])
def upload_file():
    """
    Upload a PDF file and recreate embeddings.
    """
    global vectorstore, rag_chain

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are allowed"}), 400


    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        vectorstore = create_vector_embedding(file_path)
        rag_chain = get_rag_chain(vectorstore)
        return jsonify({
            "message": f"File '{filename}' uploaded and embeddings created successfully."
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host='0.0.0.0', port=port)
