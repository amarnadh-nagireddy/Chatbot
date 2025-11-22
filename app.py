from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import logging
import uuid

from rag_chain import load_or_create_vectorstore, get_rag_chain

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI()

# Load RAG chain (same as in Flask)
vectorstore = load_or_create_vectorstore()
rag_chain = get_rag_chain(vectorstore)

# Session storage (you weren't really using it, but keeping it for parity)
sessions = {}


@app.get("/")
async def home():
    # Same semantics as Flask route, but JSON instead of plain text
    return {"message": "RAG WebSocket Server is running!"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            # Expect JSON: { "session_id": "...", "message": "User question" }
            data = await websocket.receive_json()

            user_message = data.get("message")
            session_id = data.get("session_id") or str(uuid.uuid4())

            if not user_message:
                await websocket.send_json({"error": "Message is required."})
                continue

            try:
                # Same logic as in Flask: sync RAG call with session_id
                response = rag_chain.invoke(
                    {"input": user_message},
                    config={"configurable": {"session_id": session_id}}
                )

                await websocket.send_json({
                    "session_id": session_id,
                    "response": response.get("answer", "No answer generated.")
                })

            except Exception as e:
                logger.exception("Error during RAG invocation")
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
