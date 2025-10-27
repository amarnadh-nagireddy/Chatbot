import os
import uuid
import shutil
import asyncio
import threading
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from rag_chain import create_vector_embedding, get_rag_chain

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "research_papers"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

vectorstore_lock = threading.Lock()
vectorstore = None
rag_chain = None

@app.get("/")
async def home():
    return {"message": "✅ RAG WebSocket Server is running!"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vectorstore, rag_chain
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        with vectorstore_lock:
            if vectorstore is None:
                vectorstore = create_vector_embedding(file_path)
            else:
                create_vector_embedding(file_path)
            rag_chain = get_rag_chain(vectorstore)
        os.remove(file_path)
        return {"message": f"✅ File '{file.filename}' processed and embeddings added."}
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    global rag_chain
    await websocket.accept()
    session_id = str(uuid.uuid4())
    try:
        while True:
            data = await websocket.receive_json()
            user_message = data.get("message")
            if not user_message:
                await websocket.send_json({"error": "Message is required."})
                continue
            if rag_chain is None:
                await websocket.send_json({"error": "RAG chain not initialized. Upload a PDF first."})
                continue
            try:
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: rag_chain.invoke(
                        {"input": user_message},
                        config={"configurable": {"session_id": session_id}},
                    )
                )
                await websocket.send_json({
                    "session_id": session_id,
                    "response": response.get("answer", "No answer generated.")
                })
            except Exception as e:
                await websocket.send_json({"error": str(e)})
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_json({"error": f"Unexpected error: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
