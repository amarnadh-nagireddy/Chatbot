import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from chromadb import PersistentClient

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

# ---------------- LLM & embeddings ----------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(
    groq_api_key=os.environ["GROQ_API_KEY"],
    model_name="openai/gpt-oss-120b"
)

PERSIST_DIR = "chroma_persist"
client = PersistentClient(path=PERSIST_DIR)  # persistent Chroma client

def create_vector_embedding(file_path):
    """
    Embed a PDF and store it in Chroma.
    No tracking, no list, no deletion.
    """
    vectorstore = Chroma(
        client=client,
        collection_name="rag_collection",
        embedding_function=embeddings
    )

    filename = os.path.basename(file_path)
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Add metadata for source tracking (optional)
    for doc in documents:
        doc.metadata["source"] = filename

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=160)
    final_docs = splitter.split_documents(documents)

    # Add documents to Chroma
    vectorstore.add_documents(final_docs)

    return vectorstore


def get_rag_chain(vectorstore: Chroma):
    retriever = vectorstore.as_retriever()

    # QA prompt
    system_prompt = (
        "You are a Theory of Computation teacher bot providing accurate, concise answers. "
        "Use the following context to answer the question. If you don't know, say so truthfully. "
        "Use ASCII art for equations or automata diagrams when helpful. Context:\n\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Session history store
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    return conversational_rag_chain
