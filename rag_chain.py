import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# Load env keys
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY", "")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN", "")

# Initialize embeddings and LLM
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(
    groq_api_key=os.environ['GROQ_API_KEY'],
    model_name="openai/gpt-oss-120b"
)
# In rag_chain.py
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ============= VECTORSTORE CREATION =============

def create_vector_embedding(pdf_dir="research_papers", persist_dir="chroma_persist"):
    """Loads or creates a Chroma vector DB from PDFs."""
    if os.path.exists(persist_dir):  
        # ✅ Load the existing persisted vectorstore
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        return vectorstore

    # Else: create new vectorstore
    loader = PyPDFDirectoryLoader(pdf_dir)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=160)
    final_documents = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=final_documents,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    return vectorstore




# ============= RAG CHAIN WITH HISTORY =============
def get_rag_chain(vectorstore: Chroma):
    retriever = vectorstore.as_retriever()

    # Contextualization prompt (turns chat-dependent Qs into standalone Qs)
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history, "
        "formulate a standalone question which can be understood without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Answering prompt
    system_prompt = (
        "You are an Theory of computation Teacher bot providing helpful answers based on the provided context. "
        "From the context, answer the question at the end. {context}"
        "For some cases like drwaings or mathematical equations, you can use ASCII art to represent them."
        "Use the following pieces of retrieved context to answer the question."
        "If you don't know the answer, say that you don't know but dont use any fabricated information. "
        "From the context compare the different theories with the question and apply them to get the best answers and provide examples. "
        "Keep answers concise and precise.\n\n"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Store chat history per session
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
