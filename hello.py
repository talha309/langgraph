import chainlit as cl
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from typing import TypedDict
import os 
load_dotenv()
# --- 🔐 Gemini API Key ---
api_key =os.getenv("GEMINI_API_KEY")  # Replace with your actual key

# --- 🤖 Load Gemini LLM and Embeddings ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

# --- 📄 Load and Process DOCX File ---
loader = Docx2txtLoader("book.docx")  # Make sure book.docx exists
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(pages)

# --- 🔍 Create Vector Store and Retriever ---
vectorstore = FAISS.from_documents(docs, embedding)
retriever = vectorstore.as_retriever()

# --- 🤝 Retrieval QA Chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# --- 🧠 LangGraph State Definition ---
class GraphState(TypedDict):
    question: str
    result: str

# --- 🧩 LangGraph Node ---
def rag_node(state: GraphState) -> GraphState:
    question = state["question"]
    answer = qa_chain.run(question)
    return {
        "question": question,
        "result": answer
    }

# --- 💾 Setup Persistent LangGraph Workflow ---




workflow = StateGraph(GraphState)
workflow.add_node("RAG", rag_node)
workflow.set_entry_point("RAG")
workflow.set_finish_point("RAG")

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)


# --- 💬 Chainlit Chat Handlers ---
@cl.on_chat_start
async def on_chat_start():
    await cl.Message("📘 Ask me anything from the *book.docx*!").send()

@cl.on_message
async def on_message(msg: cl.Message):
    # Create a unique session ID for tracking
    thread_id = getattr(cl.user_session, 'id', None) or getattr(cl.user_session, 'user_id', None) or str(id(cl.user_session))
    config = {"configurable": {"thread_id": thread_id}}

    # 🔄 Run through LangGraph
    result = graph.invoke({"question": msg.content}, config=config)

    # Send result back to user
    await cl.Message(content=result["result"]).send()
