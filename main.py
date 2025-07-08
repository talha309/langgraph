import chainlit as cl
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict
from dotenv import load_dotenv
import os 
load_dotenv()
# --- Gemini API Key ---
api_key = os.getenv("GEMINI_API_KEY")  # 🔁 Replace with your actual key

# --- Load Gemini LLM and Embedding ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

# --- Load and Process PDF ---
loader = Docx2txtLoader("book.docx")  # 📘 Make sure this file exists
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(pages)

# --- Create Vector Store and Retriever ---
vectorstore = FAISS.from_documents(docs, embedding)
retriever = vectorstore.as_retriever()

# --- Create RAG Chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# --- LangGraph State Definition ---
class GraphState(TypedDict):
    question: str
    result: str

# --- LangGraph Node Function ---
def rag_node(state: GraphState) -> GraphState:
    question = state["question"]
    answer = qa_chain.run(question)
    return {
        "question": question,
        "result": answer
    }

# --- Build LangGraph with persistent history ---
workflow = StateGraph(GraphState)
workflow.add_node("RAG", rag_node)
workflow.set_entry_point("RAG")
workflow.set_finish_point("RAG")
memory = MemorySaver
graph = workflow.compile(checkpointer=memory)

# --- Chainlit Integration ---
@cl.on_chat_start
async def on_chat_start():
    await cl.Message("📘 Ask me anything from the *Python for Everybody* book!").send()

@cl.on_message
async def on_message(msg: cl.Message):
    # Try to get a unique session id for thread_id
    thread_id = getattr(cl.user_session, 'id', None) or getattr(cl.user_session, 'user_id', None) or str(id(cl.user_session))
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"question": msg.content}, config=config)
    await cl.Message(content=result["result"]).send()
