
import os
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
    
# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------

documents_path = "./documents/lord_of_the_rings.txt"
persistent_directory = "./chroma_db"

# ---------------------------------------------------------------------
# Step 1: Load Documents
# ---------------------------------------------------------------------
if not os.path.exists(documents_path):
    raise FileNotFoundError(f"Document not found at {documents_path}")

loader = TextLoader(documents_path)
docs = loader.load()

# ---------------------------------------------------------------------
# Step 2: Split Documents into Chunks
# ---------------------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", "!", "?", " "],
    keep_separator=False
)
split_docs = text_splitter.split_documents(docs)

print(f"\n--- Document Chunks ---")
print(f"Number of chunks: {len(split_docs)}")
print(f"Sample chunk:\n{split_docs[0].page_content[:600]}\n")

# ---------------------------------------------------------------------
# Step 3: Create or Load Vector Store
# ---------------------------------------------------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")  # ✅ Better model

if os.path.exists(persistent_directory):
    print("Vector store already exists. Loading existing Chroma DB...\n")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
else:
    print("Persistent directory does not exist. Creating new Chroma DB...\n")
    db = Chroma.from_documents(split_docs, embeddings, persist_directory=persistent_directory)
    db.persist()
    print("--- Vector store created and persisted ---\n")

# ---------------------------------------------------------------------
# Step 4: Create Retriever and LLM
# ---------------------------------------------------------------------
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.1, "k": 5}  # ✅ Slightly looser threshold
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# ---------------------------------------------------------------------
# Step 5: Create RAG Chain
# ---------------------------------------------------------------------
prompt_template = """You are a helpful assistant that answers questions using the provided context.
If the answer is not in the context, just say "I couldn't find relevant information in the document."

Context:
{context}

Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

# ---------------------------------------------------------------------
# Step 6: Run Queries
# ---------------------------------------------------------------------
queries = [
    "Where does Gandalf meet Frodo?",
    "Who is the Ring-bearer?",
    "What is One Ring?",
]

for q in queries:
    print(f"\n--- Querying: {q} ---")
    result = rag_chain.invoke({"query": q})
    print(result["result"])
