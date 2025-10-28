import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

load_dotenv()

# Check if OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file.")

# Function to load multiple documents from a list of file paths
def load_documents_from_files(file_paths):
    all_documents = []
    for file_path in file_paths:
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            # Add source metadata to each document
            for doc in docs:
                doc.metadata["source"] = os.path.basename(file_path)  # Just filename for brevity
            all_documents.extend(docs)
            print(f"Loaded {len(docs)} document(s) from {file_path}.")
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
    return all_documents

# List of file paths (e.g., for 2-3 files)
data_dir = r"D:\AIonLogic\data"
file_paths = [
    os.path.join(data_dir, "au.txt"),  # Replace with your actual file names
    os.path.join(data_dir, "bb.txt"),
    # os.path.join(data_dir, "file3.txt"),  # Uncomment for a third file
]

# Load all documents
documents = load_documents_from_files(file_paths)

if not documents:
    print("No documents loaded. Exiting.")
    exit()

# Split the documents into chunks (preserves metadata)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(documents)
print(f"Split into {len(split_docs)} chunks across {len(file_paths)} files.")

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(split_docs, embeddings)

# Create a retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 relevant docs

# Define the prompt template (instructs model to cite sources)
prompt_template = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Use the following context to answer the question. 
    Each piece of context includes a source file—cite the source(s) in your answer using the format [Source: filename.txt].

Context: {context}

Question: {question}

Answer:"""
)

# Initialize the chat model
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Updated format_docs to include source metadata
def format_docs(docs):
    formatted = []
    for doc in docs:
        content = doc.page_content
        source = doc.metadata.get("source", "Unknown")
        formatted.append(f"{content}\n\n[Source: {source}]")
    return "\n\n".join(formatted)

# LCEL chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | model
    | StrOutputParser()
)

# Function to invoke chain and print fetched sources
def query_with_sources(question):
    # Get raw retrieved docs to print sources
    retrieved_docs = retriever.invoke(question)
    sources = list(set(doc.metadata.get("source", "Unknown") for doc in retrieved_docs))  # Unique sources
    print(f"\nFetched sources for '{question}': {', '.join(sources)}")
    
    # Invoke the chain for the response
    response = rag_chain.invoke(question)
    return response

# Example usage
if __name__ == "__main__":
    question = "What does bonbloc technologies do?"
    response = query_with_sources(question)
    print("Question:", question)
    print("Answer:", response)
    
    # Another example
    question = "Where is Anna University Located?"
    response = query_with_sources(question)
    print("\nQuestion:", question)
    print("Answer:", response)