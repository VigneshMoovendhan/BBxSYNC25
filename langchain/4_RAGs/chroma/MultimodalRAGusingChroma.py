from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma  
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from pdf2image import convert_from_path
import io
import base64
from PIL import Image
import pytesseract
import pdfplumber
from tqdm import tqdm
import json
import logging
import dotenv
import os
 
# Load environment variables
dotenv.load_dotenv()
 
 
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
 
# Initialize LangChain components
try:
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=500)
except Exception as e:
    logger.error(f"Failed to initialize LangChain components: {e}")
    raise
 
def hybrid_extract_multimodal(pdf_path, ocr_images=False):
    """
    Extract text, tables, and images/charts from PDF using LangChain's PyPDFLoader.
    Returns list of processed chunks as LangChain Documents.
   
    Args:
        pdf_path (str): Path to the PDF file.
        ocr_images (bool): Whether to perform OCR on images (slower, for scanned PDFs).
   
    Returns:
        list: List of LangChain Documents (text, table, image).
    """
    processed_chunks = []
    page_num = 1
   
    try:
        # Extract images/charts per page with pdf2image
        images = convert_from_path(pdf_path)
        logger.info(f"Extracted {len(images)} images from PDF")
       
        # Use LangChain's PyPDFLoader for text
        loader = PyPDFLoader(pdf_path, extract_images=False)
        pages = loader.load()
        logger.info(f"Extracted {len(pages)} pages from PDF")
       
        for page in pages:
            # Text extraction
            text = page.page_content
            if text.strip():
                processed_chunks.append(Document(
                    page_content=text,
                    metadata={"type": "text", "page_number": page_num}
                ))
                logger.info(f"Extracted text from page {page_num}")
           
            # Tables (using pdfplumber for tables)
            with pdfplumber.open(pdf_path) as pdf:
                if len(pdf.pages) >= page_num:
                    pdf_page = pdf.pages[page_num - 1]
                    tables = pdf_page.extract_tables()
                    for table_idx, table in enumerate(tables or []):
                        table_text = "\n".join([" | ".join(row) for row in table if row])
                        processed_chunks.append(Document(
                            page_content=table_text,
                            metadata={"type": "table", "page_number": page_num, "table_index": table_idx, "raw_table": json.dumps(table)}
                        ))
                        logger.info(f"Extracted table {table_idx} from page {page_num}")
           
            # Images/Charts
            if images and len(images) >= page_num:
                img = images[page_num - 1]
                # Base64 encode
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
               
                # Optional OCR for text in image
                img_text = ""
                if ocr_images:
                    img_text = pytesseract.image_to_string(img)
                    logger.info(f"Performed OCR on image from page {page_num}")
               
                processed_chunks.append(Document(
                    page_content=img_text or "Image/Chart on page (OCR: none if disabled)",
                    metadata={
                        "type": "image",
                        "page_number": page_num,
                        "base64": img_base64,
                        "mime_type": "image/png"
                    }
                ))
                logger.info(f"Extracted image from page {page_num}")
           
            page_num += 1
   
    except Exception as e:
        logger.error(f"Error in hybrid_extract_multimodal: {e}")
        raise
   
    return processed_chunks
 
def describe_image(base64_data, mime_type):
    """
    Describe image/chart using LangChain's ChatOpenAI with GPT-4o.
   
    Args:
        base64_data (str): Base64-encoded image data.
        mime_type (str): MIME type of the image (e.g., 'image/png').
   
    Returns:
        str: Description of the image or chart.
    """
    try:
        image_url = f"data:{mime_type};base64,{base64_data}"
        message = HumanMessage(content=[
            {"type": "text", "text": "Describe this image or chart in detail: key data, trends, labels, insights. Concise."},
            {"type": "image_url", "image_url": {"url": image_url}}  # Fixed format
        ])
        response = llm.invoke([message])
        logger.info("Successfully described image")
        return response.content
    except Exception as e:
        logger.error(f"Error in describe_image: {e}")
        return f"Error describing image: {str(e)}"
 
def summarize_table(table_text, raw_table=None):
    """
    Summarize table content using LangChain's ChatOpenAI.
   
    Args:
        table_text (str): Text representation of the table.
        raw_table (list): Raw table data as list of lists (optional).
   
    Returns:
        str: Summary of the table.
    """
    try:
        if raw_table:
            # Convert to markdown string
            md_table = "| " + " | ".join(["Col" + str(i) for i in range(len(raw_table[0]))]) + " |\n"
            md_table += "| " + " | ".join(["---"] * len(raw_table[0])) + " |\n"
            for row in raw_table:
                md_table += "| " + " | ".join([str(cell or "") for cell in row]) + " |\n"
            table_str = md_table
        else:
            table_str = table_text
       
        prompt = ChatPromptTemplate.from_template(
            """
            Summarize this table: key rows/columns, totals, trends. Include markdown if helpful.
            Table:
            {table_str}
            Summary:
            """
        )
        chain = prompt | llm
        response = chain.invoke({"table_str": table_str})
        logger.info("Successfully summarized table")
        return response.content
    except Exception as e:
        logger.error(f"Error in summarize_table: {e}")
        return f"Error summarizing table: {str(e)}"
 
def process_chunks(chunks):
    """
    Enhance chunks by summarizing tables and describing images.
   
    Args:
        chunks (list): List of LangChain Documents (text, table, image).
   
    Returns:
        list: Enhanced LangChain Documents with updated content and metadata.
    """
    enhanced_chunks = []
    for chunk in chunks:
        try:
            typ = chunk.metadata["type"]
            meta = chunk.metadata.copy()
            if typ == "text":
                enhanced_chunks.append(Document(page_content=chunk.page_content, metadata=meta))
            elif typ == "table":
                raw_table = json.loads(meta.get("raw_table", "[]"))
                summary = summarize_table(chunk.page_content, raw_table)
                if raw_table:
                    # Build markdown table
                    md_table = "| " + " | ".join(["Col" + str(i) for i in range(len(raw_table[0]))]) + " |\n"
                    md_table += "| " + " | ".join(["---"] * len(raw_table[0])) + " |\n"
                    for row in raw_table:
                        md_table += "| " + " | ".join([str(cell or "") for cell in row]) + " |\n"
                    meta["html"] = md_table
                    meta["raw_table_json"] = json.dumps(raw_table)
                else:
                    meta["html"] = chunk.page_content
                if "raw_table" in meta:
                    del meta["raw_table"]
                enhanced_chunks.append(Document(page_content=summary, metadata=meta))
            elif typ == "image":
                base64_data = meta["base64"]
                summary = describe_image(base64_data, meta["mime_type"])
                enhanced_chunks.append(Document(page_content=summary, metadata=meta))
            logger.info(f"Processed chunk of type {typ}")
        except Exception as e:
            logger.error(f"Error processing chunk of type {typ}: {e}")
            continue
   
    return enhanced_chunks
 
def store_chunks(enhanced_chunks, collection):
    """
    Store enhanced chunks in a Chroma vector store using LangChain.
   
    Args:
        enhanced_chunks (list): List of enhanced LangChain Documents.
        collection: LangChain Chroma vector store.
    """
    try:
        for i, chunk in enumerate(tqdm(enhanced_chunks, desc="Storing")):
            collection.add_documents(
                documents=[chunk],
                ids=[f"doc-{i}"]
            )
        logger.info(f"Stored {len(enhanced_chunks)} chunks in vector store")
    except Exception as e:
        logger.error(f"Error storing chunks: {e}")
        raise
 
def multimodal_rag_query(query, collection, n_results=5):
    """
    Retrieve relevant multimodal chunks and generate a response with LangChain's ChatOpenAI.
   
    Args:
        query (str): User query.
        collection: LangChain Chroma vector store.
        n_results (int): Number of results to retrieve.
   
    Returns:
        str: Generated answer based on retrieved chunks.
    """
    try:
        # Step 1: Retrieve top chunks
        results = collection.similarity_search_with_score(query, k=n_results)
        docs = [doc for doc, _ in results]
        logger.info(f"Retrieved {len(docs)} documents for query")
       
        # Step 2: Build multimodal context
        context_parts = []
        image_uris = []
       
        for doc in docs:
            typ = doc.metadata["type"]
            page = doc.metadata.get("page_number", "Unknown")
           
            if typ == "text":
                context_parts.append(f"Text (p.{page}): {doc.page_content}")
            elif typ == "table":
                context_parts.append(f"Table Summary (p.{page}): {doc.page_content}")
                html_or_md = doc.metadata.get("html", "")
                if html_or_md:
                    context_parts.append(f"Table Structure (p.{page}): {html_or_md}")
            elif typ == "image":
                context_parts.append(f"Image Description (p.{page}): {doc.page_content}")
                base64_data = doc.metadata.get("base64")
                mime = doc.metadata.get("mime_type", "image/png")
                if base64_data:
                    image_uris.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64_data}"}})  # Fixed format
            logger.info(f"Processed document of type {typ} from page {page}")
       
        # Combine context
        context = "\n\n".join(context_parts)
       
        # Step 3: Craft the prompt
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the query using ONLY the provided context (text, tables, images).
            Describe visuals, analyze tables/charts if relevant. Be factual and concise.
            If info is missing, say "Not found in document."
           
            Context:
            {context}
           
            Query: {query}
           
            Answer:
            """
        )
       
        # Step 4: Create multimodal message
        message_content = [
            {"type": "text", "text": prompt.format(context=context, query=query)}
        ] + image_uris
       
        message = HumanMessage(content=message_content)
       
        # Step 5: Generate response
        response = llm.invoke([message])
        logger.info("Successfully generated response")
        return response.content
   
    except Exception as e:
        logger.error(f"Error in multimodal_rag_query: {e}")
        return f"Error processing query: {str(e)}"
 
# Example usage
if __name__ == "__main__":
    pdf_path = r"D:\AIonLogic\data\multimodal_test.pdf"  # Windows path
   
    try:
        # Extract chunks
        processed_chunks = hybrid_extract_multimodal(pdf_path, ocr_images=False)
        print(f"Extracted {len(processed_chunks)} chunks: {set(c.metadata['type'] for c in processed_chunks)}")
        for chunk in processed_chunks[:2]:
            print(f"Type: {chunk.metadata['type']}, Summary preview: {chunk.page_content[:100]}...")
       
        # Enhance chunks
        enhanced_chunks = process_chunks(processed_chunks)
        print(f"Enhanced {len(enhanced_chunks)} chunks.")
       
        # Initialize Chroma vector store
        collection = Chroma(collection_name="pdf_chunks", embedding_function=embedding_model)
       
        # Store chunks
        store_chunks(enhanced_chunks, collection)
        print(f"Stored {collection._collection.count()} vectors.")
       
        # Query
        query = "Summarize any tables or describe charts in the document."
        answer = multimodal_rag_query(query, collection)
        print("Generated Answer:")
        print(answer)
   
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        print(f"Error: {str(e)}")