import os
import base64
import uuid
import re
import io
import requests
import time
import torch
import shutil
from dotenv import load_dotenv
from PIL import Image
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.retrievers import MultiVectorRetriever
from langchain_core.stores import InMemoryStore
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from unstructured.partition.pdf import partition_pdf
from langsmith import Client
from src.logger import logger

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Tracing Multimodal info retrieval from PDF"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
#genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize Langsmith client
client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])

# Setup models
google_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Constants
OUTPUT_PATH = os.path.abspath("extracted_data")
VECTORSTORE_DB_PATH = os.path.abspath("mm_rag_vectorstore_db")


def plt_img_base64(img):
    """Convert an image to a base64 encoded string."""
    buffered = io.BytesIO()
    
    # Check if img is a base64 string
    if isinstance(img, str) and looks_like_base64(img):
        img_data = base64.b64decode(img)
        img = Image.open(io.BytesIO(img_data))  # Convert to PIL Image
    
    # Now img should be a PIL Image
    img.save(buffered, format="JPEG")  # Use "JPEG" instead of "JPG" for compatibility
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_base64

def looks_like_base64(sb):
    """Check if the string looks like base64."""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None

def is_image_data(b64data):
    """Check if the base64 data is an image."""
    image_signatures = {
        b"\xFF\xD8\xFF": "jpg",
        b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]
        return any(header.startswith(sig) for sig in image_signatures.keys())
    except Exception:
        return False

def resize_base64_image(base64_string, size=(128, 128)):
    """Resize an image encoded as a Base64 string."""
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    resized_img = img.resize(size, Image.LANCZOS)
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def split_image_text_types(docs):
    """Split base64-encoded images and texts."""
    b64_images = []
    texts = []
    for doc in docs:
        if isinstance(doc, Document):
            doc = doc.page_content

        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(1300, 600))
            b64_images.append(doc)
        elif isinstance(doc, str):
            texts.append(doc)
        else:
            raise ValueError("Unsupported document type")
    return {"images": b64_images, "texts": texts}

def img_prompt_func(data_dict):
    """Join context into a single string for the prompt."""
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = [{"type": "text",
                 "text": f"You are a helpful assistant.\n\
                           You will be given mixed text, tables, and image(s).\n\
                           If you don't know the answer, say that you don't know.\n\
                           You keep the answer concise.\n\
                           User-provided question: {data_dict['question']}\n\
                           Text and/or tables:\n{formatted_texts}"}]

    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            messages.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}})

    return [HumanMessage(content=messages)]

# PDF Processing Functions
def doc_partition(full_path, out_path):
    """Partition the PDF document into text, tables, and images."""
    return partition_pdf(
        filename=full_path,
        strategy="hi_res",
        extract_images_in_pdf=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=False,
        extract_image_block_output_dir=out_path
    )

def get_pdf_elements(uploaded_pdf):
    """Extract elements from the uploaded PDF."""
    logger.info("=========Uploaded pdf file info ========", uploaded_pdf)
    original_file_name = uploaded_pdf.name
    logger.info("=========The actual path========", original_file_name)

    try:
        if os.path.exists(OUTPUT_PATH):
            logger.info(f"Removing existing directory: {OUTPUT_PATH}")
            os.system(f"rm -rf {OUTPUT_PATH}")

        if uploaded_pdf.name.lower().endswith('.pdf'):
            if uploaded_pdf is not None:
                file_bytes = uploaded_pdf.read()
                with open(original_file_name, "wb") as f:
                    f.write(file_bytes)

                raw_pdf_elements = doc_partition(original_file_name, OUTPUT_PATH)
                logger.info('PDF elements extracted')

                os.remove(original_file_name)
                return raw_pdf_elements

    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return []

def get_elements_list(raw_pdf_elements):
    """Categorize raw PDF elements into narrative text and tables."""
    NarrativeText = []
    Table = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.NarrativeText" in str(type(element)):
            NarrativeText.append(str(element))
        elif "unstructured.documents.elements.Table" in str(type(element)):
            Table.append(str(element))

    return NarrativeText, Table

def get_table_summary(Table):
    """Summarize tables for retrieval."""
    prompt_table = """You are an assistant tasked with summarizing tables for retrieval. \
    These summaries will be embedded and used to retrieve the raw table elements. \
    Give a concise summary of the table that is well optimized for retrieval. Table:{element}"""
    prompt = ChatPromptTemplate.from_template(prompt_table)
    model = ChatGoogleGenerativeAI(temperature=0, model="gemini-1.5-flash", max_tokens=1024, device=device)
    
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    table_summaries = summarize_chain.batch(Table, {"max_concurrency": len(Table)})
    return table_summaries

def get_text_summary(NarrativeText):
    """Summarize narrative text for retrieval."""
    prompt_text = """You are an assistant tasked with summarizing text for retrieval. \
    These summaries will be embedded and used to retrieve the raw text elements. \
    Give a concise summary of the text that is well optimized for retrieval. NarrativeText: {element}"""
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatGoogleGenerativeAI(temperature=0, model="gemini-1.5-flash", max_tokens=512, device=device)

    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    def process_in_batches(elements, batch_size=10, delay=15):
        summaries = []
        for i in range(0, len(elements), batch_size):
            batch = elements[i:i + batch_size]
            try:
                batch_summaries = summarize_chain.batch(batch, {"max_concurrency": len(batch)})
                summaries.extend(batch_summaries)
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
            time.sleep(delay)
        return summaries
    
    return process_in_batches(NarrativeText, batch_size=5, delay=50)

def get_image_summary():
    """Generate summaries for images."""
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def image_summarize(img_base64, prompt):
        chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash", max_tokens=1024, device=device)
        msg = chat.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                    ]
                )
            ]
        )
        return msg.content
    
    def generate_img_summaries(path):
        img_base64_list = []
        image_summaries = []
        prompt = """You are an assistant tasked with summarizing images for retrieval. \
        These summaries will be embedded and used to retrieve the raw image. \
        Give a concise summary of the image that is well optimized for retrieval."""
        
        for img_file in sorted(os.listdir(path)):
            if img_file.endswith(".jpg"):
                img_path = os.path.join(path, img_file)
                base64_image = encode_image(img_path)
                img_base64_list.append(base64_image)
                image_summaries.append(image_summarize(base64_image, prompt))
        return img_base64_list, image_summaries

    return generate_img_summaries(OUTPUT_PATH)


def create_vectorstore():
    """Create a new vector store for storing embeddings."""
    try:
        if os.path.exists(VECTORSTORE_DB_PATH):
            shutil.rmtree(VECTORSTORE_DB_PATH)  # Delete the old vector store
            print("Deleted existing vector store.")
        
        vectorstore = Chroma(
            collection_name="mm_rag",
            embedding_function=google_embeddings,
            persist_directory=VECTORSTORE_DB_PATH
        )        
        vectorstore.persist()  # Save the SQLite database
        logger.info("New vector store created successfully.")
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")

def load_vectorstore():
    """Load the existing vector store."""
    try:
        if not os.path.exists(VECTORSTORE_DB_PATH):
            logger.error("Vector store does not exist.")
            return None
        vectorstore = Chroma(
            embedding_function=google_embeddings,
            persist_directory=VECTORSTORE_DB_PATH
        )
        logger.info("Loaded existing vector store successfully.")
        return vectorstore
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        return None

def create_multi_vector_retriever(vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images):
    """Create a multi-vector retriever for different document types."""
    store = InMemoryStore()
    id_key = "doc_id"
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)

    def add_documents(retriever, doc_summaries, original_doc_contents, content_type):
        doc_ids = [str(uuid.uuid4()) for _ in original_doc_contents]
        summary_docs = [
            Document(
                page_content=s,
                metadata={
                    id_key: doc_ids[i],
                    "type": content_type,
                    "original_content": original_doc_contents[i]
                }
            )
            for i, s in enumerate(doc_summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, original_doc_contents)))

    if text_summaries:
        add_documents(retriever, text_summaries, texts, "text")
    if table_summaries:
        add_documents(retriever, table_summaries, tables, "table")
    if image_summaries:
        add_documents(retriever, image_summaries, images, "image")

    return retriever

def multi_modal_rag_chain(retriever):
    """Create a multi-modal RAG chain."""
    model = ChatGoogleGenerativeAI(temperature=0, model="gemini-1.5-flash", max_tokens=1024, device=device)
    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )
    return chain   


def get_conversational_chain(retriever, loaded_vectorstore, question):
    """Get response from conversational chain based on question."""
    #memory = ConversationBufferMemory(memory_key = "chat_history", k=5, return_messages=True)
    
    relevant_docs = loaded_vectorstore.similarity_search(question)
    relevant_images = [d.metadata['original_content'] for d in relevant_docs if d.metadata['type'] == 'image']
    
    # Use the chain defined in multi_modal_rag_chain
    chain = multi_modal_rag_chain(retriever)
    result = chain.invoke(question)

    if relevant_images:
        return result, relevant_images[0]
    else:
        return result, None
