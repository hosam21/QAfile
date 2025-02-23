import pysqlite3
import sys
sys.modules["sqlite3"] = pysqlite3
import streamlit as st
import tempfile
import os
import uuid
import base64
from io import BytesIO
from PIL import Image
from chromadb.config import Settings
from unstructured.partition.pdf import partition_pdf
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers.models.blip import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.stores import BaseStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
load_dotenv()  # This will load variables from the .env file into os.environ
# --- Custom In-Memory DocStore ---
class CustomInMemoryDocStore(BaseStore[str, Document]):
    def __init__(self):
        self._store = {}
    def mset(self, key_value_pairs):
        for key, value in key_value_pairs:
            self._store[key] = value
    def mget(self, keys):
        return [self._store.get(key) for key in keys]
    def mdelete(self, keys):
        for key in keys:
            self._store.pop(key, None)
    def yield_keys(self, prefix=None):
        for key in self._store:
            if prefix is None or key.startswith(prefix):
                yield key

# --- Open Source Embeddings using SentenceTransformer ---
class OpenSourceEmbeddings:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, docs):
        processed_docs = [doc if (isinstance(doc, str) and doc.strip()) else "unknown" for doc in docs]
        return self.model.encode(processed_docs, convert_to_numpy=True).tolist()
    def embed_query(self, query: str):
        query = query if (isinstance(query, str) and query.strip()) else "unknown"
        return self.model.encode([query], convert_to_numpy=True).tolist()[0]

# --- Helper function to extract images in base64 ---
def get_images_base64(chunks):
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            for el in chunk.metadata.orig_elements:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64

# --- Helper to index documents ---
def index_documents(summaries, originals, doc_type, vectorstore, docstore, id_key="doc_id"):
    doc_ids = []
    documents = []
    for summary, original in zip(summaries, originals):
        if summary and summary.strip():
            doc_id = str(uuid.uuid4())
            doc_ids.append(doc_id)
            documents.append(Document(page_content=summary, metadata={id_key: doc_id, "type": doc_type}))
    if documents:
        vectorstore.add_documents(documents)
        for doc_id, original in zip(doc_ids, originals):
            content = original if (isinstance(original, str) and original.strip()) else "unknown"
            docstore.mset([(doc_id, Document(page_content=content, metadata={"type": doc_type}))])
    else:
        st.write(f"No valid {doc_type} documents to add.")

# --- Main function to process the PDF and build the QA pipeline ---
def process_pdf(pdf_path):
    # Partition PDF into chunks
    chunks = partition_pdf(
        filename=pdf_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )
    
    # Separate texts and tables
    texts = []
    tables = []
    for chunk in chunks:
        if "Table" in str(type(chunk)):
            tables.append(chunk)
        if "CompositeElement" in str(type(chunk)):
            texts.append(chunk)
    
    # Extract images
    images = get_images_base64(chunks)
    
    # --- Summarization for texts and tables using Groq ---
    prompt_text = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.

    Respond only with the summary, no additional comment.
    Just give the summary as it is.

    Table or text chunk: {element}
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model_groq = ChatGroq(
    groq_api_key=os.environ["GROQ_API_KEY"],  # Access from environment
    temperature=0.5,
    model="llama-3.1-8b-instant"
    )
    summarize_chain = {"element": lambda x: x} | prompt | model_groq | StrOutputParser()
    
    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})
    tables_html = [table.metadata.text_as_html for table in tables]
    table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 3})
    
    # --- Summarization for images using BLIP ---
    processor_blip = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    def generate_description_from_base64(image_base64):
        image_data = base64.b64decode(image_base64)
        image_buffer = BytesIO(image_data)
        image = Image.open(image_buffer)
        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs = processor_blip(images=image, return_tensors="pt")
        outputs = model_blip.generate(**inputs)
        description = processor_blip.decode(outputs[0], skip_special_tokens=True)
        return description
    image_summaries = []
    for image_b64 in images:
        try:
            description = generate_description_from_base64(image_b64)
            image_summaries.append(description)
        except Exception as e:
            st.write(f"Error processing an image: {e}")
    
    # Prepare original content for indexing
    text_originals = [str(t) for t in texts]
    table_originals = [str(t) for t in tables]
    image_originals = [str(i) for i in image_summaries]
    
    # --- Build vectorstore and document store ---
    embedding_function = OpenSourceEmbeddings()
    client_settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory="chroma_db")
    vectorstore = Chroma(
    collection_name="multi_modal_rag",
    embedding_function=embedding_function,
    client_settings=client_settings
    )

    
    # Index documents
    index_documents(text_summaries, text_originals, "text", vectorstore, docstore, id_key)
    index_documents(table_summaries, table_originals, "table", vectorstore, docstore, id_key)
    index_documents(image_summaries, image_originals, "image", vectorstore, docstore, id_key)
    
    # --- Create a MultiVectorRetriever ---
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=id_key,
        search_kwargs={"k": 5}
    )
    
    # --- Create the RetrievalQA (RAG) pipeline ---
    llm_for_qa = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),  # Add this line
    temperature=0.5,
    model="llama-3.1-8b-instant"
    )
    rag_pipeline = RetrievalQA.from_chain_type(
        llm=llm_for_qa,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return rag_pipeline

# --- Streamlit App UI ---
st.title("PDF QA App")

# Use session state to store the QA pipeline between interactions
if 'qa_pipeline' not in st.session_state:
    st.session_state.qa_pipeline = None

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_pdf_path = tmp_file.name
    st.write("Processing PDF...")
    st.session_state.qa_pipeline = process_pdf(tmp_pdf_path)
    st.success("PDF processed and indexed successfully!")

query_text = st.text_input("Enter your query:")
if st.button("Ask"):
    if st.session_state.qa_pipeline is None:
        st.error("Please upload and process a PDF first.")
    else:
        result = st.session_state.qa_pipeline(query_text)
        st.subheader("Answer:")
        st.write(result['result'])
        st.subheader("Source Documents:")
        for doc in result.get('source_documents', []):
            st.write(doc.metadata)
            st.write(doc.page_content)
