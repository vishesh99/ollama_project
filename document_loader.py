from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
import os
from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path
import pytesseract
from PyPDF2 import PdfReader
import re

TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def load_documents_into_database(model_name: str, documents_path: str) -> FAISS:
    """
    Loads documents from the specified directory into the FAISS database
    after splitting the text into chunks.

    Returns:
        FAISS: The FAISS database with loaded documents.
    """

    print("Loading documents")
    raw_documents = load_documents(documents_path)
    if not raw_documents:
        raise ValueError("No documents loaded. Please check the directory path and document types.")

    print(f"Loaded {len(raw_documents)} raw documents")

    documents = TEXT_SPLITTER.split_documents(raw_documents)
    if not documents:
        raise ValueError("No documents after splitting. Please check the document content and splitting logic.")

    print(f"Split into {len(documents)} chunks")

    print("Creating embeddings and loading documents into FAISS")
    embeddings = OllamaEmbeddings(model=model_name)
    db = FAISS.from_documents(documents, embeddings)
    return db

def load_documents(path: str) -> List[Document]:
    """
    Loads documents from the specified directory path.

    This function supports loading of PDF, Markdown, and HTML documents by utilizing
    different loaders for each file type. It checks if the provided path exists and
    raises a FileNotFoundError if it does not. It then iterates over the supported
    file types and uses the corresponding loader to load the documents into a list.

    Args:
        path (str): The path to the directory containing documents to load.

    Returns:
        List[Document]: A list of loaded documents.

    Raises:
        FileNotFoundError: If the specified path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path does not exist: {path}")

    loaders = {
        ".pdf": DirectoryLoader(
            path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True,
        ),
        ".md": DirectoryLoader(
            path,
            glob="**/*.md",
            loader_cls=TextLoader,
            show_progress=True,
        ),
    }

    docs = []
    for file_type, loader in loaders.items():
        print(f"Loading {file_type} files")
        loaded_docs = loader.load()
        print(f"Loaded {len(loaded_docs)} {file_type} files")
        docs.extend(loaded_docs)

    # Use PyPDF2 for text extraction
    pdf_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.pdf')]
    for pdf_file in pdf_files:
        docs.extend(extract_text_from_pdf(pdf_file))

    print(f"Total documents loaded: {len(docs)}")
    return docs

def extract_text_from_pdf(pdf_path: str) -> List[Document]:
    """
    Extracts text from a PDF file using PyPDF2.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        List[Document]: A list of documents with extracted text.
    """
    documents = []
    reader = PdfReader(pdf_path)
    for page_number, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            cleaned_text = clean_text(text)
            documents.append(Document(page_content=cleaned_text, metadata={"source": pdf_path, "page": page_number}))
    return documents

def extract_text_and_tables_from_pdf(pdf_path: str) -> List[Document]:
    """
    Extracts text and tables from a PDF file using OCR and PyPDF2.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        List[Document]: A list of documents with extracted text and tables.
    """
    documents = extract_text_from_pdf(pdf_path)
    if not documents:
        documents = extract_text_from_images_in_pdf(pdf_path)
    return documents

def extract_text_from_images_in_pdf(pdf_path: str) -> List[Document]:
    """
    Extracts text from images within a PDF file using OCR.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        List[Document]: A list of documents with extracted text from images.
    """
    documents = []
    images = convert_from_path(pdf_path)
    for page_number, image in enumerate(images):
        text = pytesseract.image_to_string(image)
        if text.strip():  # Only add if text is not empty
            cleaned_text = clean_text(text)
            documents.append(Document(page_content=cleaned_text, metadata={"source": pdf_path, "page": page_number}))
    return documents

def clean_text(text: str) -> str:
    """
    Cleans the extracted text by keeping only English characters and removing extra spaces.

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    # Keep only English letters, numbers, and common punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text
