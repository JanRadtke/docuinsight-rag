"""
DocuInsight Ingest (v0.1)
=========================
PDF Ingestion: Text chunking + Image extraction + GPT-4o Vision analysis.
Stores everything in ChromaDB (local, persistent).

Usage:
    python src/ingest.py
"""

import os
import logging
import base64
import hashlib
import fitz  # PyMuPDF

logger = logging.getLogger("docuinsight")
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from llm_provider import get_llm_client, get_embedding_model, get_embedding_dim, get_embedding, LLM_ERRORS

from langdetect import detect as detect_language

_TEXT_ERRORS = (OSError, *LLM_ERRORS)
_IMAGE_ERRORS = (OSError, fitz.FileDataError, *LLM_ERRORS)

# --- CONFIGURATION ---
load_dotenv()

EMBEDDING_MODEL = get_embedding_model()
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o")
EMBEDDING_DIM = get_embedding_dim()

INPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input")
CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "chroma_db")
IMAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "images")

COLLECTION_NAME = "docuinsight"

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)

# Lazy-initialized vision client (created on first use, not at import time)
_vision_client = None


def _get_vision_client():
    global _vision_client
    if _vision_client is None:
        _vision_client = get_llm_client()
    return _vision_client


def encode_image(image_bytes: bytes) -> str:
    """Encodes image bytes as a Base64 string."""
    return base64.b64encode(image_bytes).decode('utf-8')


def analyze_image_with_gpt4o(image_bytes: bytes) -> str | None:
    """Sends the image to GPT-4o for a description."""
    base64_img = encode_image(image_bytes)
    try:
        response = _get_vision_client().chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this diagram/image in detail for a search database. Ignore logos, seals, or copyright notices. If it is NOT an informative diagram, respond only with 'SKIP'."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}", "detail": "low"}},
                    ],
                }
            ],
            max_tokens=600
        )
        return response.choices[0].message.content
    except LLM_ERRORS as e:
        logger.warning("Vision API Error: %s", e)
        return None


def detect_doc_language(pages) -> str:
    """Detects the primary language of a document from its first page(s).
    Returns ISO 639-1 code (e.g. 'en', 'de', 'zh-cn', 'he').
    Falls back to 'en' if detection fails."""
    if not pages:
        return "en"
    # Use first page, fall back to first 2 pages if first is too short
    sample_text = pages[0].page_content
    if len(sample_text) < 100 and len(pages) > 1:
        sample_text += " " + pages[1].page_content
    sample_text = sample_text[:2000]  # langdetect doesn't need more
    try:
        lang = detect_language(sample_text)
        return lang
    except Exception as e:
        logger.debug("Language detection failed, defaulting to 'en': %s", e)
        return "en"


def clean_id(text_id: str) -> str:
    """Sanitises a string for use as an ID."""
    return "".join(c if c.isalnum() else "_" for c in text_id)


def create_collection():
    """Creates (or recreates) the ChromaDB Collection."""
    logger.info("Creating ChromaDB Collection '%s'...", COLLECTION_NAME)

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete old collection if it exists
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        logger.info("Old Collection deleted.")
    except ValueError:
        pass

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    logger.info("Collection created.")
    return chroma_client, collection


def process_documents() -> None:
    """Processes all PDFs in the input/ folder and stores them in ChromaDB."""
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)

    if not os.path.exists(INPUT_DIR):
        logger.error("Input folder not found: %s", INPUT_DIR)
        logger.error("Please create the folder and add PDFs to it.")
        return

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")]

    if not files:
        logger.error("No PDFs found in the input/ folder.")
        logger.error("Please place PDF files in: %s", INPUT_DIR)
        return

    logger.info("%d PDF(s) found in input/", len(files))

    # Prepare ChromaDB
    chroma_client, collection = create_collection()

    # Image deduplication (local to this run)
    seen_image_hashes = set()

    # Batch lists for ChromaDB
    all_ids = []
    all_documents = []
    all_embeddings = []
    all_metadatas = []

    for filename in files:
        file_path = os.path.join(INPUT_DIR, filename)
        logger.info("Processing: %s...", filename)

        # A) TEXT
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()

            # Language Detection (from first page)
            doc_language = detect_doc_language(pages)
            logger.info("Language detected: %s", doc_language)

            for i, page in enumerate(pages):
                page_num = i + 1
                parent_id = clean_id(f"{filename}_page_{page_num}")

                # Parent Doc (full page — used for Parent Expansion)
                all_ids.append(parent_id)
                all_documents.append(page.page_content)
                all_embeddings.append(get_embedding(page.page_content))
                all_metadatas.append({
                    "source_file": filename,
                    "page_number": page_num,
                    "type": "parent",
                    "parent_id": "root",
                    "image_path": "",
                    "language": doc_language
                })

                # Chunks (with Embedding)
                chunks = text_splitter.split_text(page.page_content)
                for c_idx, chunk in enumerate(chunks):
                    chunk_id = f"{parent_id}_chunk_{c_idx}"
                    all_ids.append(chunk_id)
                    all_documents.append(chunk)
                    all_embeddings.append(get_embedding(chunk))
                    all_metadatas.append({
                        "source_file": filename,
                        "page_number": page_num,
                        "type": "chunk",
                        "parent_id": parent_id,
                        "image_path": "",
                        "language": doc_language
                    })
            logger.info("Text OK (%d pages).", len(pages))
        except _TEXT_ERRORS as e:
            logger.error("Text Error: %s", e)

        # B) IMAGES
        try:
            doc = fitz.open(file_path)

            for page_idx in range(len(doc)):
                page = doc[page_idx]
                image_list = page.get_images(full=True)

                for img_idx, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    w, h = base_image["width"], base_image["height"]

                    if w < 350 or h < 350:
                        continue

                    # Deduplication via Hash
                    img_hash = hashlib.sha256(image_bytes).hexdigest()

                    if img_hash in seen_image_hashes:
                        continue

                    seen_image_hashes.add(img_hash)

                    logger.info("Image found (P.%d, %dx%d). Analysing...", page_idx+1, w, h)

                    # 1. Save
                    img_id = clean_id(f"{filename}_p{page_idx+1}_img{img_idx}")
                    local_path = os.path.join(IMAGE_DIR, img_id + ".jpg")
                    with open(local_path, "wb") as f:
                        f.write(image_bytes)

                    # 2. GPT Vision
                    description = analyze_image_with_gpt4o(image_bytes)

                    # 3. Check for garbage responses or SKIP
                    if not description or "SKIP" in description or "sorry" in description.lower() or "tut mir leid" in description.lower():
                        logger.info("Image ignored (No informative content).")
                        continue

                    # 4. Indexing
                    content = f"IMAGE-DIAGRAM (P.{page_idx+1}): {description}"
                    all_ids.append(img_id)
                    all_documents.append(content)
                    all_embeddings.append(get_embedding(description))
                    all_metadatas.append({
                        "source_file": filename,
                        "page_number": page_idx + 1,
                        "type": "image",
                        "parent_id": clean_id(f"{filename}_page_{page_idx+1}"),
                        "image_path": local_path,
                        "language": doc_language
                    })
                    logger.info("Image indexed!")

        except _IMAGE_ERRORS as e:
            logger.error("Image Error: %s", e)

    # UPLOAD to ChromaDB
    if all_ids:
        logger.info("Saving %d documents to ChromaDB...", len(all_ids))

        # ChromaDB has a batch limit, so upload in chunks
        batch_size = 500
        for i in range(0, len(all_ids), batch_size):
            end = min(i + batch_size, len(all_ids))
            try:
                collection.add(
                    ids=all_ids[i:end],
                    documents=all_documents[i:end],
                    embeddings=all_embeddings[i:end],
                    metadatas=all_metadatas[i:end]
                )
                logger.info("Batch %d OK (%d documents).", i//batch_size + 1, end - i)
            except Exception as e:
                logger.error("Batch Error: %s", e)

    logger.info("Ingestion complete! Saved %d documents to ChromaDB.", len(all_ids))
    logger.info("Persistence path: %s", CHROMA_DIR)


if __name__ == "__main__":
    process_documents()
