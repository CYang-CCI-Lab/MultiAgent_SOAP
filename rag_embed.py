import os
import json
import pickle
import logging

import torch
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
from transformers import AutoTokenizer, AutoModel

from typing import List, Dict, Any

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings

# ------------------
# Logging Setup
# ------------------
logger = logging.getLogger(__name__)

# ------------------
# Environment Setup
# ------------------
def setup_environment() -> None:
    """
    Loads environment variables and configures CUDA usage.
    """
    env_path = find_dotenv()
    if not env_path:
        env_path = "/home/yl3427/.env"  # fallback path
    if not load_dotenv(env_path):
        raise Exception("Failed to load .env file")

    # Adjust GPU environment as needed
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    logger.info("Environment setup complete.")

# --------------------
# Data Loading
# --------------------
def load_cases(json_path: str) -> Dict[str, Any]:
    """
    Loads case data from a JSON file, returning {hadm_id: {before_diagnosis, after_diagnosis}}
    """
    with open(json_path, "r") as f:
        cases = json.load(f)
    logger.info(f"Loaded {len(cases)} cases from {json_path}")
    return cases

# --------------------
# Text Splitting (Chunking)
# --------------------
def create_documents(
    cases: Dict[str, Any],
    tokenizer,
    max_length: int = 512
) -> List[Document]:
    """
    Splits clinical text into smaller chunks using a tokenizer-based splitter.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        separators=[
            "\n\n", "\n", r'(?<=[.?"\s])\s+', " ", ".", ","
        ],
        tokenizer=tokenizer,
        chunk_size=max_length,
        chunk_overlap=20,
        add_start_index=True,
        strip_whitespace=True,
        is_separator_regex=True
    )

    all_docs = []
    unique_texts = set()

    for hadm_id, data in cases.items():
        full_text = data["before_diagnosis"]
        docs = text_splitter.create_documents(
            texts=[full_text],
            metadatas=[{
                "hadm_id": hadm_id,
                "full_text": full_text,
                "diagnosis": data["after_diagnosis"]
            }]
        )
        # Deduplicate documents based on text content
        for d in docs:
            if d.page_content not in unique_texts:
                unique_texts.add(d.page_content)
                all_docs.append(d)

    logger.info(f"Created {len(all_docs)} total chunks (docs) from the original cases.")
    return all_docs

# --------------------
# Embedding in Chroma with Checkpointing (Batched Saving)
# --------------------
def embed_docs_in_chroma(
    docs: List[Document],
    embedding_model,
    collection,
    max_length: int = 512,
    checkpoint_file: str = "checkpoint_progress.json",
    checkpoint_batch_size: int = 10
) -> None:
    """
    Embeds documents into the Chroma collection with checkpointing.
    It saves progress to a checkpoint file every `checkpoint_batch_size` documents.
    """
    # Load checkpoint if it exists
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            completed_docs = set(json.load(f))
        logger.info(f"Resuming from checkpoint. {len(completed_docs)} docs already processed.")
    else:
        completed_docs = set()

    batch_counter = 0  # Counts docs processed since last save
    pbar = tqdm(total=len(docs), desc="Embedding Documents")
    for doc in docs:
        doc_text = doc.page_content
        doc_meta = doc.metadata
        doc_id = f"{doc_meta['hadm_id']}_{doc_meta['start_index']}"

        if doc_id in completed_docs:
            pbar.update(1)
            continue

        logger.info(f"Embedding doc_id={doc_id}...")
        with torch.no_grad():
            embeddings = embedding_model.encode(
                [doc_text],
                instruction="",
                max_length=max_length
            )
            embeddings = embeddings.cpu().numpy().tolist()

        collection.add(
            embeddings=embeddings,
            documents=[doc_text],
            metadatas=[doc_meta],
            ids=[doc_id],
        )

        completed_docs.add(doc_id)
        batch_counter += 1

        # Save checkpoint after processing a batch
        if batch_counter >= checkpoint_batch_size:
            with open(checkpoint_file, "w") as f:
                json.dump(list(completed_docs), f)
            logger.info(f"Checkpoint saved with {len(completed_docs)} docs processed.")
            batch_counter = 0

        pbar.update(1)
        torch.cuda.empty_cache()

    # Save any remaining progress
    if batch_counter > 0:
        with open(checkpoint_file, "w") as f:
            json.dump(list(completed_docs), f)
        logger.info(f"Final checkpoint saved with {len(completed_docs)} docs processed.")

    pbar.close()
    logger.info("All documents embedded and added to Chroma.")

# --------------------
# Main Embedding Flow
# --------------------
def main():
    os.makedirs("log", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler('log/0412_rag_embed', mode='w'),
            logging.StreamHandler()
        ]
    )

    setup_environment()

    # 1) Paths and Model Settings
    json_path = "/secure/shared_data/SOAP/MIMIC/full_cases_base.json"
    chroma_db_path = "/secure/shared_data/rag_embedding_model/chroma_db"
    model_cache_dir = "/secure/shared_data/rag_embedding_model"
    model_name = "nvidia/NV-Embed-v2"

    # 2) Load Cases
    cases = load_cases(json_path)

    # 3) Create Documents (Chunking)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=model_cache_dir
    )
    # Check if chunked documents have already been saved
    chunked_doc_file = "chunked_documents.pkl"
    if os.path.exists(chunked_doc_file):
        with open(chunked_doc_file, "rb") as f:
            docs_processed = pickle.load(f)
        logger.info(f"Loaded pre-chunked documents from {chunked_doc_file}")
    else:
        docs_processed = create_documents(cases, tokenizer, max_length=512)
        with open(chunked_doc_file, "wb") as f:
            pickle.dump(docs_processed, f)
        logger.info(f"Created and saved {len(docs_processed)} chunked documents to {chunked_doc_file}")

    # 4) Connect to Chroma
    client = chromadb.PersistentClient(
        path=chroma_db_path,
        settings=Settings(allow_reset=True)
    )
    mimic_collection = client.get_or_create_collection(
        name="mimic_notes_full",
        metadata={"hnsw:space": "cosine"}
    )

    # 5) Load Embedding Model & Embed Docs
    embedding_model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=model_cache_dir,
        device_map="auto"
    )
    embed_docs_in_chroma(docs_processed, embedding_model, mimic_collection, max_length=512)

    logger.info("Embedding script completed successfully.")


if __name__ == "__main__":
    main()
