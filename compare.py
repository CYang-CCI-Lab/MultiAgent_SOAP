import os
import json
import logging
import asyncio

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

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
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
# Text Splitting
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
        # Deduplicate
        for d in docs:
            if d.page_content not in unique_texts:
                unique_texts.add(d.page_content)
                all_docs.append(d)

    logger.info(f"Created {len(all_docs)} total chunks (docs) from the original cases.")
    return all_docs

# --------------------
# Async Embedding in Chroma
# --------------------
async def embed_docs_in_chroma(
    docs: List[Document],
    embedding_model,
    collection,
    max_length: int = 512,
    checkpoint_file: str = "checkpoint_progress.json",
    checkpoint_batch_size: int = 10,
    concurrency: int = 5  # limit for concurrent tasks; adjust as needed
) -> None:
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            completed_docs = set(json.load(f))
        logger.info(f"Resuming from checkpoint. {len(completed_docs)} docs already processed.")
    else:
        completed_docs = set()

    semaphore = asyncio.Semaphore(concurrency)
    pbar = tqdm(total=len(docs), desc="Embedding Documents")

    async def process_doc(doc: Document):
        async with semaphore:
            doc_text = doc.page_content
            doc_meta = doc.metadata
            doc_id = f"{doc_meta['hadm_id']}_{doc_meta['start_index']}"
            
            if doc_id in completed_docs:
                pbar.update(1)
                return doc_id

            logger.info(f"Embedding doc_id={doc_id}...")
            # Run encoding in a separate thread to avoid blocking the event loop.
            embeddings = await asyncio.to_thread(
                embedding_model.encode,
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

            torch.cuda.empty_cache()
            pbar.update(1)
            return doc_id

    tasks = [asyncio.create_task(process_doc(doc)) for doc in docs]
    processed_count = 0

    for future in asyncio.as_completed(tasks):
        doc_id = await future
        if doc_id not in completed_docs:
            completed_docs.add(doc_id)
            processed_count += 1
            if processed_count % checkpoint_batch_size == 0:
                with open(checkpoint_file, "w") as f:
                    json.dump(list(completed_docs), f)
                logger.info(f"Checkpoint saved with {len(completed_docs)} documents processed.")

    # Final checkpoint save.
    with open(checkpoint_file, "w") as f:
        json.dump(list(completed_docs), f)
    logger.info(f"Final checkpoint saved with {len(completed_docs)} documents processed.")
    pbar.close()
    logger.info("All documents embedded and added to Chroma.")

# --------------------
# Main Async Embedding Flow
# --------------------
async def async_main():
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

    # 1) Paths
    json_path = "/secure/shared_data/SOAP/MIMIC/full_cases_base.json"
    chroma_db_path = "/secure/shared_data/rag_embedding_model/chroma_db"
    model_cache_dir = "/secure/shared_data/rag_embedding_model"
    model_name = "nvidia/NV-Embed-v2"

    # 2) Load Cases
    cases = load_cases(json_path)

    # 3) Create Documents
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=model_cache_dir
    )
    docs_processed = create_documents(cases, tokenizer, max_length=512)
    # docs_processed = docs_processed[:10]  # Uncomment for testing with a limited set

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
    await embed_docs_in_chroma(
        docs_processed,
        embedding_model,
        mimic_collection,
        max_length=512
    )

    logger.info("Embedding script completed successfully.")

if __name__ == "__main__":
    asyncio.run(async_main())
