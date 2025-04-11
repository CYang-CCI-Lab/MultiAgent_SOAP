import os
import json
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 3, 4"
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
# Embedding in Chroma
# --------------------
def embed_docs_in_chroma(
    docs: List[Document],
    embedding_model,
    collection,
    max_length: int = 512
) -> None:
    """
    Embeds documents into the Chroma collection.
    """
    pbar = tqdm(total=len(docs), desc="Embedding Documents")
    for doc in docs:
        doc_text = doc.page_content
        doc_meta = doc.metadata
        doc_id = f"{doc_meta['hadm_id']}_{doc_meta['start_index']}"
        
        # Log each doc ID as we process it
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
        pbar.update(1)
        torch.cuda.empty_cache()
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
            logging.FileHandler('log/0410_MA_3_probs_parallel_static.log', mode='w'),  # 파일로 저장
            logging.StreamHandler()  # 콘솔에 출력
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
    docs_processed = docs_processed[:10]  # Limit to 10 documents for testing

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
