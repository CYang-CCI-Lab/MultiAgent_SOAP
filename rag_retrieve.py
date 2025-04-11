import os
import json
import logging

import torch
from dotenv import load_dotenv, find_dotenv
from transformers import AutoTokenizer, AutoModel
from langchain_community.retrievers import BM25Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any

# ------------------
# Logging Setup
# ------------------

logger = logging.getLogger(__name__)

# ------------------
# Environment Setup
# ------------------
def setup_environment() -> None:
    env_path = find_dotenv()
    if not env_path:
        env_path = "/home/yl3427/.env"
    if not load_dotenv(env_path):
        raise Exception("Failed to load .env file")
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    logger.info("Environment setup complete for retrieval.")


# --------------------
# Data Loading
# --------------------
def load_cases(json_path: str) -> Dict[str, Any]:
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
    Creates BM25-compatible Documents. We do not necessarily need to chunk again
    if we only want each entire 'before_diagnosis' text as a single Document for BM25.
    But often you'll chunk for consistency with embedding. 
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

    docs = []
    unique_texts = set()

    for hadm_id, data in cases.items():
        full_text = data["before_diagnosis"]
        split_docs = text_splitter.create_documents(
            texts=[full_text],
            metadatas=[{
                "hadm_id": hadm_id,
                "full_text": full_text,
                "diagnosis": data["after_diagnosis"]
            }]
        )
        for d in split_docs:
            if d.page_content not in unique_texts:
                unique_texts.add(d.page_content)
                docs.append(d)

    logger.info(f"Created {len(docs)} doc chunks for BM25 retrieval.")
    return docs


# --------------------
# Hybrid Query (Union)
# --------------------
def hybrid_query(
    cases: Dict[str, Any],
    docs: List[Document],
    collection,
    embedding_model,
    query_text: str,
    query_prefix: str,
    max_length: int = 512,
    semantic_k: int = 5,
    bm25_k: int = 5,
    bm25_weight: float = 0.5
) -> List[str]:

    logger.info("Starting hybrid retrieval...")

    # --- Semantic Retrieval ---
    query_embedding = embedding_model.encode(
        [query_text],
        instruction=query_prefix,
        max_length=max_length
    ).cpu().numpy().tolist()

    semantic_results = collection.query(
        query_embeddings=query_embedding,
        n_results=semantic_k
    )
    results_semantic_ids = [full_id.split("_")[0] for full_id in semantic_results["ids"][0]]
    semantic_id_set = set(results_semantic_ids)
    logger.info(f"Semantic top-{semantic_k} hadm_ids: {results_semantic_ids}")

    # --- BM25 Retrieval ---
    bm25_retriever = BM25Retriever.from_documents(docs, k=bm25_k)
    bm25_results = bm25_retriever.get_relevant_documents(query_text)
    results_bm25_ids = [doc.metadata["hadm_id"] for doc in bm25_results]
    bm25_id_set = set(results_bm25_ids)
    logger.info(f"BM25 top-{bm25_k} hadm_ids: {results_bm25_ids}")

    # --------------------------
    # Union of both sets
    # --------------------------
    combined_ids = semantic_id_set | bm25_id_set
    logger.info(f"Union of semantic & BM25 -> total {len(combined_ids)} unique hadm_ids")

    # Weighted scoring
    semantic_weight = 1.0 - bm25_weight
    ids_to_score = {}

    # Build a ranking score
    for hadm_id in combined_ids:
        score = 0.0

        if hadm_id in semantic_id_set:
            idx_sem = results_semantic_ids.index(hadm_id)
            score += semantic_weight * (1 / (idx_sem + 1))

        if hadm_id in bm25_id_set:
            idx_bm25 = results_bm25_ids.index(hadm_id)
            score += bm25_weight * (1 / (idx_bm25 + 1))

        ids_to_score[hadm_id] = score

    # Sort by combined score
    sorted_ids = sorted(ids_to_score.keys(), key=lambda x: ids_to_score[x], reverse=True)

    # Build final result list with dictionaries
    retrieved_docs = []
    for doc_id in sorted_ids:
        # Determine which retriever(s) provided this doc (logging only)
        source_list = []
        if doc_id in semantic_id_set:
            source_list.append("Semantic")
        if doc_id in bm25_id_set:
            source_list.append("BM25")

        logger.info(
            f"Doc hadm_id={doc_id} => from {', '.join(source_list)}; combined_score={ids_to_score[doc_id]:.4f}"
        )

        before = cases[doc_id]["before_diagnosis"]
        after = cases[doc_id]["after_diagnosis"]
        final_text = f"{before}\nDischarge Diagnosis: {after}"

        retrieved_docs.append({
            "text": final_text,
            "diagnosis": after,
            "score": ids_to_score[doc_id]
        })

    logger.info(f"Retrieved {len(retrieved_docs)} docs total.")
    return retrieved_docs


# --------------------
# Main Retrieval Flow
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
    json_path = "/secure/shared_data/SOAP/MIMIC/cases_base.json"
    chroma_db_path = "/secure/shared_data/rag_embedding_model/chroma_db"
    model_cache_dir = "/secure/shared_data/rag_embedding_model"
    model_name = "nvidia/NV-Embed-v2"

    # 2) Load Cases
    cases = load_cases(json_path)

    # 3) Create Documents for BM25
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=model_cache_dir
    )
    docs_processed = create_documents(cases, tokenizer, max_length=512)

    # 4) Connect to existing Chroma DB
    client = chromadb.PersistentClient(
        path=chroma_db_path,
        settings=Settings(allow_reset=True)
    )
    mimic_collection = client.get_or_create_collection(
        name="mimic_notes_full",
        metadata={"hnsw:space": "cosine"}
    )

    # Load embedding model for semantic retrieval
    embedding_model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=model_cache_dir,
        device_map="auto"
    )

    # 5) Hybrid Retrieval
    query_prefix = (
        "Given the following clinical note, retrieve the most similar clinical case. "
        "The clinical note is:\n\n"
    )
    query_text = (
        "pleuritic right chest pain\n- patient started on coumadin\n..."
        "respiratory support\no2 delivery device: nasal cannula\nspo2: 98%\n..."
    )

    retrieved = hybrid_query(
        cases=cases,
        docs=docs_processed,
        collection=mimic_collection,
        embedding_model=embedding_model,
        query_text=query_text,
        query_prefix=query_prefix,
        max_length=512,
        semantic_k=5,
        bm25_k=5,
        bm25_weight=0.5
    )

    # 6) Print out final docs
    logger.info("---------- FINAL RETRIEVED DOCS ----------")
    for idx, doc_text in enumerate(retrieved, start=1):
        print(f"\n--- Retrieved Doc #{idx} ---")
        print(doc_text)


if __name__ == "__main__":

    main()
