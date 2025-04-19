# build_index.py
# -*- coding: utf-8 -*-

# Purpose: Loads scraped scheme data, preprocesses/chunks it,
#          generates embeddings, builds a FAISS index, and saves
#          the index and chunk data for the QA application.

import json
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import sys
import os
import time
import re
import torch

# ==================================
# ---        CONFIGURATION       ---
# ==================================
INPUT_JSON_FILE = os.path.join("data", "myscheme_100_schemes_generic.json")
OUTPUT_INDEX_FILE = os.path.join("data", "faiss_index.idx")
OUTPUT_CHUNKS_FILE = os.path.join("data", "scheme_chunks.json") # Store text chunks + metadata

# Choose same embedding model used for querying
embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
# Verify embedding dimension matches model (384 for all-MiniLM-L6-v2)
embedding_dim = 384

# ==================================
# ---        LOGGING SETUP       ---
# ==================================
log_format = '%(asctime)s - %(levelname)s - %(funcName)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format,
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# ==================================
# ---    DATA PROCESSING FUNC    ---
# ==================================
# <Copy the final preprocess_and_chunk() function definition here>
def preprocess_and_chunk(data):
    """ Processes scraped data into text chunks suitable for embedding. """
    chunk_texts = []
    chunk_metadata = []
    if not data: return chunk_texts, chunk_metadata
    logger.info(f"Starting preprocessing for {len(data)} schemes...")
    schemes_processed = 0
    total_sections_processed = 0
    for i, scheme in enumerate(data):
        if not isinstance(scheme, dict): continue
        scheme_name = scheme.get('main_heading_h1', '').strip() or scheme.get('page_title', f'Unknown Scheme {i+1}').strip()
        scheme_url = scheme.get('scheme_url', f'URL_{i+1}')
        page_title = scheme.get('page_title', scheme_name)
        sections = scheme.get('extracted_sections', [])
        if not isinstance(sections, list): sections = []
        if not sections: logger.warning(f"Scheme '{scheme_name}' has no valid sections.")
        schemes_processed += 1
        for section in sections:
            if not isinstance(section, dict): continue
            heading = section.get('heading_text', 'General Info').strip()
            content = section.get('content', '').strip()
            content = re.sub(r'\n{3,}', '\n\n', content).strip()
            if not content: continue
            chunk_text = f"Scheme Name: {scheme_name}\nSection: {heading}\n\n{content}"
            chunk_texts.append(chunk_text)
            metadata = {'scheme_url': scheme_url, 'scheme_name': scheme_name, 'page_title': page_title, 'section_heading': heading}
            chunk_metadata.append(metadata)
            total_sections_processed += 1
    logger.info(f"Preprocessing finished. Processed {schemes_processed} schemes.")
    logger.info(f"Created {len(chunk_texts)} text chunks from {total_sections_processed} valid sections.")
    return chunk_texts, chunk_metadata

# ==================================
# ---    EMBEDDING & INDEXING    ---
# ==================================
def embed_and_index(chunk_texts, chunk_metadata):
    """Generates embeddings and builds/saves FAISS index."""
    if not chunk_texts: logger.error("No text chunks provided."); return None, None

    if torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    logger.info(f"Loading embedding model: {embedding_model_name}")
    try:
        embedding_model = SentenceTransformer(embedding_model_name, device=device)
        embedding_dim_actual = embedding_model.get_sentence_embedding_dimension()
        if embedding_dim_actual != embedding_dim:
             logger.warning(f"Model dim {embedding_dim_actual} differs from config {embedding_dim}. Using actual.")
             dim = embedding_dim_actual
        else:
             dim = embedding_dim
        logger.info(f"Embedding model loaded. Dimension: {dim}")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}", exc_info=True); return None, None

    logger.info(f"Generating embeddings for {len(chunk_texts)} chunks...")
    start_time = time.time()
    try:
        chunk_embeddings = embedding_model.encode(chunk_texts, show_progress_bar=True, convert_to_numpy=True)
        if chunk_embeddings.dtype != np.float32:
            chunk_embeddings = chunk_embeddings.astype(np.float32)
        logger.info(f"Embeddings generated in {time.time() - start_time:.2f}s. Shape: {chunk_embeddings.shape}")
        if chunk_embeddings.shape != (len(chunk_texts), dim):
             raise ValueError("Embeddings shape mismatch!")
    except Exception as e:
        logger.error(f"Failed during embedding generation: {e}", exc_info=True); return None, None

    logger.info("Building FAISS index...")
    try:
        faiss_index = faiss.IndexFlatL2(dim)
        faiss_index.add(chunk_embeddings)
        logger.info(f"FAISS index built (IndexFlatL2). Vectors indexed: {faiss_index.ntotal}")
        if faiss_index.ntotal != len(chunk_texts): logger.warning("Index count mismatch!")
    except Exception as e:
        logger.error(f"Failed to build FAISS index: {e}", exc_info=True); return None, None

    try:
        os.makedirs(os.path.dirname(OUTPUT_INDEX_FILE), exist_ok=True)
        logger.info(f"Saving FAISS index to {OUTPUT_INDEX_FILE}")
        faiss.write_index(faiss_index, OUTPUT_INDEX_FILE)
        logger.info(f"Saving chunks and metadata to {OUTPUT_CHUNKS_FILE}")
        # Store chunks list alongside metadata for easy loading in QA app
        chunks_to_save = [{'text': text, 'metadata': meta} for text, meta in zip(chunk_texts, chunk_metadata)]
        with open(OUTPUT_CHUNKS_FILE, 'w', encoding='utf-8') as f:
            json.dump(chunks_to_save, f, ensure_ascii=False, indent=2)
        logger.info("Index and chunk data saved successfully.")
        return faiss_index, chunks_to_save
    except Exception as e:
        logger.error(f"Error saving index or chunk data: {e}", exc_info=True); return None, None

# ==================================
# ---     MAIN EXECUTION BLOCK   ---
# ==================================
def main():
    logger.info("--- Starting Index Building Process ---")
    if not os.path.exists(INPUT_JSON_FILE):
        logger.error(f"Input file not found: {INPUT_JSON_FILE}"); return
    try:
        with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f: scraped_data = json.load(f)
        logger.info(f"Loaded {len(scraped_data)} schemes from {INPUT_JSON_FILE}")
    except Exception as e: logger.error(f"Failed to load scraped data: {e}"); return

    chunk_texts, chunk_metadata = preprocess_and_chunk(scraped_data)
    if not chunk_texts: logger.error("Preprocessing failed."); return

    index, chunks_saved = embed_and_index(chunk_texts, chunk_metadata)
    if not index or not chunks_saved: logger.error("Embedding or indexing failed."); return

    logger.info("--- Index Building Process Finished Successfully ---")
    print(f"\n---> FAISS index saved to: {OUTPUT_INDEX_FILE}")
    print(f"---> Chunk data saved to: {OUTPUT_CHUNKS_FILE}")

if __name__ == "__main__":
    main()