# qa_interactive.py
# -*- coding: utf-8 -*-

# Purpose: Loads pre-built FAISS index, chunks, and models
#          to run an interactive Question Answering session about MyScheme data.

import json
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import logging
import sys
import os
import time
import torch
import pprint # For debug printing context (optional)

# ==================================
# ---        CONFIGURATION       ---
# ==================================
INDEX_FILE = os.path.join("data", "faiss_index.idx")
CHUNKS_FILE = os.path.join("data", "scheme_chunks.json")

# Ensure these match the models used during indexing and desired for generation
embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
# embedding_dim = 384 # Dimension is read from index now
generator_model_name = 'google/flan-t5-large' # Using the best performing one from tests

RETRIEVAL_TOP_K = 3      # Number of chunks to retrieve
GENERATION_MAX_NEW_TOKENS = 250 # Max length of generated answer

# ==================================
# ---        LOGGING SETUP       ---
# ==================================
log_format = '%(asctime)s - %(levelname)s - %(funcName)s - %(message)s'
# Set default level higher for cleaner user interaction
logging.basicConfig(level=logging.WARNING, format=log_format,
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# ==================================
# ---      HELPER FUNCTIONS      ---
# ==================================

def load_models_and_index():
    """Loads Embedding model, Generator model/tokenizer, FAISS index, and chunk data."""
    logger.info("--- Loading QA components ---")
    components = {}

    # Determine Device
    if torch.cuda.is_available():
        components['device'] = torch.device("cuda")
        logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        components['device'] = torch.device("cpu")
        logger.info("CUDA not available. Using CPU.")

    # 1. Load FAISS Index
    if not os.path.exists(INDEX_FILE):
         logger.error(f"FAISS index file not found: {INDEX_FILE}"); return None
    try:
        logger.info(f"Loading FAISS index from: {INDEX_FILE}")
        components['faiss_index'] = faiss.read_index(INDEX_FILE)
        # Check index dimension
        components['embedding_dim'] = components['faiss_index'].d
        logger.info(f"FAISS index loaded. Vectors: {components['faiss_index'].ntotal}, Dim: {components['embedding_dim']}")
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}", exc_info=True); return None

    # 2. Load Chunk Data
    if not os.path.exists(CHUNKS_FILE):
         logger.error(f"Chunk data file not found: {CHUNKS_FILE}"); return None
    try:
        logger.info(f"Loading chunk data from: {CHUNKS_FILE}")
        with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
            loaded_chunks_data = json.load(f) # List of {'text': ..., 'metadata': ...}
            components['chunk_texts'] = [item['text'] for item in loaded_chunks_data]
            # Store metadata too if needed for display or filtering
            # components['chunk_metadata'] = [item['metadata'] for item in loaded_chunks_data]
        logger.info(f"Loaded {len(components['chunk_texts'])} chunks.")
        if components['faiss_index'].ntotal != len(components['chunk_texts']):
            logger.error("Mismatch between index size and loaded chunk count!"); return None
    except Exception as e:
        logger.error(f"Failed to load chunk data: {e}", exc_info=True); return None

    # 3. Load Embedding Model
    try:
        logger.info(f"Loading embedding model: {embedding_model_name}")
        components['embedding_model'] = SentenceTransformer(embedding_model_name, device=components['device'])
        if components['embedding_model'].get_sentence_embedding_dimension() != components['embedding_dim']:
            logger.error("Embedding model dimension mismatch with FAISS index!")
            return None
        logger.info("Embedding model loaded.")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}", exc_info=True); return None

    # 4. Load Generator Model & Tokenizer
    try:
        logger.info(f"Loading generator tokenizer: {generator_model_name}")
        components['tokenizer'] = AutoTokenizer.from_pretrained(generator_model_name)
        logger.info(f"Loading generator model: {generator_model_name}")
        components['generator_model'] = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name).to(components['device'])
        components['generator_model'].eval()
        logger.info("Generator model and tokenizer loaded.")
    except Exception as e:
        logger.error(f"Failed to load generator model/tokenizer: {e}", exc_info=True); return None

    logger.info("--- All QA components loaded successfully ---")
    return components

# <Copy retrieve_context() definition here - uses components dict>
def retrieve_context(question, components, top_k=RETRIEVAL_TOP_K):
    logger.debug(f"Retrieving top {top_k} context chunks...")
    try:
        model = components['embedding_model']
        index = components['faiss_index']
        chunks_list = components['chunk_texts']
        question_embedding = model.encode([question], convert_to_numpy=True).astype(np.float32)
        distances, indices = index.search(question_embedding, top_k)
        retrieved_indices = indices[0]
        retrieved_chunks = [chunks_list[i] for i in retrieved_indices if 0 <= i < len(chunks_list)]
        logger.debug(f"Retrieved {len(retrieved_chunks)} chunks.")
        return retrieved_chunks
    except Exception as e: logger.error(f"Error during context retrieval: {e}"); return []

# <Copy build_rag_prompt() definition here>
def build_rag_prompt(question, context_chunks):
    if not context_chunks: return f"Question: {question}\nAnswer:"
    context_string = "\n\n---\n\n".join(context_chunks)
    prompt = f"""Answer the following question based strictly on the context provided below. If the information needed to answer the question is not present in the context, state that you cannot answer based on the context.

Context:
{context_string}

---
Question: {question}

Answer based only on the provided context:"""
    logger.debug("RAG prompt created.")
    return prompt

# <Copy generate_answer() definition here - uses components dict>
def generate_answer(prompt, components, max_new_tokens=GENERATION_MAX_NEW_TOKENS):
    logger.debug("Generating answer (using greedy decoding)...")
    try:
        model = components['generator_model']
        tokenizer = components['tokenizer']
        device = components['device']
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.debug(f"Answer generated: {answer[:100]}...")
        return answer.strip()
    except Exception as e: logger.error(f"Error during answer generation: {e}"); return "Error generating answer."

# <Copy answer_question() definition here - uses components dict>
def answer_question(user_question, components):
    """ Answers a user question using the full RAG pipeline and loaded components."""
    logger.info(f"Answering question: '{user_question}'") # Use INFO here for visibility
    start_time = time.time()
    try:
        context_chunks = retrieve_context(user_question, components)
        prompt = build_rag_prompt(user_question, context_chunks)
        final_answer = generate_answer(prompt, components)
        end_time = time.time()
        logger.info(f"Answer generated in {end_time - start_time:.2f} seconds.")
        return final_answer
    except Exception as e:
        logger.error(f"Error in answer_question pipeline: {e}", exc_info=True)
        return "Sorry, an error occurred while processing your question."

# <Copy start_qa_session_simple() definition here - uses components dict>
def start_qa_session_simple(components):
    """Starts an interactive loop (simplified output)."""
    print("\n--- MyScheme QA Bot ---")
    print("Ask questions about the scraped government schemes.")
    print("Type 'quit' or 'exit' anytime to stop.")
    print("-" * 25)
    if not components: print("ERROR: QA components not loaded."); return

    while True:
        try:
            user_question = input("\n❓ Your Question: ")
            if user_question.strip().lower() in ['quit', 'exit']: print("\nExiting QA Bot. Goodbye!"); break
            if not user_question.strip(): continue
            print("🧠 Thinking...")
            model_answer = answer_question(user_question, components) # Pass components dict
            print("\n💬 Model Answer:")
            print(model_answer)
            print("-" * 25)
        except (EOFError, KeyboardInterrupt): print("\nExiting QA Bot."); break
        except Exception as e:
             print(f"\nAn error occurred: {e}")
             logger.error(f"Error during QA loop: {e}", exc_info=True)

# ==================================
# ---     MAIN EXECUTION BLOCK   ---
# ==================================
def main():
    # Load all necessary components first
    print("Loading models and index... This may take a moment.")
    qa_components = load_models_and_index()

    if qa_components:
        # Start the interactive session
        start_qa_session_simple(qa_components)
    else:
        print("\nFailed to load necessary components for the QA system. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    # Set logger level for QA app - suppress info/debug from underlying funcs if desired
    logging.getLogger().setLevel(logging.WARNING) # Set default to WARNING for cleaner interaction
    # To see more detail during QA, comment out the above line or set level=logging.INFO
    main()