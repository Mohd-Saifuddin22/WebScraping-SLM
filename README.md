Note: Recommend listing faiss-cpu here as it's easier for average users to install across platforms. Users with GPUs can manually install faiss-gpu if they prefer.

# MyScheme QA - RAG Demo

This project demonstrates a Retrieval-Augmented Generation (RAG) based Question Answering system built using data scraped from the Indian government's MyScheme portal (`myscheme.gov.in`).

It allows users to ask natural language questions about various government schemes and receive answers based on the scraped information included in this repository.

## Included Data

This repository includes pre-processed data for **100 schemes** scraped around April 2025:

* `data/myscheme_100_schemes_generic.json`: The raw scraped data containing sections for each scheme.
* `data/scheme_chunks.json`: Pre-processed text chunks (section-based) with metadata.
* `data/faiss_index.idx`: A pre-built FAISS vector index based on the `scheme_chunks.json` using the `sentence-transformers/all-MiniLM-L6-v2` embedding model.

This allows you to run the Q&A system directly without needing to scrape or build the index first.

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd myscheme-qa
    ```

2.  **Create & Activate Python Environment:** (Recommended, requires Python 3.8+)
    ```bash
    python -m venv venv
    # Linux/macOS:
    source venv/bin/activate
    # Windows:
    # venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    * First, install PyTorch matching your system (CPU or specific CUDA version). See instructions at [pytorch.org](https://pytorch.org/). Example (CPU): `pip install torch torchvision torchaudio`
    * Then, install other requirements:
        ```bash
        pip install -r requirements.txt
        ```

## Running the QA Bot (Easy Way)

This uses the pre-built index and data included in the repository.

1.  **Activate your environment.**
2.  **Run the interactive script:**
    ```bash
    python qa_interactive.py
    ```
3.  The script will load the necessary models (`all-MiniLM-L6-v2` for embedding, `google/flan-t5-large` for generation) and the pre-built index.
    * **Note:** The first run will download the models, which can take some time and require significant disk space and RAM (especially for `flan-t5-large`). Running on a machine with a **GPU and sufficient RAM (>16GB recommended)** will provide much better performance for answer generation.
4.  Once loaded, you will be prompted: `❓ Your Question:`
5.  Ask questions about the schemes (e.g., "What are the benefits of Jagananna Chedodu?", "Who is eligible for PM-KISAN?").
6.  Type `quit` or `exit` to stop the bot.

## Optional Steps

These steps are *not* required to run the QA bot if you use the included data/index files.

### Re-scraping Data

* **Requires:** Google Chrome browser installed.
* **Warning:** Websites change structure. This scraper might break over time.
* **Command:** `python scrape_schemes.py`
* **Output:** Overwrites `data/myscheme_100_schemes_generic.json`.

### Re-building the Index

* Run this if you have re-scraped the data or want to change the embedding model (update `embedding_model_name` in `build_index.py` first).
* **Command:** `python build_index.py`
* **Output:** Overwrites `data/faiss_index.idx` and `data/scheme_chunks.json`. This step generates embeddings and can take time (faster with a GPU).
* **Important:** If you change the embedding model here, you **must** change `embedding_model_name` in `qa_interactive.py` to match.
