# WebScraping-SLM
End-to-end Python project demonstrating web scraping (Selenium/Requests) of myscheme.gov.in and building a RAG Question Answering system using Sentence Transformers, FAISS, and Flan-T5 to answer natural language questions about Indian government schemes.
WebScraping-SLM/
├── data/
│   ├── myscheme_100_schemes_generic.json
│   ├── faiss_index.idx
│   └── scheme_chunks.json
├── .gitignore
├── README.md
├── requirements.txt
├── scrape_schemes.py                    # Optional: Script to re-scrape data
├── build_index.py                       # Optional: Script to re-build FAISS index
└── qa_interactive.py                    # Main script to run the Q&A Bot