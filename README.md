# Hybrid-RAG-Chatbot-for-Document-based-Q-A
 LangChain-based RAG chatbot to process documents- PDFs / TXT / MD / HTML files

📄🔎 Upload Documents → Ask Questions (RAG + LangChain)

Objective: Build a local, no-LLM-API RAG app where users can upload PDFs/TXT/Markdown/HTML, the app indexes the content (split → embed → store), and then lets users ask questions whose answers are grounded in the uploaded documents. It emphasizes: Speed vs. accuracy presets (user can trade off). Hybrid retrieval (keyword BM25 + semantic FAISS) with optional cross-encoder reranking. Open-source generation (FLAN-T5 via Hugging Face), so it runs on CPU if needed.

A simple, friendly Streamlit UI.

***************************************************************************************************************************************************************************************************************************************************************************************** 
Frontend & Orchestration

Streamlit – Provides an interactive web UI for uploading documents, setting parameters, and viewing results.
Streamlit Session State – Stores documents, indexes, retrievers, and models across user interactions for efficiency.
Streamlit Cache Resource – Prevents reloading heavy models on every request, improving responsiveness.
***********************************************************************************************************
Document Loading

PyPDFLoader – Parses PDF files into structured text segments.
TextLoader – Reads plain text (.txt) and markdown (.md) files into LangChain Document objects.
BSHTMLLoader – Extracts readable text from HTML pages while preserving basic structure.
***********************************************************************************************************
Text Processing

MarkdownHeaderTextSplitter – Splits content based on headings (#, ##, ###) to keep semantically related text together.
RecursiveCharacterTextSplitter – Further divides text into size-controlled chunks with optional overlap for better context recall.
*********************************************************************************************************************************
Vectorization & Indexing

HuggingFaceEmbeddings – Converts text chunks into dense vector embeddings for semantic similarity search.
FAISS (Facebook AI Similarity Search) – High-performance vector database enabling fast approximate nearest neighbor search on embeddings.
*******************************************************************************************************************************************
Retrieval

BM25Retriever – Keyword-based retrieval method that excels at matching exact terms, acronyms, and numbers.
FAISS Retriever with Maximal Marginal Relevance (MMR) – Ensures retrieved chunks are both relevant and diverse.
EnsembleRetriever – Combines BM25 and FAISS retrievers with weighted scoring to balance keyword and semantic matches.
***********************************************************************************************************************************
Reranking (Optional)

CrossEncoderReranker – Uses a cross-encoder model to re-score and re-order top retrieved chunks for higher precision.
HuggingFaceCrossEncoder (ms-marco-MiniLM-L-6-v2) – Pretrained cross-encoder specialized in passage relevance ranking.
***********************************************************************************************************************************
Language Model (LLM)

HuggingFace Transformers – Framework for loading and running open-source sequence-to-sequence models.
FLAN-T5 (google/flan-t5-large / -xl) – Instruction-tuned text generation model that can follow prompts and generate concise, grounded answers.
HuggingFacePipeline – Wraps the Hugging Face model into a LangChain-compatible LLM interface.
***************************************************************************************************************************************************
Prompting & Chaining

ChatPromptTemplate – Defines structured prompts with system instructions to keep answers grounded in provided context.
create_stuff_documents_chain – Combines retrieved document chunks into a single context for the LLM.
create_retrieval_chain – Links retriever and generation chain into a unified RAG pipeline.

