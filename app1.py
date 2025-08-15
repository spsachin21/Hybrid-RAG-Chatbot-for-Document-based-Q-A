# app.py
import os
import tempfile

# must be set before importing transformers if torchvision isn't installed
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import streamlit as st

# HF transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# LangChain
from langchain_community.document_loaders import PyPDFLoader, TextLoader, BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# -----------------------------
# STREAMLIT UI CONFIG
# -----------------------------
st.set_page_config(page_title="RAGLC", page_icon="📄", layout="wide")
st.title("📄🔎 Upload Documents → Ask Questions (RAG + LangChain)")

st.markdown(
    """ Hey! take off your burden and give you the relevant :-)
**How it works**
1) Upload PDFs / TXT / MD / HTML files  
2) We split, embed (FAISS), and index them  
3) Ask questions — answers are grounded in the uploads
"""
)


# -----------------------------
# SIDEBAR SETTINGS
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    embed_model_name = st.selectbox(
        "Embedding model",
        [
            "sentence-transformers/all-MiniLM-L6-v2",      # fast
            "sentence-transformers/all-mpnet-base-v2",     # stronger
            "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        ],
        index=0,
    )

    chunk_size = st.slider("Chunk size (chars)", 300, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk overlap (chars)", 0, 400, 120, 20)
    top_k = st.slider("Top-K retrieved chunks", 2, 10, 4, 1)

    llm_name = st.selectbox(
        "Generator model (open-source)",
        [
            "google/flan-t5-large",   # good CPU baseline
            "google/flan-t5-xl",      # bigger (slower on CPU)
        ],
        index=0,
    )
    max_answer_tokens = st.slider("Max answer tokens", 64, 1024, 256, 32)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

    st.caption(
        "If answers are too short, increase max tokens; if off-topic, try mpnet embeddings or raise Top-K."
    )


# -----------------------------
# SESSION STATE
# -----------------------------
def init_state():
    if "docs" not in st.session_state:
        st.session_state.docs = []          # list[Document]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "embedding_model_name" not in st.session_state:
        st.session_state.embedding_model_name = embed_model_name

init_state()


# -----------------------------
# LOADERS
# -----------------------------
def load_file_to_documents(uploaded_file):
    """
    Write uploaded file to a temp path; load into LangChain Documents.
    Cleans up temp file afterwards.
    """
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix in (".txt", ".md"):
            loader = TextLoader(tmp_path, encoding="utf-8")
        elif suffix in (".html", ".htm"):
            loader = BSHTMLLoader(tmp_path)  # requires beautifulsoup4 (and lxml recommended)
        else:
            st.warning(f"Unsupported file type: {suffix}. Skipping {uploaded_file.name}")
            return []

        docs = loader.load()
        for d in docs:
            d.metadata["source"] = uploaded_file.name
        return docs
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def split_documents(documents, chunk_size=1000, chunk_overlap=120):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


# -----------------------------
# LLM (cache only the HF pipeline wrapper)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_llm_cached(model_name: str, max_tokens: int, temperature: float):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    gen = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_tokens,          # cap generated tokens
        do_sample=(temperature > 0.0),      # sampling only if temp > 0
        temperature=temperature if temperature > 0.0 else 0.0,
        device=-1,                          # CPU; set 0 for GPU
    )
    return HuggingFacePipeline(pipeline=gen)


# -----------------------------
# BUILD / REBUILD INDEX + RAG CHAIN
# -----------------------------
def rebuild_index_and_chain():
    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=st.session_state.embedding_model_name)

    # Split
    chunks = split_documents(
        st.session_state.docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        return None, None, None

    # Vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

    # Prompt & chains
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Use ONLY the provided context to answer. "
                "If the answer is not in the context, say you don't know.",
            ),
            ("human", "Question: {input}\n\nContext:\n{context}"),
        ]
    )

    llm = load_llm_cached(llm_name, max_answer_tokens, temperature)
    doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    # version-safe create_retrieval_chain
    try:
        rag_chain = create_retrieval_chain(retriever, doc_chain)  # newer signature
    except TypeError:
        try:
            rag_chain = create_retrieval_chain(
                retriever=retriever, combine_docs_chain=doc_chain
            )  # newer kw
        except TypeError:
            rag_chain = create_retrieval_chain(
                retriever=retriever, combine_documents_chain=doc_chain
            )  # older kw

    return vector_store, retriever, rag_chain


# -----------------------------
# FILE UPLOAD UI
# -----------------------------
st.subheader("1) Upload your documents")
uploaded_files = st.file_uploader(
    "Drop multiple files here (PDF / TXT / MD / HTML)",
    type=["pdf", "txt", "md", "html", "htm"],
    accept_multiple_files=True,
)

left, right = st.columns([2, 1])
with left:
    if st.button("➕ Add to knowledge base", disabled=(not uploaded_files)):
        added = 0
        for uf in uploaded_files:
            new_docs = load_file_to_documents(uf)
            if new_docs:
                st.session_state.docs.extend(new_docs)
                added += len(new_docs)
        if added > 0:
            st.success(f"Loaded {added} document page(s). Building index…")
            st.session_state.embedding_model_name = embed_model_name
            vs, retr, chain = rebuild_index_and_chain()
            st.session_state.vector_store = vs
            st.session_state.retriever = retr
            st.session_state.rag_chain = chain
        else:
            st.info("No documents added. Check file types.")
with right:
    if st.button("🗑️ Clear knowledge base "):
        st.session_state.docs = []
        st.session_state.vector_store = None
        st.session_state.retriever = None
        st.session_state.rag_chain = None
        st.success("Cleared! Upload fresh documents to start again.")

st.caption(f"Indexed documents: {len(st.session_state.docs)}")
if st.session_state.vector_store is None and len(st.session_state.docs) > 0:
    st.warning("You added docs but the index isn't built yet. Click 'Add to knowledge base'.")

st.divider()


# -----------------------------
# Q&A
# -----------------------------
st.subheader("2) Ask a question about your uploaded documents")
query = st.text_input("Question Please:", placeholder="e.g., How does RandomizedSearchCV work?")
ask = st.button("Press to Ask")

if ask:
    if st.session_state.rag_chain is None:
        st.error("Please upload documents and click 'Add to knowledge base' first.")
    elif not query.strip():
        st.warning("Type a question first.")
    else:
        with st.spinner("Reading your docs and thinking…"):
            result = st.session_state.rag_chain.invoke({"input": query})

        # version-safe keys
        answer = result.get("answer") or result.get("result") or ""
        st.markdown("### ✅ Answer")
        st.write(answer)

        ctx_docs = result.get("context") or result.get("source_documents") or []
        if ctx_docs:
            with st.expander("Show sources"):
                for i, d in enumerate(ctx_docs, 1):
                    src = d.metadata.get("source", "Unknown")
                    st.markdown(f"**{i}.** {src}")
                    st.code((d.page_content or "")[:800])
