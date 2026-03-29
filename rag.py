import chromadb
from chromadb.utils import embedding_functions
from pdf_loader import load_pdf, chunk_text
import uuid

# -------- INIT DB --------
client = chromadb.Client(
    chromadb.config.Settings(persist_directory="./chroma_db")
)

embedding_function = embedding_functions.DefaultEmbeddingFunction()

# -------- DOC COLLECTION (RAG) --------
collection = client.get_or_create_collection(
    name="docs",
    embedding_function=embedding_function
)

# -------- MEMORY COLLECTION --------
memory_collection = client.get_or_create_collection(
    name="memory",
    embedding_function=embedding_function
)


# -------- ADD TEXT DOCS --------
def add_documents(docs):
    for i, doc in enumerate(docs):
        collection.add(
            documents=[doc],
            ids=[f"doc_{i}_{uuid.uuid4()}"]
        )

    print("Collection size:", len(collection.get()["ids"]))


# -------- QUERY DOCS (RAG) --------
def query_documents(query):
    results = collection.query(   # ✅ FIXED
        query_texts=[query],
        n_results=3
    )

    print("📄 RAG results:", results)

    return results.get("documents", [])


# -------- ADD PDF --------
def add_pdf(file_path):
    print(f"\n📄 Loading PDF: {file_path}")

    text = load_pdf(file_path)
    if not text:
        print("❌ No text extracted")
        return

    chunks = chunk_text(text)
    print(f"📦 Chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            ids=[f"{file_path}_{i}_{uuid.uuid4()}"]
        )

    print("✅ Docs stored:", len(collection.get()["ids"]))


# -------- SAVE MEMORY --------
def save_memory(text: str, user_id: str):
    if not text or len(text.split()) < 5:
        return

    memory_collection.add(
        documents=[text],
        metadatas=[{"user_id": user_id}],
        ids=[f"mem_{uuid.uuid4()}"]
    )

    print("🧠 Saved memory:", text)


# -------- GET MEMORY --------
def get_memory(query: str, user_id: str):
    results = memory_collection.query(
        query_texts=[query],
        n_results=3,
        where={"user_id": user_id}   # ✅ correct usage here
    )

    docs = results.get("documents", [])

    if docs and isinstance(docs[0], list):
        return docs[0]

    return docs or []