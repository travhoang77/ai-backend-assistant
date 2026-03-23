import chromadb
from chromadb.utils import embedding_functions
from pdf_loader import load_pdf, chunk_text
import uuid

# -------- INIT DB (PERSISTENT) --------
client = chromadb.Client(
    chromadb.config.Settings(
        persist_directory="./chroma_db"
    )
)

embedding_function = embedding_functions.DefaultEmbeddingFunction()

collection = client.get_or_create_collection(
    name="docs",
    embedding_function=embedding_function
)

# -------- ADD TEXT DOCS --------
def add_documents(docs):
    for i, doc in enumerate(docs):
        collection.add(
            documents=[doc],
            ids=[f"doc_{i}_{uuid.uuid4()}"]
        )

    print("Collection size after add_documents:", len(collection.get()["ids"]))


# -------- QUERY --------
def query_documents(query):
    results = collection.query(
        query_texts=[query],
        n_results=3
    )

    print("Query results:", results)  # 🔥 debug

    return results.get("documents", [])


# -------- ADD PDF --------
def add_pdf(file_path):
    print(f"\n📄 Loading PDF: {file_path}")

    text = load_pdf(file_path)
    if not text:
        print("❌ No text extracted from PDF")
        return

    chunks = chunk_text(text)
    print(f"📦 Total chunks: {len(chunks)}")

    if not chunks:
        print("❌ No chunks created")
        return

    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            ids=[f"{file_path}_{i}_{uuid.uuid4()}"]  # ✅ UNIQUE IDs
        )

    print("✅ Collection size after add_pdf:", len(collection.get()["ids"]))