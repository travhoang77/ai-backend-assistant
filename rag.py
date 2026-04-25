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

# -------- MEMORY CLASSIFIER --------
def classify_memory(text: str):
    text_lower = text.lower()

    # ❌ filter questions (IMPORTANT fix)
    if any(x in text_lower for x in [
        "i don't", "i cant", "i can't", "as an ai", "i do not have"
    ]):
        return "noise"
    
    if any(q in text_lower for q in ["what", "why", "how", "?"]):
        return "noise"

    if any(x in text_lower for x in [
        "my name is",
        "i like",
        "i prefer",
        "i work as",
        "i am"
    ]):
        return "fact"

    if len(text.split()) > 12:
        return "knowledge"
    
    if len(text.split()) > 50:
        return "noise"

    return "noise"


# -------- SAVE MEMORY --------
def save_memory(text: str, user_id: str):
    if not text:
        return

    memory_type = classify_memory(text)

    if memory_type == "noise":
        return

    memory_collection.add(
        documents=[text],
        metadatas=[{
            "user_id": user_id,
            "type": memory_type
        }],
        ids=[f"mem_{uuid.uuid4()}"]
    )

    print(f"🧠 Saved {memory_type} memory:", text)

# -------- GET MEMORY --------
# def get_memory(query: str, user_id: str):
#     print("\n🧠 MEMORY QUERY:", query)

#     results = memory_collection.query(
#         query_texts=[query],
#         n_results=5,
#         where={"user_id": user_id}
#     )

#     docs = results.get("documents", [])

#     if docs and isinstance(docs[0], list):
#         docs = docs[0]

#     if not docs:
#         print("🧠 No memory found")
#         return []

#     print("🧠 Retrieved memories:")
#     for i, d in enumerate(docs):
#         print(f"  {i+1}. {d}")

#     return docs[:3]

def get_memory(query: str, user_id: str):
    print("\n🧠 MEMORY QUERY:", query)

    results = memory_collection.query(
        query_texts=[query],
        n_results=10,
        where={"user_id": user_id}
    )

    docs = results.get("documents", [])
    metas = results.get("metadatas", [])

    if docs and isinstance(docs[0], list):
        docs = docs[0]
        metas = metas[0]

    if not docs:
        print("🧠 No memory found")
        return []

    print("🧠 Retrieved memories:")
    for i, d in enumerate(docs):
        print(f"  {i+1}. {d}")

    # 🔥 deduplicate
    seen = set()
    unique_docs = []
    unique_metas = []

    for doc, meta in zip(docs, metas):
        key = doc.strip().lower()
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
            unique_metas.append(meta)

    # 🔥 prioritize facts
    facts = []
    knowledge = []

    for doc, meta in zip(unique_docs, unique_metas):
        if meta.get("type") == "fact":
            facts.append(doc)
        else:
            knowledge.append(doc)

    return facts[:5] + knowledge[:2]