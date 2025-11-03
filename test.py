from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from qdrant_client import QdrantClient
import os

# =========================================================
# ENVIRONMENT VARIABLES (Render: set in Environment tab)
# =========================================================
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "villages_guatemala"

# =========================================================
# INIT CONNECTIONS
# =========================================================
print("✅ Starting Humadex Qdrant Bridge...")

# ---- Qdrant Connection ----
print("🔌 Connecting to Qdrant...")
try:
    qdrant = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        prefer_grpc=False  # ✅ Forces HTTPS on port 443 (needed for Render)
    )
    qdrant.get_collections()
    print("✅ Qdrant connection successful")
except Exception as e:
    print("❌ Qdrant connection failed:", e)
    qdrant = None

# ---- OpenAI Connection ----
print("🔌 Connecting to OpenAI...")
try:
    client_ai = OpenAI(api_key=OPENAI_API_KEY)
    print("✅ OpenAI connection successful")
except Exception as e:
    print("❌ OpenAI connection failed:", e)
    client_ai = None

# =========================================================
# FASTAPI APP CONFIG
# =========================================================
app = FastAPI(title="Humadex Qdrant Bridge", version="0.1.1")

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# REQUEST MODEL
# =========================================================
class Query(BaseModel):
    question: str

# =========================================================
# HEALTH CHECK ROUTE
# =========================================================
@app.get("/")
def root():
    return {"status": "ok", "message": "Humadex API is running"}

# =========================================================
# QDRANT PING TEST
# =========================================================
@app.get("/ping_qdrant")
def ping_qdrant():
    try:
        collections = qdrant.get_collections()
        return {"status": "connected", "collections": collections}
    except Exception as e:
        return {"error": str(e)}

# =========================================================
# MAIN AI ENDPOINT
# =========================================================
@app.post("/ask")
def ask_question(query: Query):
    question = query.question
    if not client_ai or not qdrant:
        return {"error": "Qdrant or OpenAI not connected"}

    try:
        # 1️⃣ Create embedding for question
        emb = client_ai.embeddings.create(
            model="text-embedding-3-small",
            input=question
        ).data[0].embedding

        # 2️⃣ Search similar docs in Qdrant
        hits = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=emb,
            limit=5
        )

        if not hits:
            return {"answer": "No matching data found in Qdrant."}

        # 3️⃣ Build context from retrieved docs
        context = "\n".join([str(hit.payload) for hit in hits])

        # 4️⃣ Generate final answer with GPT
        completion = client_ai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an intelligent assistant answering based on Guatemalan village data."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ]
        )

        answer = completion.choices[0].message.content
        return {"answer": answer}

    except Exception as e:
        return {"error": str(e)}

# =========================================================
# LOCAL DEV ENTRY POINT
# =========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("test:app", host="0.0.0.0", port=8000, reload=True)
