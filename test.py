from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI, APIConnectionError, APIError, AuthenticationError
from qdrant_client import QdrantClient
import os

# =========================================================
# ENVIRONMENT VARIABLES
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
        prefer_grpc=False  # ✅ Required for HTTPS (Render compatibility)
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
    print("✅ OpenAI client initialized")
except Exception as e:
    print("❌ OpenAI initialization failed:", e)
    client_ai = None

# =========================================================
# FASTAPI CONFIG
# =========================================================
app = FastAPI(title="Humadex Qdrant Bridge", version="0.1.2")

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
# ROOT ROUTE
# =========================================================
@app.get("/")
def root():
    return {"status": "ok", "message": "Humadex API is running"}

# =========================================================
# QDRANT TEST ROUTE
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
        # 1️⃣ --- Test OpenAI connectivity ---
        try:
            _ = client_ai.models.list()
        except AuthenticationError:
            return {"error": "Invalid OpenAI API key"}
        except APIConnectionError:
            return {"error": "Cannot reach OpenAI (network issue)"}
        except APIError as e:
            return {"error": f"OpenAI API error: {str(e)}"}

        # 2️⃣ --- Generate embedding for the question ---
        try:
            emb = client_ai.embeddings.create(
                model="text-embedding-3-small",
                input=question
            ).data[0].embedding
        except Exception as e:
            return {"error": f"Embedding error: {str(e)}"}

        # 3️⃣ --- Query Qdrant for similar vectors ---
        try:
            hits = qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=emb,
                limit=5
            )
        except Exception as e:
            return {"error": f"Qdrant search error: {str(e)}"}

        if not hits:
            return {"answer": "No matching data found in Qdrant."}

        # 4️⃣ --- Build text context from retrieved payloads ---
        context = "\n".join([str(hit.payload) for hit in hits])

        # 5️⃣ --- Generate GPT answer ---
        try:
            completion = client_ai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an intelligent assistant answering based on Guatemalan village data."
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}"
                    }
                ]
            )
            answer = completion.choices[0].message.content
            return {"answer": answer}
        except Exception as e:
            return {"error": f"Completion error: {str(e)}"}

    except Exception as e:
        return {"error": str(e)}

# =========================================================
# LOCAL DEVELOPMENT ENTRY POINT
# =========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("test:app", host="0.0.0.0", port=8000, reload=True)
