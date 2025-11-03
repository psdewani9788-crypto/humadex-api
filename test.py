from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from qdrant_client import QdrantClient
import os

# ===================== SETTINGS =====================
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "villages_guatemala"

# ===================== INIT CONNECTIONS =====================
print("✅ Connecting to Qdrant...")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

print("✅ Connecting to OpenAI...")
client_ai = OpenAI(api_key=OPENAI_API_KEY)

# ===================== FASTAPI APP =====================
app = FastAPI(title="Humadex Qdrant Bridge", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: Query):
    question = query.question
    try:
        emb = client_ai.embeddings.create(
            model="text-embedding-3-small",
            input=question
        ).data[0].embedding

        hits = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=emb,
            limit=5
        )

        context = "\n".join([str(hit.payload) for hit in hits])

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("test:app", host="127.0.0.1", port=8000, reload=True)
