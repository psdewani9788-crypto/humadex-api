from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np

# ===================== SETTINGS =====================
QDRANT_URL = "https://954a7f1c-ebec-46fb-bbe5-c16ff15752d1.us-east4-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Tu5Y-Q--MrJ4oedSIv5bKFeQcnVf5O0hmyv6Sa-u3J8"
OPENAI_API_KEY = "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"
COLLECTION_NAME = "villages_guatemala"

# ===================== INIT CONNECTIONS =====================
print("✅ Connecting to Qdrant...")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

print("✅ Connecting to OpenAI...")
client_ai = OpenAI(api_key=OPENAI_API_KEY)

# ===================== FASTAPI APP =====================
app = FastAPI(title="Humadex Qdrant Bridge", version="0.1.0")

# ✅ Allow browser + Dust.tt + Render access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== REQUEST MODEL =====================
class Query(BaseModel):
    question: str

# ===================== API ROUTE =====================
@app.post("/ask")
def ask_question(query: Query):
    question = query.question

    try:
        # 1️⃣ Create embedding for the user question
        emb = client_ai.embeddings.create(
            model="text-embedding-3-small",
            input=question
        ).data[0].embedding

        # 2️⃣ Search similar data in Qdrant
        hits = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=emb,
            limit=5
        )

        # 3️⃣ Combine context from top results
        context = "\n".join(
            [str(hit.payload) for hit in hits]
        )

        # 4️⃣ Ask OpenAI to answer using that context
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

# ===================== LOCAL STARTUP MESSAGE =====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("test:app", host="127.0.0.1", port=8000, reload=True)
