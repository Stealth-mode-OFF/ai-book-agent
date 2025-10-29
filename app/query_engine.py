import os
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI

load_dotenv()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-turbo")
TOP_K = int(os.getenv("TOP_K", "6"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "900"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))

encoder = SentenceTransformer("all-MiniLM-L6-v2")
db = chromadb.PersistentClient(path="embeddings")
collection = db.get_or_create_collection("josef_knowledge")
client = OpenAI()

SYSTEM_PROMPT = (
    "You are Josef's elite business coach. "
    "Use insights from context only, focused on scaling, sales, psychology, negotiation, brand and automation."
)


def retrieve_context(question: str, top_k: int = TOP_K):
    q_emb = encoder.encode([question])[0].tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=top_k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return docs, metas


def build_user_prompt(question: str, docs, metas):
    ctx = ""
    sources = []
    for d, m in zip(docs, metas):
        ctx += f"\n---\n{d}\n"
        if m and m.get("source"):
            sources.append(m["source"])
    sources_list = list(dict.fromkeys(sources))
    return f"Context:\n{ctx}\nQuestion: {question}", sources_list


def answer_with_gpt5(question: str):
    docs, metas = retrieve_context(question, TOP_K)
    user_prompt, sources = build_user_prompt(question, docs, metas)
    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    raw = completion.choices[0].message.content.strip()
    if "Sources" not in raw and sources:
        raw += "\n\nSources:\n" + "\n".join(sources[:5])
    return raw


if __name__ == "__main__":
    print("🧠 JosefGPT (Hybrid local retrieval + GPT-5 reasoning). Type 'exit' to quit.")
    while True:
        q = input("\n🧠 Josef> ").strip()
        if q.lower() in ("exit", "quit"):
            break
        print(answer_with_gpt5(q))
