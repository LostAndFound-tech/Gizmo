"""
main.py — FastAPI entry point.

Endpoints:
  GET  /              — status
  GET  /health        — health + RAG stats
  POST /chat          — streaming chat with agent + RAG
  POST /ingest/text   — ingest raw text
  POST /ingest/file   — upload and ingest a file
  POST /ingest/url    — ingest from a URL
  GET  /collections   — list RAG collections
  DELETE /collection  — wipe a collection
  DELETE /source      — remove chunks by source name
  DELETE /session     — clear conversation history
"""

import os
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from core.agent import agent
from memory.archiver import start_archiver
from core.rag import rag
from core.ingest import ingest_text, ingest_file, ingest_url
from memory.history import get_session, clear_session

load_dotenv()

app = FastAPI(title="Chatbot API", version="0.2.0")

@app.on_event("startup")
async def startup():
    import asyncio
    from core.llm import llm
    loop = asyncio.get_event_loop()
    start_archiver(llm, loop)
    print("[App] Archiver started")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    use_rag: bool = True
    collection: str = "main"
    context: Optional[dict] = None  # e.g. {"current_host": "Oren", "mood": "focused", "recent": "..."}

class IngestTextRequest(BaseModel):
    text: str
    source: str = "manual"
    collection: str = "main"
    chunk_size: int = 400
    overlap: int = 50

class IngestURLRequest(BaseModel):
    url: str
    collection: str = "main"
    chunk_size: int = 400
    overlap: int = 50

class DeleteCollectionRequest(BaseModel):
    collection: str

class DeleteSourceRequest(BaseModel):
    source: str
    collection: str = "main"

class SessionClearRequest(BaseModel):
    session_id: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "running", "docs": "/docs", "health": "/health"}


@app.get("/health")
async def health():
    collections = rag.list_collections()
    return {
        "status": "ok",
        "collections": collections,
        "active_collection": rag.collection.name,
        "docs_in_active": rag.count,
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    history = get_session(session_id)

    # Point RAG at the requested collection for this chat
    rag.use_collection(req.collection)

    async def generate():
        yield f"[session:{session_id}]\n"
        if req.context.get("debug") if req.context else False:
            from memory.overview import get_overview
            from core.rag import rag
            overview = await get_overview(session_id, history, __import__('core.llm', fromlist=['llm']).llm)
            turn_count = len(history) // 2
            yield f"[debug:turns={turn_count}]\n"
            yield f"[debug:rag_docs={rag.count}]\n"
            yield f"[debug:overview={overview or 'none yet'}]\n"
            yield f"[debug:context={req.context}]\n---\n"
        async for token in agent.run(
            req.message,
            history,
            session_id=session_id,
            use_rag=req.use_rag,
            context=req.context,
        ):
            yield token

    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/ingest/text")
async def ingest_text_route(req: IngestTextRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")
    count = ingest_text(req.text, source=req.source, collection=req.collection,
                        chunk_size=req.chunk_size, overlap=req.overlap)
    return {"chunks_added": count, "collection": req.collection, "source": req.source}


@app.post("/ingest/file")
async def ingest_file_route(
    file: UploadFile = File(...),
    collection: str = Form("main"),
    chunk_size: int = Form(400),
    overlap: int = Form(50),
):
    supported = {".txt", ".md", ".pdf"}
    suffix = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if suffix not in supported:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}. Use: {supported}")

    # Save to temp file then ingest
    import tempfile, shutil
    from pathlib import Path
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        count = ingest_file(tmp_path, collection=collection, chunk_size=chunk_size, overlap=overlap)
        # Rename source label to original filename
        rag.use_collection(collection)
    finally:
        os.unlink(tmp_path)

    return {"chunks_added": count, "collection": collection, "filename": file.filename}


@app.post("/ingest/url")
async def ingest_url_route(req: IngestURLRequest):
    try:
        count = await ingest_url(req.url, collection=req.collection,
                                 chunk_size=req.chunk_size, overlap=req.overlap)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"chunks_added": count, "collection": req.collection, "url": req.url}


@app.get("/collections")
async def list_collections():
    collections = rag.list_collections()
    stats = {}
    for name in collections:
        rag.use_collection(name)
        stats[name] = rag.count
    rag.use_collection("main")
    return {"collections": stats}


@app.delete("/collection")
async def delete_collection(req: DeleteCollectionRequest):
    rag.delete_collection(req.collection)
    return {"deleted": req.collection}


@app.delete("/source")
async def delete_source(req: DeleteSourceRequest):
    rag.use_collection(req.collection)
    rag.delete_by_source(req.source)
    return {"deleted_source": req.source, "collection": req.collection}


@app.delete("/session")
async def clear(req: SessionClearRequest):
    clear_session(req.session_id)
    return {"cleared": req.session_id}


@app.get("/corrections")
async def corrections_log(limit: int = 20):
    """Review all logged corrections."""
    from core.rag import RAGStore
    store = RAGStore(collection_name="corrections")
    if store.count == 0:
        return {"corrections": [], "total": 0}
    raw = store.collection.get(limit=limit, include=["documents", "metadatas"])
    entries = [
        {"text": raw["documents"][i], "metadata": raw["metadatas"][i]}
        for i in range(len(raw["documents"]))
    ]
    return {"total": store.count, "corrections": entries}


@app.get("/wellness")
async def wellness_log(limit: int = 20, q: str = None):
    """Query the wellness log. Optionally filter by search query."""
    from core.rag import RAGStore
    store = RAGStore(collection_name="wellness")
    if store.count == 0:
        return {"entries": [], "total": 0}
    if q:
        results = store.retrieve(q, n_results=min(limit, store.count))
        entries = [{"text": r["text"], "metadata": r["metadata"]} for r in results]
    else:
        raw = store.collection.get(limit=limit, include=["documents", "metadatas"])
        entries = [
            {"text": raw["documents"][i], "metadata": raw["metadatas"][i]}
            for i in range(len(raw["documents"]))
        ]
    return {"total": store.count, "entries": entries}


@app.get("/rag/peek")
async def rag_peek(limit: int = 10, collection: str = "main"):
    """Show the first N documents in a collection with their metadata."""
    rag.use_collection(collection)
    results = rag.collection.get(limit=limit, include=["documents", "metadatas"])
    docs = []
    for i, doc in enumerate(results["documents"]):
        docs.append({
            "id": results["ids"][i],
            "metadata": results["metadatas"][i],
            "preview": doc[:200] + "..." if len(doc) > 200 else doc,
        })
    return {"collection": collection, "total_docs": rag.count, "showing": len(docs), "documents": docs}


@app.get("/rag/search")
async def rag_search(q: str, n: int = 4, collection: str = "main"):
    """Run a raw RAG retrieval query and see what comes back."""
    rag.use_collection(collection)
    if rag.count == 0:
        return {"results": [], "message": f"Collection '{collection}' is empty"}
    docs = rag.retrieve(q, n_results=min(n, rag.count))
    return {"query": q, "collection": collection, "results": docs}


@app.get("/rag/collections")
async def rag_collections():
    """List all collections and their document counts."""
    names = rag.list_collections()
    stats = {}
    for name in names:
        rag.use_collection(name)
        stats[name] = rag.count
    rag.use_collection("main")
    return {"collections": stats}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", 8000)),
        reload=os.getenv("DEBUG", "true").lower() == "true",
    )