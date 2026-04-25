import pathlib
import tempfile
import uuid

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader

from models.s3.upload_files import upload_files_to_s3
from models.vc_qdrant.utils import (
    add_documents,
    count_collection,
    delete_items,
    ensure_collection,
    get_collection_items,
    list_collection_names,
    openai_ef,
)

router = APIRouter()

_splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=100)
_pdf_reader = PDFReader()


def _extract_chunks(content: bytes, content_type: str) -> list[str]:
    if content_type in ("text/plain", "application/json"):
        text = content.decode("utf-8", errors="replace")
        return _splitter.split_text(text)
    if content_type == "application/pdf":
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = pathlib.Path(tmp.name)
        try:
            documents = _pdf_reader.load_data(file=tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        full_text = "\n\n".join(doc.text for doc in documents if doc.text)
        return _splitter.split_text(full_text)
    return []


@router.get("/", description="List all collections")
async def get_collections():
    try:
        names = list_collection_names()
        return JSONResponse(
            content={"collections": names, "length": len(names)},
            status_code=200,
        )
    except Exception as exc:
        print("Error listing collections:", exc)
        return JSONResponse(content={"collections": [], "length": 0}, status_code=503)


@router.get("/{collection_id}/items", description="List items in a collection")
async def list_items(collection_id: str):
    try:
        items = get_collection_items(collection_id)
        return {
            "collection_id": collection_id,
            "items": items,
            "length": count_collection(collection_id),
        }
    except Exception as exc:
        print("Error retrieving collection items:", exc)
        return JSONResponse(
            content={"collection_id": collection_id, "items": [], "length": 0},
            status_code=404,
        )


@router.post("/{collection_id}/items", description="Upload a document to S3 and index it in the collection")
async def add_item_to_collection(collection_id: str, item: UploadFile = File(...)):
    content = await item.read()
    await item.close()

    content_type = item.content_type or "application/octet-stream"
    chunks = _extract_chunks(content, content_type)
    if not chunks:
        return JSONResponse(
            content={"error": f"Unsupported file type: {content_type}"},
            status_code=400,
        )

    cleaned = [c.strip() for c in chunks if c and c.strip()]
    if not cleaned:
        return JSONResponse(content={"error": "No extractable content"}, status_code=400)

    s3_object_name = upload_files_to_s3(
        file_content=content,
        file_name=item.filename or "untitled",
        content_type=content_type,
        role="upload",
    )
    if s3_object_name is None:
        return JSONResponse(content={"error": "S3 upload failed"}, status_code=502)

    embeddings = openai_ef(input=cleaned)

    if not ensure_collection(collection_id):
        return JSONResponse(content={"error": "Collection unavailable"}, status_code=503)

    add_documents(
        collection_name=collection_id,
        ids=[str(uuid.uuid4()) for _ in cleaned],
        documents=cleaned,
        embeddings=embeddings,
        metadatas=[{"source": item.filename, "s3_key": s3_object_name} for _ in cleaned],
    )

    return JSONResponse(
        content={
            "status": "added",
            "collection_id": collection_id,
            "filename": item.filename,
            "s3_key": s3_object_name,
            "chunks": len(cleaned),
        },
        status_code=201,
    )


@router.delete("/{collection_id}/items/{item_id}", description="Remove a chunk from a collection")
async def delete_item_from_collection(collection_id: str, item_id: str):
    try:
        delete_items(collection_id, [item_id])
    except Exception as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=404)
    return JSONResponse(
        content={"status": "deleted", "collection_id": collection_id, "item_id": item_id},
        status_code=200,
    )
