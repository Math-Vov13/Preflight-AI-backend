from typing import Literal

from config.config import CONFIG
from models.supabase_storage.client import client
from models.supabase_storage.utils import generate_object_name


def upload_file_to_supabase(
    file_content: bytes,
    file_name: str,
    content_type: str = "application/octet-stream",
    role: Literal["upload", "generation"] = "upload",
) -> str | None:
    """Upload bytes to the Supabase storage bucket. Returns the object key, or None on failure."""
    if client is None:
        print("Supabase storage skipped: client not configured.", flush=True)
        return None

    folder = "user_upload" if role == "upload" else "generated"
    object_name = generate_object_name(file_name, content_type)
    object_path = f"{folder}/{object_name}"

    try:
        client.storage.from_(CONFIG.SUPABASE_STORAGE_BUCKET).upload(
            path=object_path,
            file=file_content,
            file_options={"content-type": content_type, "upsert": "true"},
        )
        print(f"Uploaded to supabase://{CONFIG.SUPABASE_STORAGE_BUCKET}/{object_path}")
        return object_path
    except Exception as exc:
        print(f"Supabase storage upload failed: {exc}", flush=True)
        return None


def create_signed_url(object_path: str, expires_in: int = 3600) -> str | None:
    if client is None:
        return None
    try:
        res = client.storage.from_(CONFIG.SUPABASE_STORAGE_BUCKET).create_signed_url(
            object_path, expires_in
        )
        return res.get("signedURL") or res.get("signed_url")
    except Exception as exc:
        print(f"Supabase signed URL failed: {exc}", flush=True)
        return None
