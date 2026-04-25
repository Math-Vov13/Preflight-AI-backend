import datetime
import os
import uuid


def generate_object_name(file_name: str, content_type: str) -> str:
    base_name, ext = os.path.splitext(os.path.basename(file_name or "untitled"))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    kind = (content_type or "application/octet-stream").split("/")[0] or "file"
    safe_base = (base_name or "untitled").replace(" ", "_").lower()
    return f"{kind}/{safe_base}-{timestamp}-{short_id}{ext.lower()}"
