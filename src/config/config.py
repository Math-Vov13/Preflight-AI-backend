import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    S3_BUCKET_NAME: str = "the-universal-agent"
    SUPABASE_STORAGE_BUCKET: str = os.getenv("SUPABASE_STORAGE_BUCKET", "chat-uploads")

CONFIG = Config()