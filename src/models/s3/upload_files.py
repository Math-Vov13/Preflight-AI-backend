import io
from typing import Literal

import boto3
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv

from config.config import CONFIG
from models.s3.utils import generate_s3_object_name

load_dotenv()


def upload_files_to_s3(
    file_content: bytes,
    file_name: str,
    content_type: str = "text/plain",
    role: Literal["upload", "generation"] = "upload",
) -> str | None:
    """Upload bytes to S3 and return the generated object key, or None on failure."""
    s3 = boto3.client("s3")
    object_name = generate_s3_object_name(file_name, content_type)
    folder = "user_upload/" if role == "upload" else "generated/"
    extra_args = {"StorageClass": "STANDARD_IA", "ContentType": content_type}

    try:
        s3.upload_fileobj(
            io.BytesIO(file_content),
            CONFIG.S3_BUCKET_NAME,
            folder + object_name,
            ExtraArgs=extra_args,
        )
        print(f"Uploaded to s3://{CONFIG.S3_BUCKET_NAME}/{folder}{object_name}")
        return object_name
    except NoCredentialsError:
        print("AWS credentials not found.")
    except Exception as exc:
        print(f"S3 upload failed: {exc}")
    return None
