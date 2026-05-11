"""Asset upload and download for local and S3 storage backends."""

import os
import shutil
from urllib.parse import urlparse

from michelangelo.lib.model_manager.constants import StorageType


def _parse_s3(path: str) -> tuple[str, str]:
    """Return (bucket, key) from an s3://bucket/key URL."""
    parsed = urlparse(path)
    return parsed.netloc, parsed.path.lstrip("/")


def download_assets(src: str, des: str, source_type: str) -> None:
    """Download assets from *src* to *des*.

    Args:
        src: Source path. For S3 use ``s3://bucket/key``.
        des: Local destination path.
        source_type: One of ``StorageType.LOCAL`` or ``StorageType.S3``.
    """
    if source_type == StorageType.LOCAL:
        if src != des:
            if os.path.isdir(src):
                shutil.copytree(src, des, dirs_exist_ok=True)
            else:
                os.makedirs(os.path.dirname(des), exist_ok=True)
                shutil.copy(src, des)
    elif source_type == StorageType.S3:
        try:
            import boto3
        except ImportError as e:
            raise ImportError("boto3 is required for S3 storage. Install with: pip install boto3") from e
        bucket, key = _parse_s3(src)
        s3 = boto3.client("s3")
        if key.endswith("/") or not os.path.splitext(key)[1]:
            # Directory-style download
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=key):
                for obj in page.get("Contents", []):
                    rel = obj["Key"][len(key):]
                    local_path = os.path.join(des, rel)
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    s3.download_file(bucket, obj["Key"], local_path)
        else:
            os.makedirs(os.path.dirname(des) if not os.path.isdir(des) else des, exist_ok=True)
            dest_file = des if not os.path.isdir(des) else os.path.join(des, os.path.basename(key))
            s3.download_file(bucket, key, dest_file)
    else:
        raise ValueError(f"Unsupported source_type: {source_type!r}. Use StorageType.LOCAL or StorageType.S3.")


def upload_assets(src: str, des: str, source_type: str) -> None:
    """Upload assets from local *src* to *des*.

    Args:
        src: Local source path (file or directory).
        des: Destination path. For S3 use ``s3://bucket/key``.
        source_type: One of ``StorageType.LOCAL`` or ``StorageType.S3``.
    """
    if source_type == StorageType.LOCAL:
        if src != des:
            if os.path.isdir(src):
                shutil.copytree(src, des, dirs_exist_ok=True)
            else:
                os.makedirs(os.path.dirname(des), exist_ok=True)
                shutil.copy(src, des)
    elif source_type == StorageType.S3:
        try:
            import boto3
        except ImportError as e:
            raise ImportError("boto3 is required for S3 storage. Install with: pip install boto3") from e
        bucket, prefix = _parse_s3(des)
        s3 = boto3.client("s3")
        if os.path.isdir(src):
            for root, _, files in os.walk(src):
                for fname in files:
                    local_path = os.path.join(root, fname)
                    rel = os.path.relpath(local_path, src)
                    s3_key = f"{prefix}/{rel}".lstrip("/")
                    s3.upload_file(local_path, bucket, s3_key)
        else:
            s3_key = prefix if not prefix.endswith("/") else f"{prefix}{os.path.basename(src)}"
            s3.upload_file(src, bucket, s3_key)
    else:
        raise ValueError(f"Unsupported source_type: {source_type!r}. Use StorageType.LOCAL or StorageType.S3.")
