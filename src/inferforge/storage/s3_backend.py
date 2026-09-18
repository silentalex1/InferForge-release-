from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

from inferforge.core.config import get_storage_config
from inferforge.storage.base import StorageBackend


class S3StorageBackend(StorageBackend):
    def __init__(
        self,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        bucket_name: str = "inferforge-models",
        region: str = "us-east-1",
    ) -> None:
        self.bucket_name = bucket_name
        self.region = region

        session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )

        self.s3_client = session.client(
            "s3",
            endpoint_url=endpoint_url,
        )

        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError:
            try:
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            except ClientError as e:
                raise RuntimeError(f"Failed to create bucket: {e}")

    def upload_file(self, local_path: Path, remote_key: str) -> str:
        config = get_storage_config()
        chunk_size = config.get("chunk_size_mb", 100) * 1024 * 1024

        file_size = local_path.stat().st_size
        if file_size > chunk_size:
            self._upload_large_file(local_path, remote_key, chunk_size)
        else:
            self.s3_client.upload_file(str(local_path), self.bucket_name, remote_key)
        return f"s3://{self.bucket_name}/{remote_key}"

    def _upload_large_file(self, local_path: Path, remote_key: str, chunk_size: int) -> None:

        part_number = 1
        parts = []

        upload_id = self.s3_client.create_multipart_upload(
            Bucket=self.bucket_name, Key=remote_key
        )["UploadId"]

        try:
            with local_path.open("rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break

                    part = self.s3_client.upload_part(
                        Bucket=self.bucket_name,
                        Key=remote_key,
                        PartNumber=part_number,
                        UploadId=upload_id,
                        Body=chunk,
                    )
                    parts.append({"PartNumber": part_number, "ETag": part["ETag"]})
                    part_number += 1

            self.s3_client.complete_multipart_upload(
                Bucket=self.bucket_name,
                Key=remote_key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
            )
        except Exception:
            self.s3_client.abort_multipart_upload(
                Bucket=self.bucket_name, Key=remote_key, UploadId=upload_id
            )
            raise

    def download_file(self, remote_key: str, local_path: Path) -> Path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.s3_client.download_file(self.bucket_name, remote_key, str(local_path))
        return local_path

    def delete_file(self, remote_key: str) -> bool:
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=remote_key)
            return True
        except ClientError:
            return False

    def file_exists(self, remote_key: str) -> bool:
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=remote_key)
            return True
        except ClientError:
            return False

    def get_file_url(self, remote_key: str, expires_in: int = 3600) -> str:
        try:
            return self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": remote_key},
                ExpiresIn=expires_in,
            )
        except ClientError as e:
            raise RuntimeError(f"Failed to generate presigned URL: {e}")

    def list_files(self, prefix: str = "") -> list[str]:
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=prefix
            )
            return [obj["Key"] for obj in response.get("Contents", [])]
        except ClientError:
            return []

    def get_file_size(self, remote_key: str) -> int:
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=remote_key)
            return response.get("ContentLength", 0)
        except ClientError:
            return 0

    def upload_fileobj(self, file_obj: Any, remote_key: str) -> str:
        self.s3_client.upload_fileobj(file_obj, self.bucket_name, remote_key)
        return f"s3://{self.bucket_name}/{remote_key}"

    def download_fileobj(self, remote_key: str, file_obj: Any) -> None:
        self.s3_client.download_fileobj(self.bucket_name, remote_key, file_obj)

    def get_etag(self, remote_key: str) -> str:
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=remote_key)
            return response.get("ETag", "").strip('"')
        except ClientError:
            return ""

    def calculate_local_etag(self, local_path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
        md5 = hashlib.md5()
        with local_path.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                md5.update(chunk)
        return md5.hexdigest()
