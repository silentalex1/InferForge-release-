from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class StorageBackend(ABC):
    @abstractmethod
    def upload_file(self, local_path: Path, remote_key: str) -> str:
        pass

    @abstractmethod
    def download_file(self, remote_key: str, local_path: Path) -> Path:
        pass

    @abstractmethod
    def delete_file(self, remote_key: str) -> bool:
        pass

    @abstractmethod
    def file_exists(self, remote_key: str) -> bool:
        pass

    @abstractmethod
    def get_file_url(self, remote_key: str, expires_in: int = 3600) -> str:
        pass

    @abstractmethod
    def list_files(self, prefix: str = "") -> list[str]:
        pass

    @abstractmethod
    def get_file_size(self, remote_key: str) -> int:
        pass

    def stream_upload(self, local_path: Path, remote_key: str) -> str:
        return self.upload_file(local_path, remote_key)

    def stream_download(self, remote_key: str, local_path: Path) -> Path:
        return self.download_file(remote_key, local_path)
