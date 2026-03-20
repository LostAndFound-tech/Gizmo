"""
persistence/gdrive.py
Google Drive sync for ChromaDB persistence.

On Lambda, /tmp is ephemeral — it wipes between cold starts.
This module pulls the ChromaDB directory from Google Drive
before processing and pushes it back after.

For a personal project this is simple and effective.
The ChromaDB directory is zipped, stored as a single file
on Drive, and unzipped on pull.

Requirements:
    pip install google-api-python-client google-auth

Setup:
    1. Create a Google Cloud project
    2. Enable the Google Drive API
    3. Create a Service Account and download the JSON key
    4. Share your Drive folder with the service account email
    5. Set GOOGLE_SERVICE_ACCOUNT_JSON env var to the key file path
       OR set GOOGLE_SERVICE_ACCOUNT_KEY to the JSON content directly

Environment variables:
    GOOGLE_SERVICE_ACCOUNT_JSON  — path to service account key file
    GOOGLE_SERVICE_ACCOUNT_KEY   — service account key JSON as string
    GDRIVE_FOLDER_ID             — Google Drive folder ID to sync to
    CHROMA_PERSIST_DIR           — local ChromaDB directory path
"""

import io
import json
import os
import zipfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GDRIVE_FOLDER_ID  = os.getenv("GDRIVE_FOLDER_ID", "")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "/tmp/chroma")
CHROMA_ZIP_NAME   = "chroma_db.zip"

SCOPES = ["https://www.googleapis.com/auth/drive"]


def _get_credentials():
    """
    Build Google credentials from service account.
    Tries JSON string first (Lambda env var), then file path.
    """
    from google.oauth2 import service_account

    key_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_KEY")
    if key_json:
        info = json.loads(key_json)
        return service_account.Credentials.from_service_account_info(
            info, scopes=SCOPES
        )

    key_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if key_file and Path(key_file).exists():
        return service_account.Credentials.from_service_account_file(
            key_file, scopes=SCOPES
        )

    raise RuntimeError(
        "No Google credentials found. Set GOOGLE_SERVICE_ACCOUNT_KEY "
        "or GOOGLE_SERVICE_ACCOUNT_JSON."
    )


def _get_drive_service():
    """Build and return a Drive API service."""
    from googleapiclient.discovery import build
    creds = _get_credentials()
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _find_file(service, filename: str) -> str | None:
    """Find a file in the configured Drive folder. Returns file ID or None."""
    query = (
        f"name='{filename}' and "
        f"'{GDRIVE_FOLDER_ID}' in parents and "
        f"trashed=false"
    )
    results = service.files().list(
        q=query,
        fields="files(id, name)",
    ).execute()
    files = results.get("files", [])
    return files[0]["id"] if files else None


def _zip_chroma() -> bytes:
    """Zip the ChromaDB directory into memory and return bytes."""
    buf = io.BytesIO()
    chroma_path = Path(CHROMA_PERSIST_DIR)

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in chroma_path.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(chroma_path.parent))

    buf.seek(0)
    return buf.read()


def _unzip_chroma(data: bytes) -> None:
    """Unzip ChromaDB bytes into the persist directory."""
    chroma_path = Path(CHROMA_PERSIST_DIR)
    chroma_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(chroma_path.parent)

    print(f"[Drive] Unzipped ChromaDB to {chroma_path.parent}")


async def pull_chroma() -> None:
    """
    Pull ChromaDB zip from Google Drive and unzip to CHROMA_PERSIST_DIR.
    If no file exists on Drive yet, silently continues with empty DB.
    """
    import asyncio

    def _pull():
        service = _get_drive_service()
        file_id = _find_file(service, CHROMA_ZIP_NAME)

        if not file_id:
            print("[Drive] No ChromaDB on Drive yet — starting fresh")
            Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
            return

        from googleapiclient.http import MediaIoBaseDownload
        buf = io.BytesIO()
        request = service.files().get_media(fileId=file_id)
        downloader = MediaIoBaseDownload(buf, request)

        done = False
        while not done:
            _, done = downloader.next_chunk()

        _unzip_chroma(buf.getvalue())
        print(f"[Drive] Pulled ChromaDB ({len(buf.getvalue())} bytes)")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _pull)


async def push_chroma() -> None:
    """
    Zip ChromaDB and push to Google Drive.
    Updates existing file or creates new one.
    """
    import asyncio
    from pathlib import Path

    chroma_path = Path(CHROMA_PERSIST_DIR)
    if not chroma_path.exists():
        print("[Drive] ChromaDB path doesn't exist — nothing to push")
        return

    def _push():
        from googleapiclient.http import MediaIoBaseUpload

        service = _get_drive_service()
        data = _zip_chroma()
        buf = io.BytesIO(data)

        file_metadata = {
            "name": CHROMA_ZIP_NAME,
            "parents": [GDRIVE_FOLDER_ID],
        }
        media = MediaIoBaseUpload(buf, mimetype="application/zip")

        existing_id = _find_file(service, CHROMA_ZIP_NAME)

        if existing_id:
            # Update existing file
            service.files().update(
                fileId=existing_id,
                media_body=media,
            ).execute()
            print(f"[Drive] Updated ChromaDB on Drive ({len(data)} bytes)")
        else:
            # Create new file
            service.files().create(
                body=file_metadata,
                media_body=media,
                fields="id",
            ).execute()
            print(f"[Drive] Created ChromaDB on Drive ({len(data)} bytes)")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _push)
