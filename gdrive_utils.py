import os
import json
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SERVICE_ACCOUNT_FILE = "service-account.json"

def get_drive_service():
    """Authenticate and return a Google Drive service instance."""
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    service = build("drive", "v3", credentials=creds)
    return service

def download_pdfs_from_drive(folder_id, local_dir="research_papers"):
    """Download all PDFs from a Google Drive folder."""
    os.makedirs(local_dir, exist_ok=True)
    service = get_drive_service()

    query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])

    for f in files:
        file_id, name = f["id"], f["name"]
        request = service.files().get_media(fileId=file_id)
        file_path = os.path.join(local_dir, name)
        with io.FileIO(file_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
    return local_dir
