import uuid
from pathlib import Path
from fastapi import UploadFile

ALLOWED_VIDEO_EXTENSIONS = [".mp4", ".webm", ".mov", ".mkv"]


# Stream an UploadFile to a temporary path on disk.Returns the path where the file was saved.
async def save_upload_to_temp(upload: UploadFile, temp_dir: Path):
    temp_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(upload.filename).suffix or ".tmp"
    temp_path = temp_dir / f"{uuid.uuid4().hex}{suffix}"

    with open(temp_path, "wb") as out_file:
        while chunk := await upload.read(1024 * 1024):  # 1 MB chunks
            out_file.write(chunk)
    return temp_path


def cleanup_temp_files(paths: list[Path]):
    for p in paths:
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass


def validate_upload_extension(filename: str):
    suffix = Path(filename).suffix.lower()
    return suffix in ALLOWED_VIDEO_EXTENSIONS
