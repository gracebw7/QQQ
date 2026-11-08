
import os
import re
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit, urljoin

import httpx

CACHE_DIR = Path("data/raw")
MAX_SIZE = 50 * 1024 * 1024  # 50 MB
MAX_TIME = 30  # seconds

ALLOWED_HOSTS = {"arxiv.org", "export.arxiv.org"}


def valid_arxiv_url(url: str) -> bool:
    parsed = urlsplit(url)
    return (
        parsed.scheme == "https"
        and parsed.hostname in ALLOWED_HOSTS
        and parsed.port in (None, 443)
        and parsed.username is None
        and parsed.password is None
        and parsed.path.startswith("/pdf/")
    )


def valid_pdf(path: Path) -> bool:
    if not path.is_file():
        return False

    if not 8 <= path.stat().st_size <= MAX_SIZE:
        return False

    with path.open("rb") as file:
        return file.read(5) == b"%PDF-"


def download_paper(paper: dict) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    document_id = paper["document_id"]

    if not re.fullmatch(r"[A-Za-z0-9._-]+", document_id):
        return {"status": "failed", "error": "Invalid document ID"}

    destination = CACHE_DIR / f"{document_id}.pdf"

    if valid_pdf(destination):
        return {"status": "cached", "error": None}

    destination.unlink(missing_ok=True)

    url = paper["pdf_url"]

    if not valid_arxiv_url(url):
        return {"status": "failed", "error": "Invalid arXiv PDF URL"}

    temp_path = None
    start = time.monotonic()

    try:
        with tempfile.NamedTemporaryFile(
            dir=CACHE_DIR, suffix=".tmp", delete=False
        ) as temp:
            temp_path = Path(temp.name)

        with httpx.Client(
            timeout=10.0, follow_redirects=False
        ) as client:
            for _ in range(4):
                if not valid_arxiv_url(url):
                    raise ValueError("Unsafe redirect destination")

                if time.monotonic() - start > MAX_TIME:
                    raise TimeoutError("Download deadline exceeded")

                with client.stream("GET", url) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise ValueError("Missing redirect URL")
                        url = urljoin(url, location)
                        continue

                    response.raise_for_status()

                    with temp_path.open("wb") as file:
                        size = 0

                        for chunk in response.iter_bytes():
                            size += len(chunk)

                            if size > MAX_SIZE:
                                raise ValueError("PDF exceeds 50 MB")

                            if time.monotonic() - start > MAX_TIME:
                                raise TimeoutError("Download deadline exceeded")

                            file.write(chunk)

                    if not valid_pdf(temp_path):
                        raise ValueError("Response is not a valid PDF")

                    os.replace(temp_path, destination)

                    return {"status": "downloaded", "error": None}

            raise ValueError("Too many redirects")

    except Exception as error:
        return {"status": "failed", "error": str(error)}

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
