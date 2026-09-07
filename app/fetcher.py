from __future__ import annotations

import hashlib
import io
import ipaddress
import socket
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

from .config import settings
from .schemas import RetrievedDocument


class SafeDocumentFetcher:
    allowed_content_types = ("text/html", "text/plain", "application/pdf")

    def fetch(self, url: str) -> RetrievedDocument:
        self._validate_url(url)
        with httpx.Client(
            timeout=settings.request_timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": "TraceLens/1.0 research-agent"},
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            self._validate_url(str(response.url))
            content_type = response.headers.get("content-type", "").split(";")[0].lower()
            if content_type not in self.allowed_content_types:
                raise ValueError(f"Unsupported content type: {content_type}")
            content = response.content
            if len(content) > settings.max_document_bytes:
                raise ValueError("Document exceeds configured size limit")

        if content_type == "application/pdf":
            reader = PdfReader(io.BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            title = urlparse(str(response.url)).path.rsplit("/", 1)[-1] or "PDF document"
        elif content_type == "text/html":
            soup = BeautifulSoup(content, "html.parser")
            for node in soup(["script", "style", "noscript", "nav", "footer"]):
                node.decompose()
            title = soup.title.get_text(" ", strip=True) if soup.title else str(response.url)
            text = soup.get_text(" ", strip=True)
        else:
            title = str(response.url)
            text = content.decode(response.encoding or "utf-8", errors="replace")

        return RetrievedDocument(
            url=url,
            final_url=str(response.url),
            title=title[:500],
            text=text[:120_000],
            content_type=content_type,
            content_hash=hashlib.sha256(content).hexdigest(),
        )

    @staticmethod
    def _validate_url(url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Only public HTTP(S) URLs are supported")
        if parsed.hostname.lower() == "localhost":
            raise ValueError("Local addresses are blocked")
        try:
            addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, None)}
        except socket.gaierror as exc:
            raise ValueError("Hostname could not be resolved") from exc
        for address in addresses:
            ip = ipaddress.ip_address(address)
            if not ip.is_global:
                raise ValueError("Private, loopback, and link-local addresses are blocked")

