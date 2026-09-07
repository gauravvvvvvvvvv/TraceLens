from __future__ import annotations

from typing import Protocol

from ddgs import DDGS

from .schemas import SearchResult


class SearchProvider(Protocol):
    def search(self, query: str, max_results: int) -> list[SearchResult]: ...


class DuckDuckGoSearchProvider:
    def search(self, query: str, max_results: int) -> list[SearchResult]:
        rows = DDGS().text(query, max_results=max_results)
        return [
            SearchResult(
                title=row.get("title", "Untitled"),
                url=row.get("href", ""),
                snippet=row.get("body", ""),
                query=query,
            )
            for row in rows
            if row.get("href")
        ]

