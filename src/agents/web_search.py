"""Web search fallback using DuckDuckGo + Crawl4AI for content extraction."""

import asyncio
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    WEB_SEARCH_CONTENT_LIMIT,
    WEB_SEARCH_ENABLED,
    WEB_SEARCH_MAX_PAGES,
    WEB_SEARCH_TIMEOUT,
)
from src.utils import StageTimer

logger = logging.getLogger(__name__)

_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


def search_web(query: str) -> list[dict]:
    """Search the web and crawl results for clean content.

    Uses DuckDuckGo for URL discovery and Crawl4AI for page extraction.
    Returns list of {"text": str, "url": str, "score": float, "metadata": dict}.
    """
    if not WEB_SEARCH_ENABLED:
        logger.info("Web search disabled.")
        return []

    with StageTimer("web_search"):
        try:
            return asyncio.run(_search_and_crawl(query))
        except Exception:
            logger.exception("Web search failed")
            return []


async def _search_and_crawl(query: str) -> list[dict]:
    """Find URLs via DuckDuckGo, then crawl them with Crawl4AI."""
    from duckduckgo_search import DDGS

    # step 1: find relevant URLs
    with DDGS() as ddgs:
        search_results = list(ddgs.text(query, max_results=WEB_SEARCH_MAX_PAGES))

    if not search_results:
        logger.info("DuckDuckGo returned no results.")
        return []

    urls = [r["href"] for r in search_results]
    logger.info("Found %d URLs, crawling with Crawl4AI...", len(urls))

    # step 2: crawl pages concurrently
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

    browser_cfg = BrowserConfig(headless=True)
    crawl_cfg = CrawlerRunConfig(
        word_count_threshold=50,
        page_timeout=WEB_SEARCH_TIMEOUT * 1000,
    )

    results: list[dict] = []

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        # crawl all pages concurrently to save time
        tasks = [_crawl_page(crawler, url, crawl_cfg) for url in urls]
        page_results = await asyncio.gather(*tasks, return_exceptions=True)

        for url, page_result in zip(urls, page_results, strict=False):
            if isinstance(page_result, BaseException):
                logger.warning("Crawl error for %s: %s", url, page_result)
                continue
            if page_result:
                results.extend(page_result)

    logger.info("Web search produced %d chunks from %d pages.", len(results), len(urls))
    return results


async def _crawl_page(crawler, url: str, config) -> list[dict]:
    """Crawl a single page and split into chunks."""
    try:
        result = await asyncio.wait_for(
            crawler.arun(url=url, config=config),
            timeout=WEB_SEARCH_TIMEOUT,
        )
    except TimeoutError:
        logger.warning("Timeout crawling %s", url)
        return []

    if not result.success or not result.markdown_v2:
        # fall back to basic markdown if v2 not available
        text = getattr(result, "markdown", "") or ""
    else:
        text = result.markdown_v2.raw_markdown

    if not text.strip():
        return []

    # trim to configured limit before splitting
    text = text[:WEB_SEARCH_CONTENT_LIMIT]
    pieces = _splitter.split_text(text)

    return [
        {
            "text": piece,
            "url": url,
            "score": 1.0,
            "metadata": {
                "filename": f"web:{url}",
                "chunk_index": i,
                "page_number": 0,
            },
        }
        for i, piece in enumerate(pieces)
    ]
