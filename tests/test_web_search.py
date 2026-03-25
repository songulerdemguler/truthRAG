"""Tests for web search (DuckDuckGo + Crawl4AI)."""

from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.web_search import search_web


class TestSearchWeb:
    def test_disabled_returns_empty(self):
        with patch("src.agents.web_search.WEB_SEARCH_ENABLED", False):
            result = search_web("test query")
            assert result == []

    def test_ddg_failure_returns_empty(self):
        """DuckDuckGo raising should not crash the pipeline."""
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.side_effect = Exception("rate limited")

        mock_module = MagicMock()
        mock_module.DDGS.return_value = mock_ddgs

        with patch.dict("sys.modules", {"duckduckgo_search": mock_module}):
            result = search_web("test query")
            assert result == []

    def test_no_results_returns_empty(self):
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = []

        mock_module = MagicMock()
        mock_module.DDGS.return_value = mock_ddgs

        with patch.dict("sys.modules", {"duckduckgo_search": mock_module}):
            result = search_web("obscure query no results")
            assert result == []

    def test_successful_search_returns_chunks(self):
        """Full flow: DDG finds URLs, Crawl4AI extracts content, chunks are returned."""
        # mock DuckDuckGo
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = [
            {"href": "https://example.com/page1"},
            {"href": "https://example.com/page2"},
        ]
        mock_ddg_module = MagicMock()
        mock_ddg_module.DDGS.return_value = mock_ddgs

        # mock Crawl4AI
        mock_crawl_result = MagicMock()
        mock_crawl_result.success = True
        mock_crawl_result.markdown_v2 = None  # force fallback to .markdown
        mock_crawl_result.markdown = "This is some extracted content from the page."

        mock_crawler = AsyncMock()
        mock_crawler.arun = AsyncMock(return_value=mock_crawl_result)
        mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
        mock_crawler.__aexit__ = AsyncMock(return_value=False)

        mock_crawl_module = MagicMock()
        mock_crawl_module.AsyncWebCrawler.return_value = mock_crawler
        mock_crawl_module.BrowserConfig.return_value = MagicMock()
        mock_crawl_module.CrawlerRunConfig.return_value = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "duckduckgo_search": mock_ddg_module,
                "crawl4ai": mock_crawl_module,
            },
        ):
            result = search_web("python RAG tutorial")

        assert len(result) >= 2  # at least one chunk per page
        for chunk in result:
            assert "text" in chunk
            assert "url" in chunk
            assert chunk["score"] == 1.0
            assert "metadata" in chunk
            assert chunk["metadata"]["filename"].startswith("web:")
