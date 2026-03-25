"""Document loader - PDFs via Docling (with PyMuPDF fallback), plain text via LangChain TextLoader."""

import logging
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from src.config import INGEST_DIR

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

_USE_DOCLING: bool | None = None


def _use_docling() -> bool:
    """Check whether to use Docling for PDF parsing (lazy, checked once).

    Docling provides superior table/layout extraction but has heavy dependencies
    (torch, transformers). Set PDF_PARSER=docling to force it; otherwise PyMuPDF
    is used as the lightweight default.
    """
    global _USE_DOCLING
    if _USE_DOCLING is None:
        import os

        if os.environ.get("PDF_PARSER", "").lower() == "docling":
            try:
                import docling.document_converter  # noqa: F401

                _USE_DOCLING = True
                logger.info("Using Docling for PDF parsing (PDF_PARSER=docling).")
            except ImportError:
                logger.warning("PDF_PARSER=docling but Docling not installed, using PyMuPDF.")
                _USE_DOCLING = False
        else:
            _USE_DOCLING = False
    return _USE_DOCLING


def _load_pdf_with_docling(file_path: Path) -> list[Document]:
    """Parse a PDF with Docling and return one Document per page."""
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(file_path))

    if result.status.name not in ("SUCCESS", "PARTIAL_SUCCESS"):
        logger.warning("Docling conversion status %s for %s", result.status, file_path.name)
        return []

    doc = result.document

    # Group content by page number
    pages: dict[int, list[str]] = {}
    for item, _level in doc.iterate_items():
        page_no = 0
        if hasattr(item, "prov") and item.prov:
            page_no = item.prov[0].page_no if item.prov[0].page_no else 0

        text = ""
        if hasattr(item, "text") and item.text:
            text = item.text
        elif hasattr(item, "export_to_markdown"):
            text = item.export_to_markdown()

        if text.strip():
            pages.setdefault(page_no, []).append(text.strip())

    # If page-level grouping didn't work, fall back to full markdown
    if not pages:
        md = doc.export_to_markdown()
        if md.strip():
            return [
                Document(
                    page_content=md,
                    metadata={"filename": file_path.name, "page": 0, "source": "docling"},
                )
            ]
        return []

    documents: list[Document] = []
    for page_no in sorted(pages.keys()):
        content = "\n\n".join(pages[page_no])
        if content.strip():
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "filename": file_path.name,
                        "page": page_no,
                        "source": "docling",
                    },
                )
            )

    logger.info(
        "Docling parsed %d pages from %s (status: %s)",
        len(documents),
        file_path.name,
        result.status.name,
    )
    return documents


def _load_pdf_with_pymupdf(file_path: Path) -> list[Document]:
    """Parse a PDF with PyMuPDF and return one Document per page."""
    import fitz

    pdf = fitz.open(str(file_path))
    documents: list[Document] = []

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text = page.get_text()
        if text.strip():
            documents.append(
                Document(
                    page_content=text.strip(),
                    metadata={
                        "filename": file_path.name,
                        "page": page_num + 1,
                        "source": "pymupdf",
                    },
                )
            )

    pdf.close()
    logger.info("PyMuPDF parsed %d pages from %s", len(documents), file_path.name)
    return documents


def _load_pdf(file_path: Path) -> list[Document]:
    """Load a PDF using Docling if available, otherwise PyMuPDF."""
    if _use_docling():
        try:
            return _load_pdf_with_docling(file_path)
        except Exception:
            logger.warning("Docling failed for %s, falling back to PyMuPDF", file_path.name)
    return _load_pdf_with_pymupdf(file_path)


def _load_text_file(file_path: Path) -> list[Document]:
    """Load a TXT or MD file with LangChain TextLoader."""
    loader = TextLoader(str(file_path))
    docs = loader.load()
    for doc in docs:
        doc.metadata["filename"] = file_path.name
    return docs


def load_documents(directory: Path | None = None) -> list[Document]:
    """Recursively load all supported documents from the given directory."""
    directory = directory or INGEST_DIR
    documents: list[Document] = []

    if not directory.exists():
        logger.warning("Ingest directory does not exist: %s", directory)
        return documents

    for file_path in directory.rglob("*"):
        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        try:
            if ext == ".pdf":
                docs = _load_pdf(file_path)
            else:
                docs = _load_text_file(file_path)

            documents.extend(docs)
            logger.info("Loaded %d pages from %s", len(docs), file_path.name)
        except Exception:
            logger.exception("Failed to load %s", file_path)

    logger.info("Total documents loaded: %d", len(documents))
    return documents
