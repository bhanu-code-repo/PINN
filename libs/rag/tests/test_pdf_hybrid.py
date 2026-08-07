"""Tests for hybrid PDF conversion utilities."""

import pytest
from rag.indexing.pdf import classify_pages, get_page_count, pdf_to_markdown


def _create_simple_pdf(path, pages=1):
    """Create a simple text-only PDF for testing."""
    try:
        import pymupdf
    except ImportError:
        pytest.skip("pymupdf not installed")

    doc = pymupdf.open()
    for i in range(pages):
        page = doc.new_page()
        # insert_text handles newlines poorly; insert line by line
        lines = [
            f"Page {i + 1} — Physics-Informed Neural Networks",
            "",
            "This is a test page with enough text content to be classified as simple.",
            "It contains multiple sentences about physics-informed neural networks.",
            "The Burgers equation is a standard benchmark for PINN implementations.",
            "Shock formation at low viscosity is the primary difficulty for standard PINNs.",
            "Residual-based adaptive refinement concentrates points near the shock.",
            "Causal training weights time slices so early residuals converge first.",
            "Loss weighting with lambda 10 to 100 enforces boundary conditions properly.",
            "The tanh activation is smooth and infinitely differentiable for autograd.",
        ]
        y = 72
        for line in lines:
            page.insert_text((72, y), line, fontsize=11)
            y += 16
    doc.save(str(path))
    doc.close()


class TestClassifyPages:
    def test_simple_text_page(self, tmp_path):
        pdf = tmp_path / "simple.pdf"
        _create_simple_pdf(pdf, pages=3)

        results = classify_pages(pdf)
        assert len(results) == 3
        for r in results:
            assert r["complexity"] == "simple"
            assert r["text_tokens"] > 30
            assert r["image_area_ratio"] == 0.0

    def test_page_num_zero_indexed(self, tmp_path):
        pdf = tmp_path / "multi.pdf"
        _create_simple_pdf(pdf, pages=2)

        results = classify_pages(pdf)
        assert results[0]["page_num"] == 0
        assert results[1]["page_num"] == 1


class TestGetPageCount:
    def test_page_count(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        _create_simple_pdf(pdf, pages=5)
        assert get_page_count(pdf) == 5


class TestPdfToMarkdown:
    def test_basic_conversion(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        _create_simple_pdf(pdf, pages=1)
        md = pdf_to_markdown(pdf)
        assert len(md) > 0
        # Should contain at least some of the text we inserted
        md_lower = md.lower()
        assert "page" in md_lower or "neural" in md_lower or "pinn" in md_lower
