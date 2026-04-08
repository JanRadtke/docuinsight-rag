"""
Unit Tests — Pure logic, no API keys or ingested data required.
"""

import os

from ingest import clean_id, encode_image


class TestCleanId:
    def test_alphanumeric_passthrough(self):
        assert clean_id("hello123") == "hello123"

    def test_special_chars_replaced(self):
        assert clean_id("file (1).pdf") == "file__1__pdf"

    def test_spaces_replaced(self):
        assert clean_id("page 1 chunk 2") == "page_1_chunk_2"

    def test_empty_string(self):
        assert clean_id("") == ""

    def test_unicode(self):
        # Unicode letters are alphanumeric in Python
        assert clean_id("Über_Größe") == "Über_Größe"


class TestEncodeImage:
    def test_roundtrip(self):
        import base64
        original = b"\x89PNG\r\n\x1a\n\x00\x00"
        encoded = encode_image(original)
        decoded = base64.b64decode(encoded)
        assert decoded == original

    def test_returns_string(self):
        assert isinstance(encode_image(b"test"), str)


class TestCleanFilename:
    """Tests the standalone clean_filename() in retriever.py."""

    def setup_method(self):
        from retriever import clean_filename
        self.clean_filename = clean_filename

    def test_removes_pdf_extension(self):
        assert self.clean_filename("report.pdf") == "report"

    def test_replaces_underscores(self):
        assert self.clean_filename("my_report_2024.pdf") == "my report 2024"

    def test_truncates_long_names(self):
        long_name = "a" * 50 + ".pdf"
        result = self.clean_filename(long_name)
        assert len(result) == 40
        assert result.endswith("...")


class TestExtractFilenameFromPrompt:
    """Tests filename extraction without needing actual PDFs in input/."""

    def setup_method(self):
        """Patch INPUT_DIR to a temp directory with fake PDFs."""
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        # Create fake PDF files
        for name in ["lopez_otin_2023.pdf", "chen_2025_immunosenescence.pdf", "kaeberlein_rapamycin.pdf"]:
            open(os.path.join(self.tmpdir, name), "w").close()

        import retriever
        self._original_input_dir = retriever.INPUT_DIR
        retriever.INPUT_DIR = self.tmpdir

    def teardown_method(self):
        import retriever
        retriever.INPUT_DIR = self._original_input_dir
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_matches_author_name(self):
        from retriever import extract_filename_from_prompt
        result = extract_filename_from_prompt("Summarize the Lopez-Otin paper")
        assert result == "lopez_otin_2023.pdf"

    def test_matches_topic(self):
        from retriever import extract_filename_from_prompt
        result = extract_filename_from_prompt("What does the rapamycin study say?")
        assert result == "kaeberlein_rapamycin.pdf"

    def test_no_match_returns_none(self):
        from retriever import extract_filename_from_prompt
        result = extract_filename_from_prompt("What is the meaning of life?")
        assert result is None

    def test_multiple_files(self):
        from retriever import extract_filenames_from_prompt
        result = extract_filenames_from_prompt("Compare Lopez-Otin and Chen")
        assert result is not None
        assert len(result) == 2


class TestTokenize:
    """Tests BM25 tokenizer without NLTK dependency."""

    def setup_method(self):
        from retriever import Retriever
        self.retriever = Retriever.__new__(Retriever)
        self.retriever._nltk_available = False
        self.retriever._stemmers = {}
        self.retriever._stopword_sets = {}

    def test_basic_tokenization(self):
        tokens = self.retriever._tokenize("Hello World Test")
        assert tokens == ["hello", "world", "test"]

    def test_lowercases(self):
        tokens = self.retriever._tokenize("UPPER Case MiXeD")
        assert all(t == t.lower() for t in tokens)

    def test_empty_string(self):
        tokens = self.retriever._tokenize("")
        assert tokens == []
