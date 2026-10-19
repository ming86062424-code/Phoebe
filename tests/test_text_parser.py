import pytest

from app.kb.parsers.base import ParseError
from app.kb.parsers.text_parser import MAX_CHARS, TextParser, split_markdown, split_text


def test_limit_is_1500():
    assert MAX_CHARS == 1500


def test_txt_merges_paragraphs_up_to_the_limit():
    paras = ["a" * 600, "b" * 600, "c" * 600, "d" * 10]
    chunks = split_text("\n\n".join(paras))
    assert chunks == ["a" * 600 + "\n\n" + "b" * 600, "c" * 600 + "\n\n" + "d" * 10]
    assert all(len(c) <= 1500 for c in chunks)


def test_txt_hard_splits_a_long_paragraph():
    chunks = split_text("short\n\n" + "x" * 3200 + "\n\ntail")
    assert chunks == ["short", "x" * 1500, "x" * 1500, "x" * 200 + "\n\ntail"]


def test_txt_blank_lines_with_spaces_and_crlf_separate_paragraphs():
    assert split_text("one\r\n  \r\ntwo\r\nstill two") == ["one\n\ntwo\nstill two"]


def test_txt_empty_gives_no_chunks():
    assert split_text(" \n\n \n") == []


def test_md_splits_at_headings_and_keeps_them():
    md = "intro line\n\n# Title\n\nbody one\n\n## Sub\n\nbody two\n"
    assert split_markdown(md) == ["intro line", "# Title\n\nbody one", "## Sub\n\nbody two"]


def test_md_hash_inside_code_fence_is_not_a_heading():
    md = "# A\n\n```bash\n# just a comment\necho hi\n```\n"
    assert split_markdown(md) == ["# A\n\n```bash\n# just a comment\necho hi\n```"]


def test_md_long_section_is_split_by_paragraph_and_keeps_heading():
    md = "# Big\n\n" + "\n\n".join(["p" * 700] * 4)
    chunks = split_markdown(md)
    assert len(chunks) > 1
    assert all(c.startswith("# Big\n\n") for c in chunks)
    assert all(len(c) <= 1500 for c in chunks)
    assert "".join(c.replace("# Big\n\n", "") for c in chunks).count("p") == 2800


def _parse(tmp_path, name, data: bytes):
    path = tmp_path / name
    path.write_bytes(data)
    out = tmp_path / "out"
    out.mkdir()
    progress = []
    chunks = TextParser().parse(str(path), out, lambda d, t: progress.append((d, t)))
    return chunks, progress


def test_parser_reads_txt_with_bom_as_page_zero(tmp_path):
    chunks, progress = _parse(tmp_path, "a.txt", "\ufeff第一段\n\n第二段".encode("utf-8"))
    assert [(c.page, c.text, c.kind, c.images, c.page_image) for c in chunks] == [
        (0, "第一段\n\n第二段", "text", [], None)
    ]
    assert progress[-1] == (1, 1)


def test_parser_uses_markdown_rules_for_md(tmp_path):
    chunks, _ = _parse(tmp_path, "a.MD", "# 一\n\n甲\n\n# 二\n\n乙".encode("utf-8"))
    assert [c.text for c in chunks] == ["# 一\n\n甲", "# 二\n\n乙"]
    assert all(c.page == 0 for c in chunks)


def test_parser_rejects_non_utf8(tmp_path):
    with pytest.raises(ParseError, match="無法讀取檔案內容"):
        _parse(tmp_path, "a.txt", "中文".encode("big5"))


def test_parser_metadata():
    parser = TextParser()
    assert parser.id == "text"
    assert parser.availability() == (True, "")
    parser.release()


def test_md_fence_closes_only_on_a_matching_marker():
    md = "# A\nx\n~~~\n```py\n# not\n~~~\n# B\ny"
    assert split_markdown(md) == ["# A\nx\n~~~\n```py\n# not\n~~~", "# B\ny"]


def test_md_longer_fence_needs_a_long_enough_closer():
    md = "# A\n````\n```\n# not\n````\n# B"
    assert split_markdown(md) == ["# A\n````\n```\n# not\n````", "# B"]


def test_md_only_real_atx_headings_split():
    md = "# A\n#hashtag\n#!/bin/sh\n####### seven\n###### six\nz\n#\nend"
    assert split_markdown(md) == ["# A\n#hashtag\n#!/bin/sh\n####### seven", "###### six\nz", "#\nend"]
