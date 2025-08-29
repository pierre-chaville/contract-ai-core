from contract_ai_core.schema import split_text_into_paragraphs


def test_split_text_into_paragraphs_basic_merge_and_cleanup():
    text = (
        "Header line without period\n"
        "continues here on next line\n\n"
        "* * *\n"  # star rule separator should be removed
        "Table | row\n"  # pipes should become tabs (treated as non-merge)
        "Another line.\n"
        "- list item should start new paragraph\n"
        "final line"
    )

    paras = split_text_into_paragraphs(text)
    # Expect merged first two lines into one paragraph
    assert paras[0].text.startswith("Header line without period continues here")
    # Ensure star rule removed and paragraphs created for following lines
    assert any("Another line." in p.text for p in paras)
    # Ensure at least 3 paragraphs resulted
    assert len(paras) >= 3


