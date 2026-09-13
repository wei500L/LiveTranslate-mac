from terminology import parse_glossary


def test_glossary_normalizes_stress_and_highlights_both_languages():
    glossary = parse_glossary("последовательность => 数列")
    original = glossary.highlight_original("Это последова́тельность.")
    translated = glossary.highlight_translation("Получаем числовую数列.")
    assert 'class="term-highlight"' in original
    assert "последова́тельность" in original
    assert 'class="term-highlight"' in translated


def test_glossary_uses_word_boundaries_and_last_duplicate_wins():
    glossary = parse_glossary(
        "# comment\nряд => row\nпоследовательность => sequence\nпоследовательность => 数列"
    )
    rendered = glossary.highlight_original("рядом последовательность")
    assert rendered.count("term-highlight") == 1
    assert glossary.entries[1].translation == "数列"


def test_glossary_escapes_user_text():
    glossary = parse_glossary("x => <b>bad</b>")
    rendered = glossary.highlight_translation("<script>x</script>")
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_overlapping_translation_terms_render_one_valid_span():
    glossary = parse_glossary("sequence => 数列\nsequence limit => 数列极限")
    rendered = glossary.highlight_translation("学习数列极限与数列")
    assert rendered.count('class="term-highlight"') == 2
    assert ">数列极限</span>" in rendered
    assert ">数列</span>" in rendered
    assert "学习<span" in rendered


def test_latin_translation_term_does_not_match_inside_a_word():
    glossary = parse_glossary("ряд => row")
    rendered = glossary.highlight_translation("row rowing row")
    assert rendered.count('class="term-highlight"') == 2
