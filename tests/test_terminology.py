from terminology import parse_glossary

from glossary_builtin import (
    BUILTIN_GLOSSARY_SELECTIONS,
    DISCRETE_MATH,
    LINEAR_ALGEBRA,
    SELECTION_BOTH,
    SELECTION_DISCRETE_MATH,
    SELECTION_LINEAR_ALGEBRA,
    SELECTION_OFF,
    SINGLE_CHAR_TRANSLATIONS,
    builtin_glossary_text,
    merged_glossary_text,
)
from terminology import normalize_term, short_cjk_translations


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


def test_single_cjk_translation_never_highlights_on_the_translation_line():
    glossary = parse_glossary("строка => 行\nстолбец => 列")
    # 行列式 starts with both single characters; neither may match.
    translated = glossary.highlight_translation("行列式与基本概念")
    assert 'class="term-highlight"' not in translated
    # The entry itself survives: the Russian side still highlights.
    original = glossary.highlight_original("строка матрицы")
    assert original.count('class="term-highlight"') == 1
    assert ">строка</span>" in original


def test_single_cjk_original_is_also_inert():
    glossary = parse_glossary("数 => число")
    rendered = glossary.highlight_original("学习数学")
    assert 'class="term-highlight"' not in rendered


def test_short_cjk_translations_reports_user_entries():
    assert short_cjk_translations("x => 基\ny => 矩阵\nz => 秩") == ["基", "秩"]
    assert short_cjk_translations("матрица => 矩阵") == []
    assert short_cjk_translations("") == []


def test_yo_is_folded_to_e_for_matching():
    assert normalize_term("путём") == normalize_term("путем")
    glossary = parse_glossary("путем => 路径")
    assert 'class="term-highlight"' in glossary.highlight_original("путём")


def test_builtin_glossaries_parse_without_duplicates():
    for text in (LINEAR_ALGEBRA, DISCRETE_MATH):
        glossary = parse_glossary(text)
        # 300–400 entries per course, word-form expansion included.
        assert 300 <= len(glossary.entries) <= 400
        originals = [e.normalized_original for e in glossary.entries]
        assert len(originals) == len(set(originals))
    # Entries shared by both courses must agree on the translation, so the
    # "both" selection resolves duplicates without changing behaviour.
    la = {
        e.normalized_original: e.translation
        for e in parse_glossary(LINEAR_ALGEBRA).entries
    }
    dm = {
        e.normalized_original: e.translation
        for e in parse_glossary(DISCRETE_MATH).entries
    }
    for key in la.keys() & dm.keys():
        assert la[key] == dm[key]


def test_builtin_single_char_translations_are_registered_and_russian_side_only():
    used = set()
    for text in (LINEAR_ALGEBRA, DISCRETE_MATH):
        glossary = parse_glossary(text)
        inert = {
            e.translation
            for e in glossary.entries
            if len(e.normalized_translation) == 1
        }
        # Every single-character translation is a deliberate, documented
        # Russian-side-only term — no accidental ones slip in.
        assert inert <= SINGLE_CHAR_TRANSLATIONS
        used |= inert
        # They are excluded from translation-line matching.
        for entry, _ in glossary._translation_terms:
            assert len(entry.normalized_translation) != 1
    # The registry carries no dead entries.
    assert used == set(SINGLE_CHAR_TRANSLATIONS)
    # Spot-check the rule over a real built-in table: 行 never lights up
    # inside 基本概念, while определитель highlights 行列式 as one whole
    # term (the longest match wins, not the single characters inside it).
    la = parse_glossary(LINEAR_ALGEBRA)
    assert 'class="term-highlight"' not in la.highlight_translation("基本概念")
    assert ">行列式</span>" in la.highlight_translation("行列式的值")


def test_builtin_glossaries_cover_common_inflections():
    la = parse_glossary(LINEAR_ALGEBRA)
    rendered = la.highlight_original(
        "Умножим матрицу на вектор, найдём ранг и базис."
    )
    assert rendered.count('class="term-highlight"') == 4
    # Expansion-era terms: phrases and case forms added later still match.
    rendered = la.highlight_original(
        "Приведём матрицу к диагональному виду и разложим вектор "
        "по элементам ортонормированного базиса."
    )
    assert rendered.count('class="term-highlight"') == 4
    dm = parse_glossary(DISCRETE_MATH)
    rendered = dm.highlight_original(
        "Рассмотрим граф, дерево и множество вершин."
    )
    assert rendered.count('class="term-highlight"') == 4
    rendered = dm.highlight_original(
        "Матрица смежности и полустепень захода задают ориентированный граф."
    )
    assert rendered.count('class="term-highlight"') == 3


def test_merged_glossary_user_entries_override_builtin():
    merged = parse_glossary(
        merged_glossary_text(
            SELECTION_LINEAR_ALGEBRA, "матрица => 方阵\nслед матрицы => 矩阵的迹"
        )
    )
    by_original = {e.normalized_original: e for e in merged.entries}
    # The user's translation wins on the same normalized original...
    assert by_original["матрица"].translation == "方阵"
    # ...while inflected forms the user did not list keep the built-in one
    # (matching is literal, so an override never stems the other cases).
    assert by_original["матрицу"].translation == "矩阵"


def test_merged_glossary_text_selections():
    assert merged_glossary_text(SELECTION_OFF, "x => y") == "x => y"
    assert merged_glossary_text(None, "") == ""
    assert merged_glossary_text(SELECTION_DISCRETE_MATH, "") == DISCRETE_MATH.strip()
    both = merged_glossary_text(SELECTION_BOTH, None)
    assert both.startswith(LINEAR_ALGEBRA.strip())
    assert both.endswith(DISCRETE_MATH.strip())
    for key in BUILTIN_GLOSSARY_SELECTIONS:
        # Every selection yields text that parses (or empty for "off").
        parse_glossary(builtin_glossary_text(key))
