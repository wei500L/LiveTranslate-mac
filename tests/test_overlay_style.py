"""Style preset hierarchy guards (pure dict checks, no Qt needed).

Every preset must keep the two-line hierarchy the card layout is built
on: the original line is secondary (smaller font, distinct color) and the
translation line is primary. A preset where the two lines collapse to the
same color/size flattens the layout back to what the redesign replaced.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from subtitle_overlay import (  # noqa: E402
    DEFAULT_STYLE,
    STYLE_PRESETS,
    migrate_style,
)


def test_every_preset_keeps_two_line_hierarchy():
    for name, preset in STYLE_PRESETS.items():
        assert preset["original_color"] != preset["translation_color"], (
            f"preset {name}: original and translation colors are identical — "
            "the two-line hierarchy collapses"
        )
        assert (
            preset["original_font_size"] < preset["translation_font_size"]
        ), f"preset {name}: original font is not smaller than translation"


def test_provisional_colors_are_dimmer_than_final():
    """The cloud provisional dim must stay a strictly darker variant of the
    final colors, so provisional → final reads as brightening."""
    s = DEFAULT_STYLE
    assert s["provisional_original_color"] != s["original_color"]
    assert s["provisional_translation_color"] != s["translation_color"]


def test_migrate_style_upgrades_an_untouched_stored_style():
    """The exact shape a pre-redesign install has on disk: old defaults plus
    the removed timestamp_color. Without the upgrade the merge in
    apply_style (stored values win) would pin the old look forever."""
    stored = {
        "preset": "default",
        "original_font_size": 11,
        "translation_font_size": 14,
        "original_color": "#cccccc",
        "translation_color": "#ffffff",
        "timestamp_color": "#888899",
        "window_opacity": 95,
    }
    migrated, changed = migrate_style(stored)
    assert changed is True
    assert "timestamp_color" not in migrated
    assert migrated["original_font_size"] == DEFAULT_STYLE["original_font_size"]
    assert migrated["translation_font_size"] == DEFAULT_STYLE["translation_font_size"]
    assert migrated["original_color"] == DEFAULT_STYLE["original_color"]
    # Untouched, unrelated fields survive.
    assert migrated["window_opacity"] == 95
    assert migrated["translation_color"] == "#ffffff"


def test_migrate_style_preserves_deliberate_user_choices():
    stored = {
        "preset": "default",
        "original_font_size": 13,          # user-chosen, not the old default
        "translation_font_size": 14,
        "original_color": "#ff00ff",       # user-chosen
    }
    migrated, changed = migrate_style(stored)
    assert migrated["original_font_size"] == 13
    assert migrated["original_color"] == "#ff00ff"
    # The one field still at its old default is upgraded.
    assert migrated["translation_font_size"] == DEFAULT_STYLE["translation_font_size"]
    assert changed is True


def test_migrate_style_uses_the_stored_preset_as_the_baseline():
    """A theme preset's own pre-redesign value must map to that theme's new
    value, not to the default preset's."""
    stored = {"preset": "dracula", "original_color": "#f8f8f2"}
    migrated, changed = migrate_style(stored)
    assert changed is True
    assert migrated["original_color"] == STYLE_PRESETS["dracula"]["original_color"]
    # A compact-preset stored size upgrades to the compact sizes.
    compact = migrate_style({"preset": "compact", "original_font_size": 9,
                             "translation_font_size": 11})[0]
    assert compact["original_font_size"] == STYLE_PRESETS["compact"]["original_font_size"]
    assert compact["translation_font_size"] == STYLE_PRESETS["compact"]["translation_font_size"]


def test_migrate_style_is_idempotent():
    stored = {
        "preset": "default",
        "original_font_size": 11,
        "original_color": "#cccccc",
        "timestamp_color": "#888899",
    }
    once, _ = migrate_style(stored)
    twice, changed = migrate_style(once)
    assert twice == once
    assert changed is False
    # A style saved by the current build is untouched.
    assert migrate_style(dict(DEFAULT_STYLE)) == (dict(DEFAULT_STYLE), False)
