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
    FONT_SCALE_MAX,
    FONT_SCALE_MIN,
    FONT_SCALE_REFERENCE_WIDTH,
    MIN_SCALED_FONT_PT,
    STYLE_PRESETS,
    font_scale_for_width,
    migrate_style,
    scaled_font_size,
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
    # A compact-preset stored size upgrades to the compact sizes. The sampled
    # values must be genuinely superseded ones (8 was compact's original size
    # two generations back; 9 is the current value and would make this half of
    # the test vacuous).
    compact = migrate_style({"preset": "compact", "original_font_size": 8,
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


def test_migrate_style_carries_an_interim_style_forward():
    """A style saved by the *first* two-line build (10/15) must keep moving to
    the current sizes: the superseded sets hold every generation, so skipping
    a release does not freeze a stored style at an interim value."""
    interim = {
        "preset": "default",
        "original_font_size": 10,
        "translation_font_size": 15,
        "original_color": "#9a9aa8",
    }
    migrated, changed = migrate_style(interim)
    assert changed is True
    assert migrated["translation_font_size"] == DEFAULT_STYLE["translation_font_size"]
    assert migrated["original_font_size"] == DEFAULT_STYLE["original_font_size"]
    assert migrated["original_color"] == DEFAULT_STYLE["original_color"]
    # And the result is stable under a second pass.
    assert migrate_style(migrated) == (migrated, False)


def test_migration_covers_every_preset_that_was_retuned():
    """For each preset, a style carrying that preset's superseded values must
    land exactly on the preset's current values — the guard against retuning a
    preset without extending the superseded table."""
    from subtitle_overlay import (
        _SUPERSEDED_PRESET_VALUES,
        _SUPERSEDED_STYLE_VALUES,
    )

    for preset_key, preset in STYLE_PRESETS.items():
        stored = {"preset": preset_key}
        for field, superseded in _SUPERSEDED_STYLE_VALUES.items():
            values = _SUPERSEDED_PRESET_VALUES.get(preset_key, {}).get(field, superseded)
            # Pick a superseded value that is not already the current one.
            stale = sorted(v for v in values if v != preset[field])
            if stale:
                stored[field] = stale[0]
        if len(stored) == 1:
            continue
        migrated, _ = migrate_style(stored)
        for field in stored:
            if field == "preset":
                continue
            assert migrated[field] == preset[field], (
                f"{preset_key}.{field}: superseded {stored[field]!r} did not "
                f"upgrade to the current {preset[field]!r}"
            )


# ── window-driven font scaling ────────────────────────────────────────────


def test_font_scale_is_one_at_the_reference_width():
    assert font_scale_for_width(FONT_SCALE_REFERENCE_WIDTH) == 1.0
    assert scaled_font_size(DEFAULT_STYLE["translation_font_size"], 1.0) == (
        DEFAULT_STYLE["translation_font_size"]
    )


def test_font_scale_clamps_at_both_ends():
    assert font_scale_for_width(300) == FONT_SCALE_MIN
    assert font_scale_for_width(4000) == FONT_SCALE_MAX
    # The reachable shrink end (the window's minimum width) sits inside the
    # clamp, so the floor is margin rather than a value users hit.
    assert FONT_SCALE_MIN < font_scale_for_width(480) < 1.0


def test_font_scale_handles_a_degenerate_width():
    for width in (0, -1, None):
        assert font_scale_for_width(width) == 1.0


def test_scaled_size_rounds_half_up_not_bankers():
    """Python's round() is banker's rounding: round(34.5) == 34. A size that
    stalls — or drops — as the window grows is exactly the jitter this avoids."""
    assert scaled_font_size(23, 1.5) == 35
    assert round(23 * 1.5) == 34  # what round() would have produced
    assert scaled_font_size(12, 1.5) == 18


def test_scaled_size_never_reaches_zero():
    assert scaled_font_size(1, FONT_SCALE_MIN) == MIN_SCALED_FONT_PT
    assert scaled_font_size(6, FONT_SCALE_MIN) >= MIN_SCALED_FONT_PT


def test_every_preset_keeps_the_two_line_hierarchy_at_every_scale():
    """The multiplicative companion to the check above: scaling both lines by
    the same factor must not let them collapse into each other at any
    reachable window width."""
    factors = [FONT_SCALE_MIN + i * 0.1 for i in range(12)]
    for name, preset in STYLE_PRESETS.items():
        for factor in factors:
            original = scaled_font_size(preset["original_font_size"], factor)
            translation = scaled_font_size(preset["translation_font_size"], factor)
            assert original < translation, (
                f"preset {name} @factor {factor:.1f}: {original}pt is not "
                f"smaller than {translation}pt"
            )


def test_default_style_ships_window_scaling_on():
    assert DEFAULT_STYLE["scale_with_window"] is True
    # Every preset inherits it from the base.
    for name, preset in STYLE_PRESETS.items():
        assert preset.get("scale_with_window") is True, name
