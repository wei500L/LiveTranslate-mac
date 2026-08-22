"""Application-wide visual language for LiveTranslate.

The optional qdarktheme package provides platform-aware palette defaults.  The
project stylesheet then adds the glass workspace treatment and consistent
control metrics used by every window.
"""

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication


COLORS = {
    "canvas": "#0b0f14",
    "surface": "#151c24",
    "surface_alt": "#1b2530",
    "surface_hover": "#302a23",
    "line": "#453f38",
    "text": "#f4efe7",
    "muted": "#ada79d",
    "accent": "#e7b96f",
    "accent_strong": "#c98b42",
    "success": "#9caf91",
    "warning": "#efc46b",
    "danger": "#e58b83",
}


APP_QSS = f"""
* {{
    font-family: "Helvetica Neue", "Segoe UI", sans-serif;
    font-size: 13px;
    color: {COLORS['text']};
}}
QMainWindow, QDialog {{ background: {COLORS['canvas']}; }}
QWidget#settingsNavigation, QWidget#glassPanel {{ background: {COLORS['canvas']}; }}
QFrame#glassPanel, QWidget#glassPanel {{
    background: rgba(21, 28, 36, 238);
    border: 1px solid {COLORS['line']};
    border-radius: 12px;
}}
QLabel#pageTitle {{ font-size: 20px; font-weight: 700; color: {COLORS['text']}; }}
QLabel#pageSubtitle {{ color: {COLORS['muted']}; font-size: 12px; }}
QLabel#statusLabel {{ color: {COLORS['accent']}; font-size: 12px; }}
QGroupBox {{
    background: rgba(21, 28, 36, 220);
    border: 1px solid {COLORS['line']};
    border-radius: 10px;
    margin-top: 14px;
    padding: 18px 14px 14px 14px;
    font-weight: 650;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 12px; top: 4px;
    padding: 0 6px; color: {COLORS['accent']};
}}
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: #111820; border: 1px solid #3d484d; border-radius: 8px;
    padding: 7px 10px; selection-background-color: {COLORS['accent_strong']};
    min-height: 18px;
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus,
QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {COLORS['accent']}; }}
QComboBox::drop-down {{ border: 0; width: 28px; }}
QComboBox::down-arrow {{ image: none; border: 0; width: 0; height: 0; }}
QComboBox QAbstractItemView {{
    background: {COLORS['surface_alt']}; border: 1px solid {COLORS['line']};
    selection-background-color: {COLORS['accent_strong']};
}}
QPushButton {{
    background: #252a2d; border: 1px solid #4a4c48; border-radius: 8px;
    padding: 7px 13px; min-height: 18px; font-weight: 600;
}}
QPushButton:hover {{ background: {COLORS['surface_hover']}; border-color: #b18a58; }}
QPushButton:pressed {{ background: #1e1d1b; padding-top: 8px; padding-bottom: 6px; }}
QPushButton:disabled {{ color: #6e746f; background: #1a1d1e; border-color: #2e3332; }}
QPushButton#primaryButton {{ background: {COLORS['accent_strong']}; color: #17120b; border-color: {COLORS['accent']}; }}
QPushButton#dangerButton {{ background: #542a32; color: #ffd9dc; border-color: #9a4d58; }}
QCheckBox, QRadioButton {{ spacing: 8px; color: {COLORS['muted']}; }}
QCheckBox:hover, QRadioButton:hover {{ color: {COLORS['text']}; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 16px; height: 16px; border: 1px solid #65706d; border-radius: 4px; background: #171c1d; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {COLORS['accent']}; }}
QCheckBox::indicator:checked {{ background: {COLORS['accent_strong']}; border-color: {COLORS['accent']}; }}
QTabWidget::pane {{ border: 0; }}
QScrollArea {{ border: 0; background: transparent; }}
QScrollBar:vertical {{ width: 9px; background: transparent; margin: 3px 0; }}
QScrollBar::handle:vertical {{ background: #4e504b; border-radius: 4px; min-height: 32px; }}
QScrollBar::handle:vertical:hover {{ background: #8c704b; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ height: 9px; background: transparent; margin: 0 3px; }}
QScrollBar::handle:horizontal {{ background: #4e504b; border-radius: 4px; min-width: 32px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QSlider::groove:horizontal {{ height: 5px; background: #2b3a45; border-radius: 3px; }}
QSlider::handle:horizontal {{ width: 16px; margin: -6px 0; background: {COLORS['accent']}; border-radius: 8px; }}
QListWidget {{ background: #111820; border: 1px solid #344047; border-radius: 9px; padding: 5px; outline: 0; }}
QListWidget::item {{ padding: 10px 11px; margin: 1px 0; border-radius: 7px; color: {COLORS['muted']}; }}
QListWidget::item:hover {{ background: #292d2c; color: {COLORS['text']}; }}
QListWidget::item:selected {{ background: #4b3925; color: {COLORS['text']}; border-left: 3px solid {COLORS['accent']}; padding-left: 8px; }}
QListWidget#settingsNavigation {{ background: #101417; border: 1px solid #303839; padding: 8px 6px; }}
QListWidget#settingsNavigation::item {{ min-height: 22px; padding: 11px 12px; margin: 2px 0; }}
QListWidget#settingsNavigation::item:selected {{ background: #3b3023; border-left: 3px solid {COLORS['accent']}; padding-left: 9px; }}
QWidget#chatMessage {{ background: rgba(255, 255, 255, 6); border: 1px solid rgba(164, 143, 111, 36); border-radius: 8px; }}
QWidget#chatMessage:hover {{ background: rgba(231, 185, 111, 15); border-color: rgba(231, 185, 111, 72); }}
QProgressBar {{ background: #18232b; border: 1px solid #30404e; border-radius: 5px; text-align: center; color: {COLORS['muted']}; }}
QProgressBar::chunk {{ background: {COLORS['accent_strong']}; border-radius: 4px; }}
QTextEdit#logView {{ background: #101416; color: #d8e4e8; border: 1px solid #343d3d; border-radius: 9px; padding: 10px; selection-background-color: #5a452d; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 6px; padding: 5px; }}
QToolButton:hover {{ background: #2b2b28; border-color: #5a503f; }}
QHeaderView::section {{ background: #202526; color: {COLORS['muted']}; border: 0; border-bottom: 1px solid #3b4241; padding: 8px; font-weight: 600; }}
QToolTip {{ background: #1b2530; color: {COLORS['text']}; border: 1px solid #4b6572; padding: 5px; }}
"""


def apply_theme(app: QApplication) -> None:
    """Apply the optional mature dark palette plus LiveTranslate styling."""
    try:
        import qdarktheme

        qdarktheme.setup_theme("dark", custom_colors={"primary": COLORS["accent"]})
    except Exception:
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(COLORS["canvas"]))
        palette.setColor(QPalette.ColorRole.Base, QColor("#111820"))
        palette.setColor(QPalette.ColorRole.Text, QColor(COLORS["text"]))
        palette.setColor(QPalette.ColorRole.Button, QColor("#252a2d"))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(COLORS["text"]))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(COLORS["accent_strong"]))
        app.setPalette(palette)
    app.setStyleSheet(app.styleSheet() + APP_QSS)
