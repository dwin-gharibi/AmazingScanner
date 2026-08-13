from __future__ import annotations

import gradio as gr

__all__ = ["build_theme", "CSS", "hero", "stat_card", "note"]


def build_theme() -> gr.Theme:
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=("Inter", "ui-sans-serif", "system-ui", "-apple-system",
              "Segoe UI", "Roboto", "Helvetica Neue", "Arial", "sans-serif"),
        font_mono=("JetBrains Mono", "ui-monospace", "SFMono-Regular",
                   "Menlo", "Consolas", "monospace"),
    ).set(
        body_background_fill="*neutral_50",
        body_background_fill_dark="*neutral_950",
        block_radius="14px",
        block_shadow="0 1px 3px rgba(15,23,42,.08), 0 8px 24px -12px rgba(15,23,42,.18)",
        button_primary_background_fill="linear-gradient(96deg, *primary_600, *primary_500)",
        button_primary_background_fill_hover="linear-gradient(96deg, *primary_700, *primary_600)",
        button_primary_text_color="white",
        block_title_text_weight="600",
        section_header_text_weight="600",
    )


CSS = """
:root { --ds-radius: 14px; }

.ds-hero {
  border-radius: var(--ds-radius);
  padding: 26px 30px;
  margin-bottom: 6px;
  background:
    radial-gradient(1200px 220px at 8% -30%, rgba(129,140,248,.35), transparent 60%),
    linear-gradient(103deg, #1e1b4b 0%, #312e81 45%, #4338ca 100%);
  color: #eef2ff;
  box-shadow: 0 10px 30px -18px rgba(49,46,129,.9);
}
/* Gradio's base stylesheet colours headings and paragraphs, and its rules are
   more specific than a bare element selector -- without !important the hero
   renders dark-on-dark. */
.ds-hero h1 {
  margin: 0 0 6px 0; font-size: 1.7rem; font-weight: 700; letter-spacing: -.02em;
  color: #ffffff !important;
}
.ds-hero p {
  margin: 0; font-size: .95rem; line-height: 1.5;
  color: rgba(238,242,255,.90) !important;
}
.ds-hero .ds-chips { margin-top: 14px; display: flex; gap: 8px; flex-wrap: wrap; }
.ds-hero .ds-chip {
  font-size: .74rem; padding: 4px 11px; border-radius: 999px;
  color: #eef2ff !important;
  background: rgba(238,242,255,.16); border: 1px solid rgba(238,242,255,.30);
}

.ds-stats { display: flex; gap: 10px; flex-wrap: wrap; margin: 2px 0 4px 0; }
.ds-stat {
  flex: 1 1 118px; padding: 11px 13px; border-radius: 11px;
  background: var(--block-background-fill);
  border: 1px solid var(--border-color-primary);
}
.ds-stat .k { font-size: .68rem; text-transform: uppercase; letter-spacing: .07em;
              opacity: .62; font-weight: 600; }
.ds-stat .v { font-size: 1.22rem; font-weight: 700; margin-top: 3px; line-height: 1.15; }
.ds-stat .s { font-size: .7rem; opacity: .58; margin-top: 1px; }
.ds-stat.good .v { color: #059669; }
.ds-stat.warn .v { color: #d97706; }
.ds-stat.bad  .v { color: #dc2626; }

.ds-note {
  border-left: 3px solid var(--color-accent);
  background: var(--block-background-fill);
  padding: 10px 14px; border-radius: 0 10px 10px 0; font-size: .87rem; line-height: 1.5;
}
.ds-note b { color: var(--color-accent); }

footer { display: none !important; }
.gradio-container { max-width: 1420px !important; }
.ds-tight .gap { gap: 8px !important; }
table.ds-table { width: 100%; border-collapse: collapse; font-size: .84rem; }
table.ds-table th, table.ds-table td {
  border-bottom: 1px solid var(--border-color-primary); padding: 6px 9px; text-align: left;
}
table.ds-table th { font-weight: 600; opacity: .75; }
table.ds-table tr:hover td { background: var(--background-fill-secondary); }
"""


def hero(title: str, subtitle: str, chips: list[str] | None = None) -> str:
    chip_html = ""
    if chips:
        chip_html = ('<div class="ds-chips">'
                     + "".join(f'<span class="ds-chip">{c}</span>' for c in chips)
                     + "</div>")
    return f'<div class="ds-hero"><h1>{title}</h1><p>{subtitle}</p>{chip_html}</div>'


def stat_card(key: str, value: str, sub: str = "", tone: str = "") -> str:
    cls = f"ds-stat {tone}".strip()
    sub_html = f'<div class="s">{sub}</div>' if sub else ""
    return f'<div class="{cls}"><div class="k">{key}</div><div class="v">{value}</div>{sub_html}</div>'


def note(text: str) -> str:
    return f'<div class="ds-note">{text}</div>'
