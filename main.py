"""
NLP Translation Pipeline
Auto-detects source language and translates using HuggingFace MarianMT models.
Supports: FR↔EN, DE↔EN, ES↔EN, IT↔EN, PT↔EN

Install dependencies:
    pip install transformers langdetect sentencepiece gradio torch
"""

from langdetect import detect, DetectorFactory
from transformers import pipeline
import gradio as gr

# Ensure consistent language detection results
DetectorFactory.seed = 42

# ─────────────────────────────────────────────
# Supported language pairs and their MarianMT models
# ─────────────────────────────────────────────

LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
}

# Maps (source_lang, target_lang) → HuggingFace model name
MODEL_MAP = {
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("it", "en"): "Helsinki-NLP/opus-mt-it-en",
    ("en", "it"): "Helsinki-NLP/opus-mt-en-it",
    ("pt", "en"): "Helsinki-NLP/opus-mt-ROMANCE-en",
    ("en", "pt"): "Helsinki-NLP/opus-mt-en-ROMANCE",  # covers PT
}

# Cache loaded pipelines to avoid reloading models
_pipeline_cache: dict = {}


def get_translation_pipeline(src: str, tgt: str):
    """Load (or retrieve from cache) the MarianMT pipeline for a language pair."""
    key = (src, tgt)
    if key not in _pipeline_cache:
        model_name = MODEL_MAP.get(key)
        if not model_name:
            raise ValueError(f"No model available for {src} → {tgt}")
        print(f"Loading model: {model_name} ...")
        _pipeline_cache[key] = pipeline("text2text-generation", model=model_name)
    return _pipeline_cache[key]


# ─────────────────────────────────────────────
# Core translation function
# ─────────────────────────────────────────────

def translate(text: str, target_lang: str, auto_detect: bool = True, source_lang: str = "en") -> str:
    """
    Translate `text` into `target_lang`.

    Args:
        text:         Input text to translate.
        target_lang:  2-letter target language code (e.g. 'en', 'fr').
        auto_detect:  If True, detect source language automatically.
        source_lang:  Used only when auto_detect=False.

    Returns:
        Translated string, or an informative error message.
    """
    text = text.strip()
    if not text:
        return "⚠️ Please enter some text to translate."

    # 1. Detect source language
    if auto_detect:
        try:
            detected = detect(text)
            # Normalise: langdetect sometimes returns 'pt' variants
            src = detected[:2].lower()
        except Exception as e:
            return f"❌ Language detection failed: {e}"
    else:
        src = source_lang.lower()

    tgt = target_lang.lower()

    if src == tgt:
        return f"ℹ️ Source and target language are the same ({LANGUAGE_NAMES.get(src, src)}). No translation needed."

    if src not in LANGUAGE_NAMES:
        return (
            f"❌ Detected language '{src}' is not supported.\n"
            f"Supported languages: {', '.join(LANGUAGE_NAMES.values())}"
        )

    if (src, tgt) not in MODEL_MAP:
        return (
            f"❌ Translation from {LANGUAGE_NAMES.get(src, src)} → "
            f"{LANGUAGE_NAMES.get(tgt, tgt)} is not supported.\n"
            f"Try routing through English as an intermediate language."
        )

    # 2. Load model & translate
    try:
        translator = get_translation_pipeline(src, tgt)
        result = translator(text, max_length=512)
        translation = result[0].get("generated_text") or result[0].get("translation_text", "")
        detected_label = f"(auto-detected: {LANGUAGE_NAMES.get(src, src)})" if auto_detect else ""
        return translation
    except Exception as e:
        return f"❌ Translation error: {e}"


def translate_with_info(text: str, target_lang_name: str, auto_detect: bool, source_lang_name: str) -> tuple[str, str]:
    """Wrapper for Gradio: converts display names → codes, returns (translation, info)."""
    name_to_code = {v: k for k, v in LANGUAGE_NAMES.items()}
    tgt = name_to_code.get(target_lang_name, "en")
    src = name_to_code.get(source_lang_name, "en")

    # Detect source for info display
    info = ""
    if auto_detect and text.strip():
        try:
            detected_code = detect(text.strip())[:2].lower()
            detected_name = LANGUAGE_NAMES.get(detected_code, detected_code.upper())
            info = f"🔍 Detected language: **{detected_name}**"
        except Exception:
            info = "🔍 Language detection unavailable"

    translation = translate(text, tgt, auto_detect=auto_detect, source_lang=src)
    return translation, info


# ─────────────────────────────────────────────
# Gradio Interface
# ─────────────────────────────────────────────

SUPPORTED_LANGUAGE_NAMES = list(LANGUAGE_NAMES.values())

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --ink:     #0f0e0c;
    --paper:   #f5f0e8;
    --accent:  #c8522a;
    --mid:     #7a7060;
    --border:  #d6cfc0;
    --warm-bg: #ede8df;
}

body, .gradio-container {
    background: var(--paper) !important;
    font-family: 'DM Mono', monospace !important;
    color: var(--ink) !important;
}

h1.title {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.8rem !important;
    color: var(--ink) !important;
    letter-spacing: -0.02em;
    margin-bottom: 0 !important;
    line-height: 1.1;
}

.subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: var(--mid);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 4px;
}

.divider {
    border: none;
    border-top: 2px solid var(--accent);
    width: 60px;
    margin: 12px 0 24px 0;
}

textarea, .gr-textbox textarea {
    font-family: 'DM Mono', monospace !important;
    background: white !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 4px !important;
    color: var(--ink) !important;
    font-size: 0.95rem !important;
}

textarea:focus {
    border-color: var(--accent) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(200,82,42,0.12) !important;
}

.gr-button-primary, button.primary {
    background: var(--accent) !important;
    border: none !important;
    color: white !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-radius: 4px !important;
    padding: 10px 28px !important;
    transition: opacity 0.15s !important;
}

.gr-button-primary:hover { opacity: 0.85 !important; }

.pair-badge {
    display: inline-block;
    background: var(--warm-bg);
    border: 1px solid var(--border);
    border-radius: 3px;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: var(--mid);
    padding: 2px 8px;
    margin: 2px 3px;
    letter-spacing: 0.05em;
}

label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    color: var(--mid) !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
"""

def build_interface():
    with gr.Blocks(css=CSS, title="Cool Translator — Translation Pipeline") as demo:

        gr.HTML("""
        <div style="padding: 8px 0 4px 0">
          <h1 class="title">Translate to English</h1>
          <div class="subtitle">MarianMT · HuggingFace · Auto-detection</div>
          <hr class="divider">
          <div style="margin-bottom:8px; font-family:'DM Mono',monospace; font-size:0.76rem; color:#7a7060;">
            Supported pairs &nbsp;
            <span class="pair-badge">FR ↔ EN</span>
            <span class="pair-badge">DE ↔ EN</span>
            <span class="pair-badge">ES ↔ EN</span>
            <span class="pair-badge">IT ↔ EN</span>
            <span class="pair-badge">PT ↔ EN</span>
          </div>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                input_text = gr.Textbox(
                    label="Source Text",
                    placeholder="Type or paste text here…",
                    lines=8,
                )
                with gr.Row():
                    auto_detect_toggle = gr.Checkbox(
                        label="Auto-detect language",
                        value=True,
                    )
                    source_lang_dropdown = gr.Dropdown(
                        choices=SUPPORTED_LANGUAGE_NAMES,
                        value="French",
                        label="Source Language (if not auto)",
                        interactive=True,
                        visible=False,
                    )

                target_lang_dropdown = gr.Dropdown(
                    choices=SUPPORTED_LANGUAGE_NAMES,
                    value="English",
                    label="Target Language",
                )

                translate_btn = gr.Button("Translate →", variant="primary")

            with gr.Column(scale=1):
                output_text = gr.Textbox(
                    label="Translation",
                    lines=8,
                    interactive=False,
                )
                detection_info = gr.Markdown("")

        # Show/hide manual source dropdown based on toggle
        auto_detect_toggle.change(
            fn=lambda checked: gr.update(visible=not checked),
            inputs=auto_detect_toggle,
            outputs=source_lang_dropdown,
        )

        translate_btn.click(
            fn=translate_with_info,
            inputs=[input_text, target_lang_dropdown, auto_detect_toggle, source_lang_dropdown],
            outputs=[output_text, detection_info],
        )

        # Also trigger on Enter in the textbox
        input_text.submit(
            fn=translate_with_info,
            inputs=[input_text, target_lang_dropdown, auto_detect_toggle, source_lang_dropdown],
            outputs=[output_text, detection_info],
        )

        gr.HTML("""
        <div style="margin-top:24px; border-top:1px solid #d6cfc0; padding-top:14px;
                    font-family:'DM Mono',monospace; font-size:0.72rem; color:#7a7060;">
            Models load on first use and are cached for subsequent requests.
            First translation may take 20–60 seconds per language pair.
        </div>
        """)

    return demo


# ─────────────────────────────────────────────
# Quick CLI test (no Gradio)
# ─────────────────────────────────────────────

def run_tests():
    test_cases = [
        ("Bonjour, comment allez-vous ?",      "en"),  # FR → EN
        ("Hello, how are you?",                "fr"),  # EN → FR
        ("Guten Morgen, wie geht es Ihnen?",   "en"),  # DE → EN
        ("Good morning, how are you?",         "de"),  # EN → DE
        ("Hola, ¿cómo estás?",                 "en"),  # ES → EN
        ("Hello, how are you?",                "es"),  # EN → ES
        ("Buongiorno, come stai?",             "en"),  # IT → EN
        ("Good morning, how are you?",         "it"),  # EN → IT
        ("Bom dia, como vai você?",            "en"),  # PT → EN
        ("Good morning, how are you?",         "pt"),  # EN → PT
    ]

    print("=" * 60)
    print("TRANSLATION PIPELINE — TEST SUITE")
    print("=" * 60)
    for text, tgt in test_cases:
        result = translate(text, tgt)
        src_name = LANGUAGE_NAMES.get(detect(text)[:2], "?")
        tgt_name = LANGUAGE_NAMES.get(tgt, tgt)
        print(f"\n[{src_name} → {tgt_name}]")
        print(f"  IN : {text}")
        print(f"  OUT: {result}")
    print("\n" + "=" * 60)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        run_tests()
    else:
        app = build_interface()
        app.launch(share=False)
