"""Check the configured Gemini API with a synthetic transcript."""

from src.gemini_preprocessing import GeminiError, gemini_config, prepare_gemini_sources
from src.segmentation import Segment


def main() -> None:
    try:
        config = gemini_config()
        if config is None:
            raise GeminiError("Isi GEMINI_API_KEY di .env dan pastikan GEMINI_ENABLED=true.")
        sources = prepare_gemini_sources([
            Segment(0, 4, "Nah, kita gunakan range(5) untuk membuat perulangan di Python."),
            Segment(4, 8, "Perulangan menghasilkan angka 0 sampai 4, dan 5 tidak ikut diambil."),
        ], config)
        if not sources:
            raise GeminiError("Gemini merespons, tetapi cuplikannya belum lolos pemeriksaan sumber.")
    except GeminiError as error:
        raise SystemExit(str(error)) from None
    print(f"Gemini OK ({config.model}) · {len(sources)} cuplikan cocok dengan sumber.")


if __name__ == "__main__":
    main()
