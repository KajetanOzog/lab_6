import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

import pymupdf


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_suffix(".tex")
OUTPUT = ROOT / "images" / "sdf"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tectonic", default=shutil.which("tectonic"))
    args = parser.parse_args()
    if not args.tectonic:
        parser.error("Podaj kompilator LaTeX: --tectonic SCIEZKA_DO_TECTONIC")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lab6-sdf-latex-") as temporary:
        subprocess.run(
            [args.tectonic, "--outdir", temporary, str(SOURCE)],
            cwd=ROOT, check=True,
        )
        with pymupdf.open(Path(temporary) / "sdf_definition.pdf") as document:
            assert len(document) == 1
            page = document[0]
            text = page.get_text()
            for phrase in ["wewnątrz obiektu", "na powierzchni", "na zewnątrz obiektu"]:
                assert phrase in text, (phrase, text)
            svg = page.get_svg_image(text_as_path=True)
            (OUTPUT / "sdf_definition.svg").write_text(svg, encoding="utf-8")
    print(OUTPUT / "sdf_definition.svg")


if __name__ == "__main__":
    main()
