"""Extract public-safe research modules from the submitted notebook.

The submitted notebook stores source modules in ``%%writefile`` cells. This
utility extracts those modules, replaces study-specific pseudonyms with
synthetic labels, and excludes the vendored RETFound implementation. It never
copies notebook outputs or research data.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


WRITEFILE_PREFIX = "%%writefile "
SOURCE_PREFIX = Path("_notebook_source_self_contained/src/thesis_pipeline")
CASE_PATTERN = re.compile(r"USH-[A-F0-9]+")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def _synthetic_case_map(texts: list[str]) -> dict[str, str]:
    case_ids = sorted({match.group(0) for text in texts for match in CASE_PATTERN.finditer(text)})
    return {
        case_id: f"PUBLIC-PARTICIPANT-{index:03d}"
        for index, case_id in enumerate(case_ids, start=1)
    }


def extract(notebook_path: Path, output_root: Path) -> list[Path]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    modules: list[tuple[Path, str]] = []

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        first_line, separator, remainder = source.partition("\n")
        if not separator or not first_line.startswith(WRITEFILE_PREFIX):
            continue

        embedded_path = Path(first_line.removeprefix(WRITEFILE_PREFIX).strip())
        try:
            relative_path = embedded_path.relative_to(SOURCE_PREFIX)
        except ValueError:
            continue

        # RETFound is third-party CC BY-NC 4.0 code. The public release links
        # to the pinned upstream project instead of relicensing a vendored copy.
        if relative_path == Path("implementation/official_retfound/models_vit.py"):
            continue
        modules.append((relative_path, remainder))

    replacements = _synthetic_case_map([source for _, source in modules])
    written: list[Path] = []
    header = (
        "# Public-release copy extracted from the submitted MSc notebook.\n"
        "# Study-specific pseudonyms and governed data are intentionally absent.\n\n"
    )

    for relative_path, source in modules:
        for original, replacement in replacements.items():
            source = source.replace(original, replacement)
        destination = output_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(header + source, encoding="utf-8")
        written.append(destination)

    return written


def main() -> None:
    args = _parse_args()
    written = extract(args.notebook, args.output)
    print(f"Extracted {len(written)} public research modules to {args.output}")


if __name__ == "__main__":
    main()
