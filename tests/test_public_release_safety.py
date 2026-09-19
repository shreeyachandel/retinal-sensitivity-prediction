from __future__ import annotations

import json
import py_compile
import re
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_RESEARCH_AREAS = [ROOT / "research_code", ROOT / "docs" / "thesis-public-source"]
TEXT_SUFFIXES = {".bib", ".json", ".md", ".py", ".tex", ".txt"}


class PublicReleaseSafetyTests(unittest.TestCase):
    def test_no_study_identifiers_or_local_paths(self) -> None:
        forbidden_patterns = {
            "study pseudonym": re.compile(r"U" + r"SH-[A-F0-9]+"),
            "student number": re.compile("2204" + "8742"),
            "local user path": re.compile(r"/Users/" + r"shreeyachandel"),
        }
        violations: list[str] = []

        for area in PUBLIC_RESEARCH_AREAS:
            for path in area.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                for label, pattern in forbidden_patterns.items():
                    if pattern.search(text):
                        violations.append(f"{path.relative_to(ROOT)}: {label}")

        self.assertEqual([], violations)

    def test_participant_qc_images_are_not_present(self) -> None:
        prohibited_names = {
            "figure_registration_qc_" + "example.png",
            "figure_supplementary_qc_" + "panel.png",
        }
        present = {path.name for path in ROOT.rglob("*") if path.is_file()}
        self.assertTrue(prohibited_names.isdisjoint(present))

    def test_restricted_data_formats_are_absent(self) -> None:
        prohibited_suffixes = {
            ".dcm", ".e2e", ".h5", ".hdf5", ".nii", ".parquet", ".pth", ".pt", ".xlsx"
        }
        violations = [
            str(path.relative_to(ROOT))
            for path in ROOT.rglob("*")
            if path.is_file() and path.suffix.lower() in prohibited_suffixes
        ]
        self.assertEqual([], violations)

    def test_research_modules_compile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            for index, path in enumerate((ROOT / "research_code" / "src").rglob("*.py")):
                py_compile.compile(
                    str(path),
                    cfile=str(output / f"module-{index}.pyc"),
                    doraise=True,
                )

    def test_public_notebook_has_no_outputs(self) -> None:
        path = ROOT / "notebooks" / "retinal_sensitivity_demo.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                self.assertEqual([], cell.get("outputs", []))
                self.assertIsNone(cell.get("execution_count"))


if __name__ == "__main__":
    unittest.main()
