"""analysis.py (ячейки «# %%») → paris_dvf_eda.ipynb, исполненный.

    python make_notebook.py
"""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent


def cells(src: str):
    blocks, cur, kind = [], [], None
    for line in src.splitlines():
        if line.startswith("# %%"):
            if cur:
                blocks.append((kind, cur))
            kind, cur = ("md" if "[markdown]" in line else "code"), []
        else:
            cur.append(line)
    blocks.append((kind, cur))
    for kind, lines in blocks:
        text = "\n".join(lines).strip("\n")
        if kind == "md":
            yield nbformat.v4.new_markdown_cell("\n".join(x[2:] if x.startswith("# ") else x.lstrip("#")
                                                          for x in text.splitlines()))
        elif text:
            yield nbformat.v4.new_code_cell(text)


def main() -> None:
    nb = nbformat.v4.new_notebook()
    nb.cells = list(cells((ROOT / "analysis.py").read_text(encoding="utf-8")))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(nb, ROOT / "paris_dvf_eda.ipynb")
    print("paris_dvf_eda.ipynb")


if __name__ == "__main__":
    main()
