"""Usage: python3 scripts/port_papers.py

The ACL version is the source; the ICML and NeurIPS versions are built from it.
Each keeps its own preamble (style file, title block, authors) and takes the ACL
text from the abstract to the end, with the few venue changes below. Run after
every edit to paper/acl/driftwatch_acl.tex, then build all three.
"""

from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent / "paper"
acl = (PAPER / "acl" / "driftwatch_acl.tex").read_text()
body = acl[acl.index("\\begin{abstract}") :]


def wrap(text: str, marker: str, width: str) -> str:
    """Scale the tabular that starts at `marker` to `width`, unless it already is."""
    i = text.index(marker)
    if text[max(0, i - 40) : i].rstrip().endswith("{!}{%"):
        return text
    j = text.index("\\end{tabular}", i) + len("\\end{tabular}")
    return text[:i] + "\\resizebox{" + width + "}{!}{%\n" + text[i:j] + "}" + text[j:]


# ICML: the ethics section is an Impact Statement; icml2026 bibliography style.
icml_path = PAPER / "icml" / "driftwatch_icml.tex"
icml = icml_path.read_text()
pre = icml[: icml.index("\\printAffiliationsAndNotice{}") + len("\\printAffiliationsAndNotice{}")]
b = body.replace("\\section*{Ethics Statement}", "\\section*{Impact Statement}")
b = b.replace(
    "\\bibliography{../references}", "\\bibliography{../references}\n\\bibliographystyle{icml2026}"
)
icml_path.write_text(pre + "\n\n" + b)

# NeurIPS: one column, so the two widest tables are scaled to the text width.
neurips_path = PAPER / "driftwatch.tex"
neurips = neurips_path.read_text()
pre = neurips[: neurips.index("\\maketitle") + len("\\maketitle")]
b = body.replace(
    "\\bibliography{../references}", "\\bibliographystyle{abbrvnat}\n\\bibliography{references}"
)
b = wrap(b, "\\begin{tabular}{l|ccc|ccc|ccc}", "\\textwidth")
b = wrap(
    b, "\\begin{tabular}{lrrrrrrrr}\n    \\toprule\n    setting & threshold change", "\\textwidth"
)
neurips_path.write_text(pre + "\n\n" + b)
print("ported: icml/driftwatch_icml.tex, driftwatch.tex")
