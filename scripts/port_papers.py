"""Usage: python3 scripts/port_papers.py.

The ACL version is the source; the ICML, NeurIPS and TMLR versions are built from it.
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


def unscale(text: str) -> str:
    """One column: tables scaled to the column width would fill the page; keep them natural."""
    return text.replace("\\resizebox{\\columnwidth}{!}{%", "{%")


# ICML: the ethics section is an Impact Statement, its AI paragraph becomes the
# required AI Use Statement of its own; icml2027 bibliography style.
icml_path = PAPER / "icml" / "driftwatch_icml.tex"
icml = icml_path.read_text()
pre = icml[: icml.index("\\printAffiliationsAndNotice{}") + len("\\printAffiliationsAndNotice{}")]
b = body.replace("\\section*{Ethics Statement}", "\\section*{Impact Statement}")
b = b.replace("\\paragraph{AI assistance} ", "\\section*{AI Use Statement}\n")
b = b.replace(
    "\\bibliography{../references}", "\\bibliography{../references}\n\\bibliographystyle{icml2027}"
)
icml_path.write_text(pre + "\n\n" + b)

# NeurIPS: one column, so the two widest tables are scaled to the text width.
neurips_path = PAPER / "driftwatch.tex"
neurips = neurips_path.read_text()
pre = neurips[: neurips.index("\\maketitle") + len("\\maketitle")]
b = body.replace(
    "\\bibliography{../references}", "\\bibliographystyle{abbrvnat}\n\\bibliography{references}"
)
b = wrap(unscale(b), "\\begin{tabular}{l|ccc|ccc|ccc}", "\\textwidth")
b = wrap(
    b, "\\begin{tabular}{lrrrrrrrr}\n    \\toprule\n    setting & threshold change", "\\textwidth"
)
neurips_path.write_text(pre + "\n\n" + b)

# TMLR: one column like NeurIPS, so the same tables are scaled; the ethics section
# is a Broader Impact Statement; tmlr bibliography style.
tmlr_path = PAPER / "tmlr" / "driftwatch_tmlr.tex"
tmlr = tmlr_path.read_text()
pre = tmlr[: tmlr.index("\\maketitle") + len("\\maketitle")]
b = body.replace("\\section*{Ethics Statement}", "\\section*{Broader Impact Statement}")
b = b.replace(
    "\\bibliography{../references}", "\\bibliography{../references}\n\\bibliographystyle{tmlr}"
)
b = wrap(unscale(b), "\\begin{tabular}{l|ccc|ccc|ccc}", "\\textwidth")
b = wrap(
    b, "\\begin{tabular}{lrrrrrrrr}\n    \\toprule\n    setting & threshold change", "\\textwidth"
)
tmlr_path.write_text(pre + "\n\n" + b)
print("ported: icml/driftwatch_icml.tex, driftwatch.tex, tmlr/driftwatch_tmlr.tex")  # noqa: T201
