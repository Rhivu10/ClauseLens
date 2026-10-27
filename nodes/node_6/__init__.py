"""
ClauseLens — Node 6

Report Assembler: combines Node 5's results into one layered JSON report
(overview -> stakeholders/sections -> changes -> evidence), with links from
every change to the exact sentences in both documents and back.
"""

# ============================================================
# ASSEMBLY
# ============================================================

from .assembler import (
    REPORT_VERSION,
    Node6Config,
    ReportAssembler,
)


# ============================================================
# EVIDENCE AND PARTIES
# ============================================================

from .evidence import (
    mark_sentences,
    build_evidence,
)

from .parties import (
    normalize_party,
    normalize_parties,
)


# ============================================================
# EXPORT
# ============================================================

from .export import (
    save_report,
    load_report,
    to_markdown,
    save_markdown,
)


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    # Assembly
    "REPORT_VERSION",
    "Node6Config",
    "ReportAssembler",

    # Evidence and parties
    "mark_sentences",
    "build_evidence",
    "normalize_party",
    "normalize_parties",

    # Export
    "save_report",
    "load_report",
    "to_markdown",
    "save_markdown",
]
