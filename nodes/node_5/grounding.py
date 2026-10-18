"""
ClauseLens — Node 5
Grounding checks: rule-based fact checks on the Impact Agent's summary.

They run BEFORE the Critic Agent. A failed check counts as HALLUCINATION
without asking the Critic, because these facts can be checked exactly:

    - every number in the summary appears in the clause text
    - every AFFECTED party is named in the text, is one of the contract's parties
      (from its opening clause), or is a stakeholder hint
    - at most `max_unsupported_words` content words are missing from the text
    - the summary is not empty

No model or GPU dependency.
"""

from __future__ import annotations

import re
from typing import Any

from .ontology import party_is_grounded

_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORD_RE = re.compile(r"[a-z]+")

# Words a summary may use freely: describing a change, duties and parties in
# general, plus common function words. Everything else must appear in the text.
_FREE_WORDS = set("""
about above after again against agreement also amended amendment amount apply applies applied around
because been before being between both cannot change changed changes clause contract could
decrease decreased deleted does doing done during each earlier either every extended extend
from given gives have having higher however impact include includes including increase increased
increases instead into less longer lower made make makes many means modified modify more most must
neither none obligation obligations obliged only other others over party parties period permission
permit permits permitted previous previously prohibit prohibited prohibition prohibits provision
provisions reduced reduce regarding related remove removed removes require required requirement
requirements requires right rights same section shall should shorter similar since some still such
than that their them then there these they this those through under upon version very what when
where whether which while will with within without would added adds allow allowed allows affect
affected affects anyone everyone longer stated states statement text clause clauses term terms
able responsible responsibility relevant specific specified certain particular entire overall
old older wording worded words duty duties completely entirely fully another anymore making
side sides whatever whenever thereby therefore although
one two three four five six seven eight nine ten eleven twelve fifteen twenty thirty forty fifty
sixty seventy eighty ninety hundred thousand million percent
said says sentence sentences language phrase phrases additions additionally furthermore moreover
finally clarify clarifies clarified describe describes described mention mentions mentioned refer
refers referenced item items part parts point points rule rules scope just rather found remain
remains stays compared comparison difference differences listed details detailed
""".split())


# 3-letter words that carry no facts ("fee", "pay", "day" do, so they are checked)
_SHORT_STOP = set("""
the and for its has had not are but can all any now new who how may own per was one two out off
via yet nor our his her him she they too did does got let put set use due end top was way
""".split())


def _related(a: str, b: str) -> bool:
    """Same word family: shared prefix of min(5, len) letters, at least 3 ("pay"/"paying", "week"/"weekly")."""
    n = min(5, len(a), len(b))
    return n >= 3 and a[:n] == b[:n]


def unsupported_words(summary: str, text: str) -> list[str]:
    """
    Content words of the summary with no word of the same family in the text.
    Words describing a change ("removed", "prohibiting", ...) are always allowed.
    """
    vocab = set(_WORD_RE.findall(text.lower()))
    out = []
    for w in dict.fromkeys(_WORD_RE.findall(summary.lower())):
        if len(w) < 3 or (len(w) == 3 and w in _SHORT_STOP) or w in vocab:
            continue
        if any(_related(w, f) for f in _FREE_WORDS) or any(_related(w, v) for v in vocab):
            continue
        out.append(w)
    return out


def number_cores(text: str) -> set[str]:
    """Numbers reduced to their digits: "$65,000" -> "65000", "1%" -> "1", "2.1" -> "2.1"."""
    out = set()
    for m in _NUM_RE.finditer(text or ""):
        core = m.group(0).replace(",", "").rstrip(".")
        if core:
            out.add(core)
    return out


def check_summary(
    summary: str,
    affected: list[str],
    view: dict[str, Any],
    min_summary_chars: int = 20,
    max_unsupported_words: int = 2,
) -> list[str]:
    """Returns a list of problems; empty list means the summary passed."""
    issues = []
    if len(summary.strip()) < min_summary_chars:
        issues.append("the summary is empty or too short")
        return issues

    text = " ".join(t for t in (view.get("source_text"), view.get("target_text"),
                                view.get("context_text"), view.get("heading")) if t)

    missing = sorted(number_cores(summary) - number_cores(text))
    if missing:
        issues.append("the summary mentions numbers that are not in the text: " + ", ".join(missing))

    allowed = (list(view.get("parties", [])) + list(view.get("contract_parties", []))
               + list(view.get("stakeholders", [])))
    ungrounded = [a for a in affected if not party_is_grounded(a, text, allowed)]
    if ungrounded:
        issues.append("these affected parties are not named in the text: " + ", ".join(ungrounded))

    # Invented details: tuned on Kaggle output, where Gemma added "personally
    # identifiable information" and "pay a fee ... each year" to clauses that say neither
    unsupported = unsupported_words(summary, " ".join([text] + list(view.get("contract_parties", []))))
    if len(unsupported) > max_unsupported_words:
        # Kaggle run 5: a bare word list made the retry drop the specific facts
        # ("from 60 to 90 days" became "a certain number of days")
        issues.append("the summary uses words that are not in the text (" + ", ".join(unsupported[:8])
                      + "). Use the text's own words, but keep the specific facts such as numbers, "
                        "deadlines and who must do what")

    return issues
