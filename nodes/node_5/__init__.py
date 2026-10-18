"""
ClauseLens — Node 5

Parallel multi-agent core: the Impact Agent (GPU 0) writes who is affected
by each change, the Critic Agent (GPU 1) checks it against the text, and the
routing logic retries or flags it before it goes to Node 6.
"""

# ============================================================
# CONFIGURATION
# ============================================================

from .config import (
    Node5Config,
    NODE5_VERSION,
    VERIFIED,
    HALLUCINATION,
    MISSING_CONTEXT,
    STATUS_VERIFIED,
    STATUS_LOW_CONFIDENCE,
    STATUS_SKIPPED,
)


# ============================================================
# ONTOLOGY
# ============================================================

from .ontology import (
    load_ontology,
    detect_categories,
    stakeholders_for,
    find_parties,
)


# ============================================================
# PROMPTS AND CHECKS
# ============================================================

from .prompts import (
    build_impact_prompt,
    build_critic_prompt,
    parse_impact,
    parse_critic,
)

from .grounding import (
    check_summary,
)


# ============================================================
# AGENTS
# ============================================================

from .agents import (
    load_agent_model,
    load_critic_model,
    CRITIC_BASE_MODELS,
    ImpactAgent,
    CriticAgent,
)


# ============================================================
# CORE
# ============================================================

from .core import (
    Node5ImpactCore,
    read_node5_output,
)


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    # Configuration
    "Node5Config",
    "NODE5_VERSION",
    "VERIFIED",
    "HALLUCINATION",
    "MISSING_CONTEXT",
    "STATUS_VERIFIED",
    "STATUS_LOW_CONFIDENCE",
    "STATUS_SKIPPED",

    # Ontology
    "load_ontology",
    "detect_categories",
    "stakeholders_for",
    "find_parties",

    # Prompts and checks
    "build_impact_prompt",
    "build_critic_prompt",
    "parse_impact",
    "parse_critic",
    "check_summary",

    # Agents
    "load_agent_model",
    "load_critic_model",
    "CRITIC_BASE_MODELS",
    "ImpactAgent",
    "CriticAgent",

    # Core
    "Node5ImpactCore",
    "read_node5_output",
]
