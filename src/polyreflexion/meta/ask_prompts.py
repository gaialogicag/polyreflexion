"""Answer-oriented prompt overrides used by ``ask_meta.py``.

The default engine templates in ``prompts.json`` are tuned for OpenToM story
summaries ("integrated summary of perspectives").  Interactive Q&A needs the
opposite: use the three perspectives as *reasoning material*, then emit one
best answer to the question — never a catalogue of competing views.
"""

from __future__ import annotations

from polyreflexion.engine import Perspective, PromptRegistry
from polyreflexion.meta.prompts import MetaPromptRegistry

# Final synthesis: one answer, not a survey of O / S / B.
ASK_SUMMARY = """\
Question:
{input}

You have reasoned about this question from three complementary angles.
Use them as internal material only — do not list or summarize them separately
in your reply.

Objectivity (O) notes:
{corner_o}

Subjectivity (S) notes:
{corner_s}

Conceptuality (B) notes:
{corner_b}

Task: Write the single best possible answer to the question above.
Requirements:
- Answer the question directly and decisively.
- Prefer one clear position (or a precise conditional) over surveying options.
- Do not structure the reply as "from an objective view / subjective view /
  conceptual view" or similar.
- Plain English. No math puzzles, codes, segment IDs, or meta-commentary about
  the reasoning process.
"""

ASK_PERSPECTIVES = {
    Perspective.OBJECTIVITY: """\
Question / input:
{input}

Region context:
{context}

From an objective standpoint, contribute 2-4 plain-English sentences that help
answer the question. Focus on facts, constraints, and what can be checked.
Do not write a full multi-view essay — just the objective contribution.
No math, codes, IDs, or puzzle formatting.
""",
    Perspective.SUBJECTIVITY: """\
Question / input:
{input}

Region context:
{context}

From a subjective standpoint, contribute 2-4 plain-English sentences that help
answer the question. Focus on intentions, beliefs, values, and lived stakes.
Do not write a full multi-view essay — just the subjective contribution.
No math, codes, IDs, or puzzle formatting.
""",
    Perspective.CONCEPTUALITY: """\
Question / input:
{input}

Region context:
{context}

From a conceptual standpoint, contribute 2-4 plain-English sentences that help
answer the question. Focus on concepts, tensions, and how the issue hangs
together. Do not write a full multi-view essay — just the conceptual contribution.
No math, codes, IDs, or puzzle formatting.
""",
}

ASK_BOUNDARIES = {
    "objectivity_subjectivity": (
        "Given this question / text:\n{input}\n\n"
        "In one or two plain-English sentences, where is the boundary between "
        "objectivity and subjectivity for answering it? No math, codes, IDs, "
        "or puzzle formatting."
    ),
    "objectivity_conceptuality": (
        "Given this question / text:\n{input}\n\n"
        "In one or two plain-English sentences, where is the boundary between "
        "objectivity and conceptuality for answering it? No math, codes, IDs, "
        "or puzzle formatting."
    ),
    "conceptuality_subjectivity": (
        "Given this question / text:\n{input}\n\n"
        "In one or two plain-English sentences, where is the boundary between "
        "conceptuality and subjectivity for answering it? No math, codes, IDs, "
        "or puzzle formatting."
    ),
}

ASK_EXPAND_INPUT = """\
Original question:
{question}

Current answer:
{answer}

Judge feedback on the current answer (address every NOT satisfied item):
{judge_feedback}

Expansion axis: {dimension}

Semantic boundary for this axis:
{boundary}

Task: Produce the single best possible answer to the original question.
Requirements:
- Fix every failing judge concern listed above.
- Preserve what satisfied judges already approved.
- If dialectical was R (redundant), add genuinely new substance — do not
  paraphrase the previous answer.
- Use the boundary to find what the current answer still misses.
- Answer directly and decisively; do not catalogue viewpoints.
Plain English only.
"""

ASK_REFINE_INPUT = """\
Original question:
{question}

Current answer:
{answer}

Judge feedback on the current answer:
{judge_feedback}

Strength to preserve at apex {apex}:
{apex_guidance}

[Region]
Apex: {apex}
Left boundary ({left_label}): {left_boundary}
Right boundary ({right_label}): {right_boundary}

Task: Deepen inside this refined region, then output the single best direct
answer to the original question. The left/right boundaries are failing judges'
concerns — satisfy them while keeping the apex strength. Target: earn approval
from all three judges (authentic, objectively defensible, dialectically novel).
Do not list boundaries or competing perspectives in the reply. Plain English only.
"""


def build_ask_engine_prompts() -> PromptRegistry:
    """Engine prompts that synthesize a best answer (not a multi-view summary)."""
    prompts = PromptRegistry()
    prompts._summary = ASK_SUMMARY
    prompts._perspectives = dict(ASK_PERSPECTIVES)
    prompts._boundaries = dict(ASK_BOUNDARIES)
    return prompts


def build_ask_meta_prompts() -> MetaPromptRegistry:
    """Meta expand/refine templates that demand a best answer, not a survey."""
    from polyreflexion.meta.profiles import get_meta_profile

    return get_meta_profile("ask").build_registry()


def apply_ask_templates(base: MetaPromptRegistry) -> MetaPromptRegistry:
    """Apply ask-mode overrides onto a registry (used by ``AskMetaProfile``)."""
    base._templates["expand_input"] = ASK_EXPAND_INPUT
    base._templates["refine_input"] = ASK_REFINE_INPUT
    return base
