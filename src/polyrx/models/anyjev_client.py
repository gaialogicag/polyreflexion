"""AnyJev backend: a typed yes/no verdict, not a text completion.

AnyJev (https://github.com/nokia-applied-research/AnyJev) answers a typed
``Question`` (choice/score/yes-no) with a calibrated probability, read from a
model served locally through transformers (``backend_kind: hf``) or vLLM
(``backend_kind: vllm``). That is a different shape of call than this
project's ``complete(prompt) -> str``, so ``complete()`` stays an explicit
refusal -- the real entry point is :meth:`AnyJevClient.decide_verdict`, built
around AnyJev's own ``Decider``/``Question`` objects, which is what
:class:`polyrx.meta.judges.PolyJudge` calls directly for an AnyJev-backed
judge role instead of going through ``complete()`` and parsing JSON out of
free text.

Needs the ``anyjev[hf]`` (or plain ``anyjev`` for a vLLM-only setup) extra,
not a dependency of this package by default -- it pulls in ``torch`` and
``transformers``, which most runs never need.
"""

from __future__ import annotations

from polyrx.config import AnyJevConfig


class AnyJevClient:
    """Wraps one AnyJev ``Decider`` per (model, backend_kind), lazily built.

    The backend is built on first use, not at construction -- for
    ``backend_kind: hf`` that means loading real model weights into memory,
    which the credential-free, instant-to-construct clients for every other
    provider do not do.
    """

    def __init__(self, config: AnyJevConfig | None = None) -> None:
        self.config = config or AnyJevConfig()
        self._decider = None

    def _get_decider(self):
        if self._decider is not None:
            return self._decider
        try:
            from anyjev import Decider
        except ImportError as exc:
            raise ImportError(
                "AnyJev is not installed. Run: pip install 'anyjev[hf]' "
                "(or 'anyjev' alone for a vLLM-only setup)."
            ) from exc

        if self.config.backend_kind == "hf":
            from anyjev.backends.hf import HFBackend

            backend = HFBackend(self.config.model, device="cpu", dtype="float32")
        elif self.config.backend_kind == "vllm":
            from anyjev.backends.vllm import VLLMBackend

            backend = VLLMBackend(self.config.base_url, self.config.model)
        else:
            raise ValueError(
                f"Unknown AnyJev backend_kind {self.config.backend_kind!r}; use 'hf' or 'vllm'."
            )
        self._decider = Decider(backend, level=self.config.level)
        return self._decider

    def decide_verdict(self, state: str, question_text: str) -> tuple[bool, float]:
        """A yes/no verdict and its probability, for one piece of judged content.

        ``state`` is everything the model should read (the problem, the
        answer under evaluation, any supporting artifacts); ``question_text``
        is the single yes/no question asked about it. Returns
        ``(positive, p_true)`` -- ``positive`` is ``p_true >= 0.5``.
        """
        from anyjev import Question

        decider = self._get_decider()
        question = Question.noul(question_text, name="verdict")
        decision = decider.decide(state, [question])["verdict"]
        return decision.p_true >= 0.5, decision.p_true

    def complete(self, prompt: str) -> str:
        raise NotImplementedError(
            "AnyJevClient has no text-completion mode -- call decide_verdict(state, "
            "question_text) instead. complete() exists only to satisfy the LLMClient "
            "protocol for call sites that have not been updated to the typed interface."
        )
