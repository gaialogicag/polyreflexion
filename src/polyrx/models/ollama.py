"""Local Ollama LLM backend."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

from polyrx.config import OllamaConfig


def extract_final_answer(raw: str) -> str:
    """Strip reasoning blocks and return the final answer from a model response.

    Phi reasoning models often emit ``<think>...`` (sometimes unclosed) and then a
    short answer after ``</think>`` or as ``**label**`` / ``\\boxed{...}``.
    """
    if not raw:
        return ""

    text = raw.strip()

    # Prefer everything after the last closed thinking block when present.
    close_tags = list(re.finditer(r"</(?:redacted_)?thinking>|</think>", text, flags=re.IGNORECASE))
    if close_tags:
        text = text[close_tags[-1].end() :].strip()
    else:
        # Drop a fully closed thinking span if the model used proper tags.
        text = re.sub(
            r"<(?:redacted_)?thinking>.*?</(?:redacted_)?thinking>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()
        # If we only have an unclosed <think>... blob, keep looking for markers
        # inside it rather than returning the whole chain-of-thought.
        if re.match(r"^<(?:redacted_)?thinking>|<think", text, flags=re.IGNORECASE):
            boxed_inside = _extract_boxed(text)
            if boxed_inside is not None:
                return boxed_inside
            bold = _extract_bold_answer(text)
            if bold is not None:
                return bold
            # Last non-empty line is a weak fallback for truncated CoT.
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            return lines[-1] if lines else ""

    final_answer_re = re.compile(r"\*\*Final Answer:?\*\*", re.IGNORECASE)
    if final_answer_re.search(text):
        text = final_answer_re.split(text)[-1].strip()

    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed

    bold = _extract_bold_answer(text)
    if bold is not None:
        return bold

    text = re.sub(r"^Rewritten text:\s*", "", text, flags=re.IGNORECASE)
    return text.strip().strip('"')


def _extract_bold_answer(text: str) -> str | None:
    """Return the last short ``**answer**`` span, if it looks like a label."""
    matches = list(re.finditer(r"\*\*([^*]{1,80})\*\*", text))
    if not matches:
        return None
    candidate = matches[-1].group(1).strip()
    # Ignore long analysis headings accidentally wrapped in bold.
    if len(candidate.split()) > 8:
        return None
    return candidate


def _extract_boxed(text: str) -> str | None:
    match = re.search(r"\\boxed\{", text)
    if not match:
        return None

    start = match.end()
    depth = 1
    i = start
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    inner = text[start : i - 1]
    text_match = re.match(r"\\text\{(.+)\}", inner, re.DOTALL)
    return (text_match.group(1) if text_match else inner).strip()


def looks_degenerate(text: str) -> bool:
    """Heuristic: Phi sometimes collapses into token-salad / fake math puzzles."""
    if not text or len(text.strip()) < 20:
        return True
    sample = text[:4000]
    digit_ratio = sum(ch.isdigit() for ch in sample) / max(len(sample), 1)
    if digit_ratio > 0.18:
        return True
    # Typical degeneration phrases from failed phi4-mini-reasoning runs.
    bad_markers = (
        "your name is",
        "named as",
        "segment",
        "\\boxed",
        "the value of r",
        "process results",
    )
    lower = sample.lower()
    hits = sum(1 for marker in bad_markers if marker in lower)
    return hits >= 2


class OllamaClient:
    """Local Ollama backend returning only the final answer."""

    def __init__(self, config: OllamaConfig | None = None) -> None:
        self.config = config or OllamaConfig()
        self.model = self.config.model
        self.host = self.config.host
        self.num_ctx = self.config.num_ctx
        self.num_predict = self.config.num_predict
        self.timeout = self.config.timeout_s
        self.retries = self.config.retries
        self.temperature = self.config.temperature
        self.strip_reasoning = self.config.strip_reasoning

    def complete(self, prompt: str) -> str:
        raw = self._raw(prompt)
        return extract_final_answer(raw) if self.strip_reasoning else raw

    def _raw(self, prompt: str) -> str:
        body: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
                "temperature": self.temperature,
            },
        }
        # Newer Ollama builds accept a top-level "think" flag for reasoning
        # models. ``None`` omits it, which older builds require.
        if self.config.think is not None:
            body["think"] = self.config.think

        payload = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.host.rstrip('/')}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())["response"]
            except TimeoutError as exc:
                last_error = exc
                print(f"Ollama timeout (attempt {attempt}/{self.retries}), retrying...")
                time.sleep(min(30, 5 * attempt))
            except urllib.error.URLError as exc:
                last_error = exc
                if attempt == self.retries:
                    raise ConnectionError(
                        f"Ollama not reachable at {self.host}. Start with: ollama serve"
                    ) from exc
                print(f"Ollama URL error (attempt {attempt}/{self.retries}): {exc}")
                time.sleep(min(30, 5 * attempt))
        raise TimeoutError(
            f"Ollama request timed out after {self.retries} attempts (timeout={self.timeout}s)"
        ) from last_error
