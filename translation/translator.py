"""Batch translation with Aya Expanse via vLLM.

One prompt carries all text fields of one dataset row as a JSON array, so the
model translates them together with shared context; failed parses fall back to
one-text-per-prompt, and as a last resort the original English is kept.
"""

import json
import logging
import os
from functools import cached_property

import config

log = logging.getLogger(__name__)

PROMPT_TEMPLATE = """\
You are a professional, native-level scientific and technical translator.
Translate the following list of English texts into the target language: {target_language}.

Core Translation Constraints:
Semantic and Logical Equivalence: The translated text must assert the exact same logical, physical, or spatial facts as the source English text. Do not omit information. However, do NOT translate word-for-word; always prioritize natural grammatical structure, flow, and standard word order in {target_language}.
Terminology Consistency: Identify key nouns, technical terms, and domain-specific vocabulary (e.g., medical, physics, culinary, programming, or spatial terms) in the input. Ensure that each specific technical term is translated mathematically and scientifically correctly (e.g., translate terms like 'vector', 'list', 'slope', or 'capacity' to their exact scientific/mathematical technical equivalents in the target language, avoiding colloquial approximations or related but incorrect terms like using velocity instead of vector).
Variable and Code Preservation:
Do NOT translate programming code, syntax, variable names, parameter/list names (e.g., x, arr, i, dx, dy, mid, rows, source, target), mathematical operators, or logic statements.
Keep all code elements, variable names, and parameters EXACTLY in their original script and character casing as they appear in the source text. Do NOT translate or transliterate them into any other alphabet, writing system, or script.

Input texts (JSON array of {n} strings):
{texts_json}

Respond with ONLY a JSON array of exactly {n} translated strings, in the same order as the input. No commentary, no markdown fences."""


def parse_json_array(raw: str, expected_len: int) -> list[str] | None:
    """Extract a JSON array of `expected_len` strings from model output, or None."""
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end <= start:
        return None
    try:
        result = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    if isinstance(result, list) and len(result) == expected_len and all(isinstance(s, str) for s in result):
        return result
    return None


class AyaTranslator:
    def __init__(self, model_id: str = config.MODEL_ID, max_model_len: int = 8192):
        self.model_id = model_id
        self.max_model_len = max_model_len

    @cached_property
    def _llm(self):
        # FlashInfer's sampler JIT-compiles with nvcc, which compute nodes lack.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM  # deferred: import + model load cost only paid when translating

        return LLM(model=self.model_id, max_model_len=self.max_model_len, gpu_memory_utilization=0.92)

    @cached_property
    def _sampling_params(self):
        from vllm import SamplingParams

        return SamplingParams(temperature=0.0, max_tokens=4096)

    def _chat(self, batches: list[list[str]], target_language: str) -> list[list[str] | None]:
        """One conversation per batch; returns parsed translations (None where unparseable)."""
        conversations = [
            [{
                "role": "user",
                "content": PROMPT_TEMPLATE.format(
                    target_language=target_language,
                    n=len(batch),
                    texts_json=json.dumps(batch, ensure_ascii=False),
                ),
            }]
            for batch in batches
        ]
        outputs = self._llm.chat(conversations, self._sampling_params)
        return [parse_json_array(out.outputs[0].text, len(batch)) for out, batch in zip(outputs, batches)]

    def translate_batches(self, batches: list[list[str]], target_language: str) -> list[list[str]]:
        """Translate every string in every batch; shape of the output mirrors the input."""
        compact = [[text for text in batch if text.strip()] for batch in batches]
        todo = [i for i, batch in enumerate(compact) if batch]

        parsed = dict(zip(todo, self._chat([compact[i] for i in todo], target_language)))

        failed = [i for i in todo if parsed[i] is None]
        if failed:
            log.warning("%d/%d batches unparseable; retrying one text per prompt", len(failed), len(todo))
            singles = [(i, j) for i in failed for j in range(len(compact[i]))]
            retried = self._chat([[compact[i][j]] for i, j in singles], target_language)
            for (i, j), result in zip(singles, retried):
                if parsed[i] is None:
                    parsed[i] = list(compact[i])  # start from originals, patch what succeeded
                if result is not None:
                    parsed[i][j] = result[0]
                else:
                    log.error("keeping original text (batch %d item %d): %.80s", i, j, compact[i][j])

        translated = []
        for i, batch in enumerate(batches):
            replacements = iter(parsed.get(i) or [])
            translated.append([next(replacements) if text.strip() else text for text in batch])
        return translated
