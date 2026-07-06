"""Mean-difference activation steering over the decoder layer stack.

Extraction: run the model on paired prompts (faithful vs conflicting caption),
record every decoder layer's hidden state at the last prompt token, and take
vector[layer] = mean(faithful states) - mean(conflicting states).

Application: while generating, add alpha * unit(vector[layer]) to the chosen
layers' residual-stream outputs at every position.
"""

import logging

import torch

import config

log = logging.getLogger(__name__)


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


class _Recorder:
    """Hooks that keep each layer's hidden state at the last position of the
    first forward pass (the full-prompt pass during generation)."""

    def __init__(self, layers: torch.nn.ModuleList):
        self.states: dict[int, torch.Tensor] = {}
        self._handles = [
            layer.register_forward_hook(self._make_hook(i)) for i, layer in enumerate(layers)
        ]

    def _make_hook(self, index: int):
        def hook(_module, _inputs, output):
            if index not in self.states:  # ignore later single-token decode steps
                self.states[index] = _hidden(output)[0, -1, :].detach().float().cpu()
        return hook

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for handle in self._handles:
            handle.remove()

    def stacked(self) -> torch.Tensor:
        return torch.stack([self.states[i] for i in sorted(self.states)])  # [n_layers, hidden]


def capture(vlm, image, prompt: str) -> torch.Tensor:
    """Last-prompt-token hidden state per decoder layer: [n_layers, hidden]."""
    with _Recorder(vlm.decoder_layers()) as recorder:
        vlm.generate(image, prompt, max_new_tokens=1)
    return recorder.stacked()


def mean_difference(positive: list[torch.Tensor], negative: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(positive).mean(0) - torch.stack(negative).mean(0)


class Steer:
    """Context manager applying `alpha * unit(vector)` at the given layers."""

    def __init__(self, vlm, vectors: torch.Tensor, layer_ids: list[int], alpha: float = config.DEFAULT_ALPHA):
        self._handles = []
        layers = vlm.decoder_layers()
        for index in layer_ids:
            direction = torch.nn.functional.normalize(vectors[index], dim=-1)
            self._handles.append(layers[index].register_forward_hook(self._make_hook(direction, alpha)))
        log.info("steering %d layers %s with alpha=%s", len(layer_ids), layer_ids, alpha)

    @staticmethod
    def _make_hook(direction: torch.Tensor, alpha: float):
        def hook(_module, _inputs, output):
            hidden = _hidden(output)
            hidden += alpha * direction.to(hidden.device, hidden.dtype)
            return output
        return hook

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for handle in self._handles:
            handle.remove()


def default_layer_ids(n_layers: int) -> list[int]:
    return sorted({int(fraction * n_layers) for fraction in config.DEFAULT_LAYER_FRACTIONS})


def save_vectors(path, vectors: torch.Tensor, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vectors": vectors, **meta}, path)
    log.info("saved steering vectors %s -> %s", tuple(vectors.shape), path)


def load_vectors(path) -> tuple[torch.Tensor, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload.pop("vectors"), payload
