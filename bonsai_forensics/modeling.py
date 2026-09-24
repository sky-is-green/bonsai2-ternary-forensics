"""Model-loading adapters used by the reproducible QAT pilot.

Most candidate checkpoints are ordinary ``AutoModelForCausalLM`` models.  The
official Qwen3.8-27B checkpoint is different: its config advertises
``Qwen3_5ForConditionalGeneration`` and includes a vision tower even when the
pilot is text-only.  ``load_text_causal_lm`` turns that checkpoint into a
text-only causal-LM view without changing the underlying weights.

This adapter solves the architecture/interface mismatch; it does **not** make a
27B model fit on a 24 GB card.  Large Qwen3.8 experiments still need the
sharded/block-wise path documented in the model registry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class TextOnlyCausalLM(nn.Module):
    """Expose ``Qwen3_5ForConditionalGeneration.model.language_model`` as a LM."""

    def __init__(self, language_model: nn.Module, lm_head: nn.Module,
                 config: Any, loss_function: Any = None) -> None:
        super().__init__()
        self.model = language_model
        self.lm_head = lm_head
        self.config = config
        # The wrapped text tower is a *base* model, so the conditional model's
        # loss function (which owns the label-shifting convention) is captured
        # separately.  Without it a ``labels=`` forward raises AttributeError.
        self.loss_function = loss_function

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.model.set_input_embeddings(value)

    def gradient_checkpointing_enable(self, **kwargs):
        return self.model.gradient_checkpointing_enable(**kwargs)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                labels=None, **kwargs):
        from transformers.modeling_outputs import CausalLMOutputWithPast

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            if self.loss_function is not None:
                loss = self.loss_function(
                    logits=logits,
                    labels=labels,
                    vocab_size=self.config.vocab_size,
                )
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def _is_qwen35_conditional(config) -> bool:
    model_type = str(getattr(config, "model_type", "")).lower()
    architectures = " ".join(str(x) for x in (getattr(config, "architectures", None) or ()))
    return model_type.startswith("qwen3_5") and "ConditionalGeneration" in architectures


def load_text_causal_lm(model_dir: str | Path, *, revision: str | None = None,
                        dtype: torch.dtype = torch.bfloat16,
                        device: torch.device | str | None = None,
                        trust_remote_code: bool = False,
                        local_files_only: bool = False,
                        device_map: str | dict | None = None):
    """Load a causal LM, including Qwen3.8's multimodal text-only view."""
    from transformers import AutoConfig, AutoModelForCausalLM

    load_kwargs = {
        "revision": revision,
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    config = AutoConfig.from_pretrained(model_dir, **load_kwargs)
    if _is_qwen35_conditional(config):
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3.8 requires a Transformers build with "
                "AutoModelForImageTextToText"
            ) from exc
        full = AutoModelForImageTextToText.from_pretrained(
            model_dir, dtype=dtype, device_map=device_map, **load_kwargs)
        if not hasattr(full.model, "language_model"):
            raise TypeError(
                f"{type(full).__name__} has no model.language_model text tower")
        language_model = full.model.language_model
        lm_head = full.lm_head
        # Capture the conditional model's loss function (it owns the label
        # shifting) before dropping the wrapper.
        loss_function = getattr(full, "loss_function", None)
        # Drop the vision/MTP wrapper before returning so its parameters do not
        # remain in the optimizer/state dict.  The text modules are retained by
        # reference, so this does not copy the 27B weights.
        del full
        model = TextOnlyCausalLM(language_model, lm_head, config.text_config,
                                 loss_function)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=dtype, device_map=device_map, **load_kwargs)
    if device is not None and device_map is None:
        model = model.to(device)
    return model


__all__ = ["TextOnlyCausalLM", "load_text_causal_lm"]
