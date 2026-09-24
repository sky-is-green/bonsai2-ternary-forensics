import pytest
import torch


def test_qwen35_conditional_loader_exposes_text_causal_lm(tmp_path):
    transformers = pytest.importorskip("transformers")
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5TextConfig
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

    from bonsai_forensics.modeling import TextOnlyCausalLM, load_text_causal_lm
    from bonsai_forensics.targets import target_coverage

    text = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=["linear_attention", "full_attention"],
    )
    vision = Qwen3_5VisionConfig(
        hidden_size=32, depth=1, num_heads=4, intermediate_size=64, out_hidden_size=32)
    model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(
        text_config=text, vision_config=vision))
    text_model = transformers.Qwen3_5ForCausalLM(text)
    report = target_coverage(text_model, profile="qwen3_5")
    assert report["selected_linear_tensors"] == report["expected_selected_linear_tensors"]
    assert report["selected_linear_tensors"] == 14
    model.save_pretrained(tmp_path)

    loaded = load_text_causal_lm(tmp_path, dtype=torch.float32, device="cpu")
    assert isinstance(loaded, TextOnlyCausalLM)
    assert len(loaded.model.layers) == 2
    output = loaded(torch.randint(0, 64, (1, 8)))
    assert output.logits.shape == (1, 8, 64)
    assert torch.isfinite(output.logits).all()
