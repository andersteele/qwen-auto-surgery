"""A tiny random 8-bit Qwen3.5-MoE with the real architecture, for testing the surgery code."""
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3_5_moe

TEXT = dict(
    model_type="qwen3_5_moe_text", hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
    num_key_value_heads=2, head_dim=16, vocab_size=512, rms_norm_eps=1e-6, max_position_embeddings=512,
    num_experts=8, num_experts_per_tok=2, moe_intermediate_size=64, shared_expert_intermediate_size=64,
    full_attention_interval=4, linear_conv_kernel_dim=4, linear_key_head_dim=16, linear_value_head_dim=16,
    linear_num_key_heads=2, linear_num_value_heads=4, tie_word_embeddings=False,
    rope_parameters={"rope_theta": 10000, "partial_rotary_factor": 0.25, "mrope_section": [2, 1, 1]},
    attn_output_gate=True,
)


class Tok:
    eos_token_ids = {511}

    def encode(self, text, add_special_tokens=False):
        return [b % 512 for b in text.encode()]

    def decode(self, ids):
        return "".join(chr(32 + (i % 90)) for i in ids)

    def apply_chat_template(self, messages, add_generation_prompt=True, enable_thinking=False, **_):
        """The shape of the real template: a marked turn per message, then a hand-over to the assistant."""
        ids = []
        for m in messages:
            ids += [500, 501 if m["role"] == "user" else 502] + self.encode(m["content"]) + [503]
        if add_generation_prompt:
            ids += [500, 504, 505] + ([] if enable_thinking else [506, 507])
        return ids


def build(seed=0, dtype=mx.bfloat16):
    mx.random.seed(seed)
    args = qwen3_5_moe.ModelArgs.from_dict({"model_type": "qwen3_5_moe", "text_config": TEXT})
    model = qwen3_5_moe.Model(args)
    model.set_dtype(dtype)
    pred = getattr(model, "quant_predicate", None)

    def class_predicate(path, m):
        if not hasattr(m, "to_quantized"):
            return False
        return pred(path, m) if pred else True

    nn.quantize(model, group_size=64, bits=8, class_predicate=class_predicate)
    mx.eval(model.parameters())
    model.eval()
    return model, Tok()


if __name__ == "__main__":
    m, t = build()
    for k, v in tree_flatten(m.parameters())[:40]:
        print(k, v.shape, v.dtype)
