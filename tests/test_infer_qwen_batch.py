import torch

from scripts import infer_qwen


class FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(item) for item in token_ids)


def test_decode_batch_new_tokens_slices_after_padded_prompt_width():
    generated = torch.tensor(
        [
            [101, 102, 201, 202],
            [0, 103, 301, 302],
        ]
    )

    decoded = infer_qwen.decode_batch_new_tokens(FakeTokenizer(), generated, prompt_width=2)

    assert decoded == ["201 202", "301 302"]


def test_infer_help_exposes_batch_size_argument():
    parser = infer_qwen.build_arg_parser()

    help_text = parser.format_help()

    assert "--batch-size" in help_text


def test_infer_help_exposes_prompt_variant_argument():
    parser = infer_qwen.build_arg_parser()

    help_text = parser.format_help()

    assert "--prompt-variant" in help_text


def test_infer_help_exposes_retrieval_candidate_arguments():
    parser = infer_qwen.build_arg_parser()

    help_text = parser.format_help()

    assert "--retrieval-candidates" in help_text
    assert "--retrieval-top-k" in help_text
