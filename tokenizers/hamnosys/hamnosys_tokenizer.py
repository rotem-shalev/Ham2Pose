from pathlib import Path
from typing import List
import torch
from fontTools.ttLib import TTFont

from data.collator import zero_pad_collator

MAX_TEXT_LEN = 100


class HamNoSysTokenizer:

    def __init__(self):
        self.pad_token_id = 0
        self.bos_token_id = 1

        self.font_path = Path(__file__).parent.joinpath("HamNoSysUnicode.ttf")

        with TTFont(self.font_path) as font:
            tokens = [chr(key) for key in font["cmap"].getBestCmap().keys()]

        self.i2s = {(i + 2): c for i, c in enumerate(tokens)}
        self.s2i = {c: i for i, c in self.i2s.items()}

    def __len__(self):
        return len(self.i2s) + 2

    def tokenize(self, text: str):
        return [self.bos_token_id] + [self.s2i[c] for c in text]

    def __call__(self, texts: List[str], device=None):
        all_tokens = [self.tokenize(text) for text in texts]

        tokens_batch = zero_pad_collator([{
            "tokens_ids": torch.tensor(tokens, dtype=torch.long, device=device),
            "attention_mask": torch.ones(len(tokens), dtype=torch.bool, device=device),
            "positions": torch.arange(0, len(tokens), dtype=torch.long, device=device)
        } for tokens in all_tokens])
        # In transformers, 1 is mask, not 0
        tokens_batch["attention_mask"] = torch.logical_not(tokens_batch["attention_mask"])

        return tokens_batch


if __name__ == "__main__":
    tokenizer = HamNoSysTokenizer()
    hamnosys = [
        "",  # bsl one
        "",  # gsl one
        "\ue000\ue071",
        "\ue000\ue071\ue012\ue029\ue03f\ue089\ue0c6\ue0d8 \ue000\ue071"
    ]
    print(hamnosys)
    print(tokenizer(hamnosys))
