from dataclasses import dataclass


@dataclass
class GPTconfig:
    block_size: int = 1024
    vocab_size: int = 50257
    # 50000 BPE merges  + 256 bytes tokens + 1 |<endoftext>|
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
