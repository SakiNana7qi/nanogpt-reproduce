import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/autodl-tmp/.cache/huggingface"
os.makedirs("/root/autodl-tmp/tmp", exist_ok=True)
os.makedirs("/root/autodl-tmp/datasets_cache", exist_ok=True)
os.environ["TMPDIR"] = "/root/autodl-tmp/tmp"  # 重定向所有临时文件
os.environ["HF_DATASETS_CACHE"] = "/root/autodl-tmp/datasets_cache"  # datasets专用缓存
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]


def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (
        tokens_np < 2**16
    ).all(), "token dictionary didn't in range"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


def main():
    # cache_dir = "/root/autodl-tmp/datasets_cache"
    local_dir = "/root/autodl-tmp/edu_fineweb10B"
    remote_name = "sample-10BT"
    shard_size = int(1e8)

    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize, fw, chunksize=16):
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(
                        total=shard_size, unit="tokens", desc=f"Shard {shard_index}"
                    )
                progress_bar.update(len(tokens))
            else:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(
                    DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}"
                )
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count : token_count + remainder] = tokens[
                    :remainder
                ]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(
                DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}"
            )
            write_datafile(filename, all_tokens_np[:token_count])


if __name__ == "__main__":
    # Windows多进程支持
    mp.freeze_support()

    try:
        main()
    except Exception as e:
        import traceback

        traceback.print_exc()
        input("按Enter退出...")
