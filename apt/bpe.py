import regex as re
from collections import Counter
from itertools import pairwise


def pretokenize(text: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    if special_tokens:
        pattern = "|".join(map(re.escape, sorted(special_tokens, key=len, reverse=True)))
        segments = re.split(pattern, text)
    else:
        segments = [text]

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    return Counter(
        tuple(bytes([b]) for b in m.encode("utf-8"))
        for s in segments
        for m in re.findall(PAT, s)
    )


def merge(counts: dict[tuple[bytes, ...], int], num_merges: int) -> list[tuple[bytes, bytes]]:
    merges = []
    for _ in range(num_merges):
        freq = Counter()
        for t, c in counts.items():
            for p in pairwise(t):
                freq[p] += c
        pair = max(freq, key=lambda k: (freq[k], k))
        new = Counter()
        for t, c in counts.items():
            new_t = []
            i = 0
            while i < len(t):
                if i < len(t) - 1 and (t[i], t[i+1]) == pair:
                    new_t.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_t.append(t[i])
                    i += 1
            new[tuple(new_t)] += c
        counts = new
        merges.append(pair)
    return merges

def run_train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    counts = pretokenize(text, special_tokens=special_tokens)
    num_merges = vocab_size - len(special_tokens) - 256
    merges = merge(counts=counts, num_merges=num_merges)
    vocab = {i: bytes([i]) for i in range(256)}
    i = 256
    for t in special_tokens:
        vocab[i] = t.encode(encoding="utf-8")
        i += 1
    for a, b in merges:
        vocab[i] = a + b
        i += 1
    return (vocab, merges)