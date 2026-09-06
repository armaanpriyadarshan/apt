import json
import regex as re
from apt.bpe import str_to_bytes, PAT
from itertools import pairwise
from collections.abc import Iterable, Iterator


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.bytes_to_id = {b: i for i, b in self.vocab.items()}
        for t in self.special_tokens:
            b = t.encode("utf-8")
            if b not in self.bytes_to_id:
                i = len(self.vocab)
                self.vocab[i] = b
                self.bytes_to_id[b] = i
        self.ranks = {p: i for i, p in enumerate(self.merges)}


    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab = {i: str_to_bytes(b) for b, i in json.load(f).items()}

        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges = [(str_to_bytes(a), str_to_bytes(b)) for a, b in (line.split() for line in f.read().splitlines())]

        return cls(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            pattern = "(" + "|".join(map(re.escape, sorted(self.special_tokens, key=len, reverse=True))) + ")"
            segments = re.split(pattern, text)
        else:
            segments = [text]

        ids = []
        for s in segments:
            if s in self.special_tokens:
                ids.append(self.bytes_to_id[s.encode("utf-8")])
                continue

            for m in re.findall(PAT, s):
                t = tuple(bytes([b]) for b in m.encode("utf-8"))
                while len(t) > 1:
                    pair = min(pairwise(t), key=lambda p: self.ranks.get(p, float("inf")))
                    if pair not in self.ranks:
                        break
                    new_t = []
                    i = 0
                    while i < len(t):
                        if i < len(t) - 1 and (t[i], t[i+1]) == pair:
                            new_t.append(pair[0] + pair[1])
                            i += 2
                        else:
                            new_t.append(t[i])
                            i += 1
                    t = tuple(new_t)
                ids.extend(self.bytes_to_id[b] for b in t)

        return ids


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            yield from self.encode(s)


    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")