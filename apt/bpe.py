import regex as re
import os
from collections import Counter, defaultdict
from itertools import pairwise
from typing import BinaryIO
from multiprocessing import Pool
from functools import lru_cache


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize(text: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    if special_tokens:
        pattern = "|".join(map(re.escape, sorted(special_tokens, key=len, reverse=True)))
        segments = re.split(pattern, text)
    else:
        segments = [text]

    return Counter(
        tuple(bytes([b]) for b in m.encode("utf-8"))
        for s in segments
        for m in re.findall(PAT, s)
    )


def merge(counts: dict[tuple[bytes, ...], int], num_merges: int) -> list[tuple[bytes, bytes]]:
    merges = []
    ts, cs = list(counts), list(counts.values())
    freq, where = Counter(), defaultdict(set)
    for w, t in enumerate(ts):
        for p in pairwise(t):
            freq[p] += cs[w]
            where[p].add(w)
    for _ in range(num_merges):
        if not freq:
            break
        pair = max(freq, key=lambda k: (freq[k], k))
        for w in list(where[pair]):
            t, c = ts[w], cs[w]
            for p in pairwise(t):
                freq[p] -= c
                where[p].discard(w)
            new_t = []
            i = 0
            while i < len(t):
                if i < len(t) - 1 and (t[i], t[i+1]) == pair:
                    new_t.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_t.append(t[i])
                    i += 1
            ts[w] = tuple(new_t)
            for p in pairwise(ts[w]):
                freq[p] += c
                where[p].add(w)
        freq = +freq
        merges.append(pair)
    return merges


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize_chunk(path: str | os.PathLike, start: int, end: int, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    return pretokenize(chunk, special_tokens=special_tokens)


def pretokenize_file(path: str | os.PathLike, num_processes: int, special_tokens: list[str], chunks_per_process: int = 16) -> dict[tuple[bytes, ...], int]:
    with open(path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes * chunks_per_process, b"<|endoftext|>")

    with Pool(num_processes) as pool:
        counters = pool.starmap(pretokenize_chunk, [(path, start, end, special_tokens) for start, end in pairwise(boundaries)])

    counts = Counter()
    for c in counters:
        counts.update(c)

    return counts


def train_bpe(path: str | os.PathLike, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    counts = pretokenize_file(path=path, num_processes=os.cpu_count() or 1, special_tokens=special_tokens)
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


@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d


BYTE_TO_UNICODE = gpt2_bytes_to_unicode()
UNICODE_TO_BYTE = {value: key for key, value in BYTE_TO_UNICODE.items()}


def bytes_to_str(bytestring: bytes) -> str:
    return "".join(BYTE_TO_UNICODE[b] for b in bytestring)


def str_to_bytes(s: str) -> bytes:
    return bytes([UNICODE_TO_BYTE[c] for c in s])