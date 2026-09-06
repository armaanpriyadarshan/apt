import argparse
import time
import os
import json
import apt.bpe as bpe


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input",
        type=str,
        default="data/TinyStoriesV2-GPT4-train.txt"
    )

    parser.add_argument(
        "-v", "--vocab_size",
        type=int,
        default=10000
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default="data/tokenizer"
    )

    args = parser.parse_args()

    start_time = time.perf_counter()
    vocab, merges = bpe.train_bpe(args.input, args.vocab_size, ["<|endoftext|>"])
    end_time = time.perf_counter()

    os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({bpe.bytes_to_str(value): key for key, value in vocab.items()}, f)

    with open(os.path.join(args.output, "merges.txt"), "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{bpe.bytes_to_str(a)} {bpe.bytes_to_str(b)}\n")

    elapsed = end_time - start_time
    print(f"Training took {elapsed:.6f} seconds")
    print(f"Longest token was {max(vocab.values(), key=len)}")

if __name__ == "__main__":
    main()