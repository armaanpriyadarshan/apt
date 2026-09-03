# apt

Armaan Pre-trained Transformer. A decoder-only Transformer language model and its
training stack, written from scratch in PyTorch.

## The constraint

Nothing from `torch.nn`, `torch.nn.functional` or `torch.optim`, with four
exceptions: `nn.Parameter`, the `nn` container classes (`Module`, `ModuleList`,
`Sequential`), and the `torch.optim.Optimizer` base class. Everything else in
torch is fair game. Linear, embedding, RMSNorm, SiLU, SwiGLU, softmax, attention,
RoPE, cross entropy and AdamW all get written here rather than imported.

## Layout

    apt/        the model and training code
    tests/      the test suite and its fixtures, unmodified
    docs/       the handout this is built against

`tests/adapters.py` is glue. Each function there wires one piece of `apt/` into
the test that checks it, and holds no logic of its own.

## Running the tests

    uv run pytest                        # everything
    uv run pytest -k test_linear         # one component
    uv run pytest tests/test_model.py    # the architecture

## Scope

The model and the training loop: the handout's sections 3, 4 and 5. Section 2 is
a BPE tokenizer and is skipped on purpose. It does not transfer to robot
policies, and the handout puts it at 30 GB of RAM for TinyStories against 15 GB
on this machine, so `tests/test_tokenizer.py` and `tests/test_train_bpe.py` stay
red.

## Where this came from

The tests, fixtures and handout are Stanford CS336 assignment 1 (spring 2026,
handout 26.0.3), used under the MIT license in `LICENSE`. An untouched clone sits
at `../assignment1-basics` for pulling upstream fixes; `docs/upstream-CHANGELOG.md`
is their record of what has changed.

`AGENTS.md` and `CLAUDE.md` carry over from that repo unchanged. They say an
agent explains, reviews and diagnoses, and never writes the implementation. The
point of the project is that the code in `apt/` is written by hand.
