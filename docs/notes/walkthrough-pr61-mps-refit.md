# Walkthrough: #61, the refit on Apple's GPU (`--device mps`)

Merged as 5ae9c5a. The rung-4 PR walkthroughs up to #60 are PDFs outside the repo
(`~/Desktop/Claude/poker/code-walkthroughs/`). This one and #62's are kept here as Markdown.

## The problem it solves

Every Deep CFR iteration refits each seat's network from scratch, and that refit is about 95%
of the iteration. On the CPU it kept only about 1.5 cores busy. Each Adam step on a
10,000-row minibatch is too small to spread across 18 cores, so more threads did not help.
The step is a dense matrix multiply, which is exactly what a GPU is for. The M5 Pro has one,
reachable from PyTorch as the `mps` device.

## The picture

    reservoir (numpy, on cpu)
        │  copied to the device ONCE per refit:
        │  rows, t, targets, the feature layout's two halves, the legal gather
        ▼
    ┌──────────── on the GPU, 4,000 times ────────────┐
    │ draw a minibatch index (from a CPU generator)    │
    │ gather its features on the device                │
    │ forward → _weighted_loss → backward → clip → step│
    └──────────────────────────────────────────────────┘
        ▼
    net copied back to the cpu, eval() mode → materialised into the table as before

The rule is to move the data once, then never return to numpy inside the loop. A version that
built each batch's features in numpy and copied them over every step would spend its time on
transfers.

## Same rule, not the same bits

An mps fit uses the same initial weights, the same minibatch indices (drawn from a CPU
`torch.Generator`, so they match the CPU fit's), and the same loss function. Only the order of
floating-point arithmetic changes, so the weights agree to about 1e-3, not bit for bit. That is
why `device` is not in `SAMPLE_FIELDS`. A resume may switch devices, as it may change the worker
count W, because neither changes which rows are collected. Old `state.json` files load with the
default `"cpu"`.

## Measured (M5 Pro, the live 4 × 256, 8,000-step run)

| | cpu | mps | speed-up |
|---|---|---|---|
| fit, per seat | 127 s | 34 s | 3.7× |
| whole iteration | 269 s | 87 s | 3.1× |
| one step, 4 × 256, on a 17M-row memory | 23.9 ms | 5.0 ms | 4.8× |
| one step, 3 × 128 | 9.5 ms | 2.5 ms | 3.8× |

That run (`runs/sweep-256x4-s8000`) stopped at t = 37 on the CPU and resumed on mps from its
save, and its losses carried on at the same scale.

## The pieces, by purpose

- **`FitConfig.device`** (`fit.py`). The one switch: `"cpu"` keeps the old loop untouched,
  `"mps"` sends the refit to the GPU. It is validated on construction, so a typo fails before
  a 70-second refit rather than inside it.
- **`fit_from_scratch`** (`fit.py`). It still owns the CPU loop and the divergence check (a
  non-finite weight is an error, because a NaN net would materialise to uniform play with
  nothing in the log). It now hands a non-CPU device to `_fit_on_device`.
- **`_fit_on_device`** (`fit.py`). The same loop, run where the data lives: it copies the
  memory and the feature tables over once, gathers features on the device, and returns the net
  on the CPU in eval mode. `device` is a parameter rather than read from the config, so a test
  can run this exact loop on the CPU of a machine with no GPU and check it against the
  ordinary fit.
- **`_weighted_loss`** (`fit.py`). The one copy of the t-weighted squared error that both
  paths call. With one copy, the CPU and GPU fits cannot drift apart in what they minimise,
  only in arithmetic order.
- **`DeepConfig.device`** (`train.py`). It carries the choice from the command line into each
  seat's refit, and it is deliberately left out of the fields a resume must match.
- **`--device cpu|mps`** (`scripts/deepcfr/run_deep_cfr.py`). The command-line face of it.

## Tests

`tests/deepcfr/test_fit.py`: an mps fit tracks the CPU fit from the same seed to 1e-3 and comes
back on the CPU in eval mode (skipped on a machine without an Apple GPU), and an unknown device
is refused. `tests/deepcfr/test_train.py`: a resume may change the device.
