# Qwen3.8-Flash-Next from the SSD of a 16 GB MacBook

Code for running [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), a 180B-parameter
mixture-of-experts model, on a base M4 MacBook Pro with 16 GB of RAM: about 9 GB in memory, 112 GiB of weights on the
internal SSD, 5.7 tokens/s on a benchmark prompt and 4.6 tokens/s in an agent session under stock OpenCode.

Weights: [jalenluorion/Qwen3.8-Flash-Next-moved-store](https://huggingface.co/jalenluorion/Qwen3.8-Flash-Next-moved-store)
on Hugging Face (experts at 4-bit, the n-gram table at 8-bit, trained routers).

The released weights are frozen. The only trained part is a small router per layer (576M parameters in total, 0.3% of
the model) that chooses each layer's experts four layers before that layer runs, so the SSD reads them while the
layers in between compute.

## Quick start

Apple Silicon with 16 GB or more, and about 140 GB of free disk (the download is 123 GiB; the first run writes a
4.3 GiB cache next to it):

```bash
git clone https://github.com/jalenluorion/Qwen3.8-Flash-Next-moved-store
cd Qwen3.8-Flash-Next-moved-store
pip install -r requirements.txt
hf download jalenluorion/Qwen3.8-Flash-Next-moved-store --local-dir model --exclude "routers3/*"

python mlx_run.py --dir model                          # chat; -v prints the per-turn statistics
python mlx_run.py --dir model --prompt "Explain what a mixture of experts is." --max-new 200
python mlx_run.py --dir model --bench --max-new 120    # the timing JSON behind the numbers below
cd demo && ./demo.sh                                   # the same engine behind an OpenAI-compatible API, in OpenCode
```

The first start quantises the in-memory part to 8-bit once and caches it as `model/resident_int8.safetensors`. On a 16 GB machine close other apps first: the model takes about 9 GB, and macOS
starts compressing its memory when the rest of the machine wants more than what is left.

`demo.sh` needs [OpenCode](https://opencode.ai) (tested with 1.19.0). It starts the server if it is not running and
opens OpenCode with OpenCode's stock prompt and tools. See [demo/README.md](demo/README.md).

## How it works

A mixture-of-experts model uses a small part of itself per token: here 10 of 512 experts in each of 48 layers. In the
released model each layer picks its experts from the hidden state at that layer, so an expert is needed the moment it
is chosen, and every expert has to sit in memory in case it is picked.

Here the experts are stored once on the SSD, one contiguous 2.6 MiB record each. A small router per layer reads the
token's state **four layers earlier** and names the experts that layer will want. Those records are read while the
four layers in between compute. When the layer runs, the released router makes its usual choice among the experts
that are in memory.

- **Prefetch mode (default).** Nothing is read during a layer's own computation. If the routers named the wrong
  expert, the released router chooses among what is in memory: the output changes slightly, nothing stalls. `--keep`
  sets how many tokens a fetched expert stays in memory while it keeps being named (default 3), `--max-slots` the
  expert slots per layer (default 24, 3.0 GB in total).
- **Exact mode (`--mode exact`).** The released router's choice is always honoured; an expert that is not in memory
  is read on the spot. With `--routers /nonexistent` nothing is read ahead at all, which is the fetch-on-demand
  baseline in the table below.

The model's n-gram memory (a 51.2B-parameter hash table) is a second store. A token looks up 16 rows whose addresses
follow from the previous token ids alone, so they are read the moment a token is sampled.

| part | parameters | where |
|---|---|---|
| attention layers, shared experts, norms, embeddings, output head, the released router gates | 4.9B | memory, 8-bit (4.3 GiB), the input embedding read from disk one row per token |
| lag-4 routers | 576M | memory, fp16 (1.1 GB) |
| 24,576 routed experts, 4-bit | 120.8B | SSD, 63.4 GiB |
| n-gram memory, 8-bit | 51.2B | SSD, 48.3 GiB |

The release's multi-token-prediction head and vision tower are not included.

## Speed and memory

Base M4 MacBook Pro, 16 GB, internal SSD, `mlx_run.py --bench --max-new 120`, one prompt. The OS file cache is
bypassed on the stores, so every byte counted was read from the drive.

| configuration | tokens/s (each run) | read per token | memory (MLX peak) |
|---|---|---|---|
| experts chosen four layers early (default) | 5.70, 5.84, 5.55 | 488 MB | 8.5 GB |
| chosen early, experts kept only across adjacent tokens (`--keep 0`) | 4.24, 4.34 | 682 MB | 8.5 GB |
| fetched on demand (exact mode, no routers), default keep window | 2.42, 2.50, 2.49 | 587 MB | 7.5 GB |
| fetched on demand, kept only across adjacent tokens (`--keep 1`) | 2.16, 2.04 | 741 MB | 7.5 GB |
| fetched on demand, nothing kept (`--keep 0`) | 1.44, 1.34 | 1,268 MB | 7.5 GB |

For comparison, stock llama.cpp 1.0.0 (Homebrew) with Unsloth's UD-Q4_K_XL file of the same model (111 GB), on the same Mac
and prompt: with its default settings the load never finishes on 16 GB (it repacks the 4-bit weights into ordinary
memory); with memory-mapped weights, CPU only and `--no-repack` it runs at 0.45 and 0.39 tokens/s (a third run gave
0.11), and at 0.20 with the GPU allowed. That is a different 4-bit file and a CPU computation, so it shows what the
stock tool gives, not a controlled ablation; the fetch-on-demand rows above are the like-for-like comparison. Logs: [results/llama_cpp](results/llama_cpp).

At the default a token takes 170 to 180 ms, about 100 ms of it waiting on expert reads, and 83% of the released router's top-10
are in memory when the layer runs. The drive sets the speed: the same engine with reads skipped (`--sim-reads`,
timing only) runs at 10.5 tokens/s.

In the recorded agent session (stock OpenCode, an 8k to 9k-token context, thinking on) the ten requests ran at 4.3 to
5.4 tokens/s, 4.6 on average, reading 545 to 706 MB per token at about 3.0 GB/s. Activity Monitor showed about 9 GB
for the process.
The logs are in [results/mac_bench](results/mac_bench).

## Quality

Loss on held-out text against the released bf16 model (wikitext-103 test / FineWeb-Edu / UltraChat in the chat
template; released: 1.611 / 2.089 / 1.828 nats), measured on GPUs with each step added to the previous ones:

| configuration | loss gap, nats |
|---|---|
| experts at 4-bit | +0.013 / +0.007 / -0.028 |
| + n-gram memory at 8-bit | +0.014 / +0.008 / -0.026 |
| + in-memory weights at 8-bit | +0.014 / +0.008 / -0.037 |
| + experts chosen four layers early (the configuration that runs on the Mac) | +0.027 / +0.019 / -0.008 |

The chat set comes out below the released model once the weights are quantised. We have not explained that and do
not read it as a gain. The cost of choosing early alone, against the same quantised model with every expert in
memory, is +0.011 / +0.011 / +0.019 nats. Per window that gap is a small mean with wide two-sided noise (chat: median
+0.016, standard deviation 0.133 over 128 windows).

Not measured: thinking traces, code and tool calls, which the routers were not trained on, and task benchmarks.

## What is in the repo

| path | what it is |
|---|---|
| `mlx_run.py` | the engine: chat, OpenAI-compatible server, benchmark and check modes |
| `qwen4_exp/` | the model's MLX definition, text only, vendored from the community MLX port ([nopmobiel/Qwen3.8-Flash-Next-MLX-4bit](https://huggingface.co/nopmobiel/Qwen3.8-Flash-Next-MLX-4bit)) on [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) |
| `demo/` | `demo.sh` and `opencode.json` for stock OpenCode, plus the two small files of the bug-fix task in the video |
| `convert.py`, `check.py` | rebuild the stores from the released weights (resumable, one shard at a time), and the check that gates the conversion |
| `make_tiny.py`, `test_tiny.py` | a random tiny model with the release's structure, and the runner tested against the reference model on it |
| `drivebench.c` | the drive's rate for the store's read pattern, without Python or the model |
| `results/` | the logs and JSON behind the tables above |

The script that trained the routers is not in the repo yet. They were trained on GPUs by imitation of the released
routers with the model frozen, on 20.5M tokens of wikitext-103, FineWeb-Edu and UltraChat.

## Limits

- This is a retrofit of a model that was not built to be read from a drive. A token reads about 490 MB on a plain
  prompt and 550 to 700 MB in an agent session, so the laptop's SSD sets the speed.
- The experts are 4-bit and the rest 8-bit on a 16 GB Mac. The numbers above are for that configuration, not for the
  released bf16 weights.
- Timings are from one machine at batch 1. A machine under memory pressure is slower: macOS compresses the model's
  memory when other apps need it.
- Text only. No vision tower, no multi-token prediction.

## Paper

Paper soon. This repository retrofits a released model. Models built for this from the first training step are what
we call prefetchable, and they are the subject of the paper.

## License

The code in this repository is MIT (see `LICENSE`), except `qwen4_exp/`, which comes from a port distributed under
the Qwen Community License 1.0 (`LICENSE-QWEN`) and includes helper functions from mlx-vlm (MIT). The weights are
under the Qwen Community License 1.0.
