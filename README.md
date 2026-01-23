# Nano-vLLM-v1

A lightweight [vLLM](https://github.com/vllm-project/vllm) implementation built from scratch. While built upon the foundation of [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm), this project significantly re-engineers the core architecture to **reproduce the [vLLM v1 scheduler](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)** and introduces **Chunked Prefill**.

## Key Features

* 🚀 **Fast offline inference** - Comparable online inference speeds and offline throughput performance to vLLM v1.
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Paged Attention, Prefix Caching, Chunked Prefill, Tensor Parallelism, Torch Compilation, and full reproduction of the vLLM v1 scheduling strategy, etc.
