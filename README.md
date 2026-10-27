# DeepSeek RTX inference image

An auditable Linux AMD64 image for DeepSeek V4.1 Flash 1017 on RTX PRO 6000 Blackwell 96GB GPUs. The runtime is a pinned vLLM release with matching FlashInfer SM120 kernel packages, isolated Vast tooling, and explicit network, GPU and budget gates.

This is a candidate until GPU validation and full-model generation pass. CI tests package identity, runtime imports, control logic and inert startup without GPU or network access. It does not measure model performance.

See [build details](rtx96/README.txt) and [operator steps](rtx96/runtime/OPERATOR.txt). No checkpoint, account records or credentials are distributed. Startup never downloads weights or starts inference automatically.

Published images: `ghcr.io/sasank-c/deepseek-rtx96`. Use an immutable registry digest after a successful workflow, not the moving `candidate` tag.
