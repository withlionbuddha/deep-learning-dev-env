DEEP-LEARNING-ENV(build Context)/
├── .devcontainer/
├── drive/
├── license/
├── .dockerignore
├── .env
├── .env.example
├── .gitignore
├── .DIRS_ARCHITECTURE.md
├── docker-compose-linux-nvidia-cuda.yml
├── docker-compose-wsl-intel-nvidia-cuda.yml
├── docker-compose-wsl-intel-openvino.yml
├── docker-compose-wsl-intel-pytorch-xpu.yml
├── docker-compose-wsl-intel-tensorflow-xpu.yml
├── docker-compose-wsl-intel-xpu.yml
├── Dockerfile.intel-nvidia-cuda
├── Dockerfile.intel-openvino
├── Dockerfile.intel-pytorch-xpu
├── Dockerfile.intel-tensorflow-xpu
├── Dockerfile.intel-xpu
└── README.md
└── ${PROEJCT_NAME}/


Host(Windows)
 ├── ${SOURCE_PATH}
 │        │
 │        └──── bind mount
 │
 └── ${DRIVE_PATH}
 |        │
 |        └──── readonly bind mount
 |
 └── ${AI_COUMPUTE_BENCHMARK_PATH}
 |        │
 |        └──── bind mount


Container Runtime Context
 ├── filesystem
 │     ├── /home/workspace/${PROJECT_NAME}
 │     │      ↳ host source_path bind mount
 │     │
 │     └── /home/drive
 │            ↳ readonly host drive_path bind mount
 │     ├── /home/workspace/ai-compute-benchmark
 │            ↳ host ai_compute_benchmark_path bind mount
 │
 ├── cwd = /home/workspace
 ├── process = jupyter/python/bash
 ├── env vars
 ├── network namespace
 └── writable layer
