DEBIAN-GPU-JUPYTER-ENV(build Context)/
├── .devcontainer/
├── drive/
├── .dockerignore
├── .env
├── .env.example
├── .gitignore
├── .DIRS_ARCHITECTURE.md
├── docker-compose-intel-i7.yml
├── docker-compose-nvidia.yml
├── Dockerfile.intel-i7
├── Dockerfile.nvidia
└── ${PROEJCT_NAME}/


Host(Windows)
 ├── ${SOURCE_PATH}
 │        │
 │        └──── bind mount
 │
 └── ${DRIVE_PATH}
          │
          └──── readonly bind mount


Container Runtime Context
 ├── filesystem
 │     ├── /home/workspace/${PROEJCT_NAME}
 │     │      ↳ host source bind mount
 │     │
 │     └── /home/drive
 │            ↳ readonly host drive bind mount
 │
 ├── cwd = /home/workspace
 ├── process = jupyter/python/bash
 ├── env vars
 ├── network namespace
 └── writable layer
