# Deep Learning Dev Environment — README v0.1.0

> 이 문서는 현재 README 개편 이전의 내용을 보존하기 위한 v0.1.0 아카이브입니다.
>
> 원본 README의 라이선스, 저장소 개요, 환경별 Compose/Dockerfile 구조, 포트 설계, Dev Container 사용법, Intel/NVIDIA 환경 설명을 보존합니다.

## 0. 라이선스(License)

기존 README에 정의된 연구 사용권 및 Eligible Academic Users / Restricted Educational or Promotional Entities 관련 조항을 적용합니다.

## 1. 개요

이 저장소는 딥러닝 실습과 개발 환경을 Docker Compose와 VS Code Dev Containers로 분리 관리하기 위한 개발 환경입니다.

```text
Base repository : git@github.com:withlionbuddha/oa-docker.git
Base branch     : experiment/debian-gpu-jupyter-env
Current repo    : git@github.com:withlionbuddha/deep-learning-dev-env.git
```

## 2. 주요 목적

- Windows + WSL 2 + Docker Desktop 기반 딥러닝 실습 환경 구성
- Linux + NVIDIA CUDA 기반 딥러닝 실습 환경 구성
- Jupyter Notebook / JupyterLab 실행 환경 제공
- TensorFlow XPU, PyTorch XPU, OpenVINO, NVIDIA CUDA 환경 분리
- Intel GPU/XPU, NVIDIA GPU, CPU 기반 실험 환경을 목적별로 관리
- VS Code Dev Containers에서 필요한 환경만 선택 실행
- Docker Compose profile 기반 서비스 실행 관리
- 컨테이너 내부 Jupyter 포트는 `8888`로 통일
- 외부 접속 포트는 `.env`와 compose 파일에서 환경별로 분리

## 3. 기존 환경 구성

기존 README에서는 TensorFlow XPU, PyTorch XPU, OpenVINO, NVIDIA CUDA, Numpy XPU 환경을 각각 별도 Dockerfile/Compose/Dev Container로 관리했습니다.

## 4. 저장소 구조

```text
deep-learning-dev-env/
├─ .devcontainer/
├─ docker-compose-wsl-intel-tensorflow-xpu.yml
├─ docker-compose-wsl-intel-pytorch-xpu.yml
├─ docker-compose-wsl-intel-openvino.yml
├─ docker-compose-wsl-intel-nvidia-cuda.yml
├─ docker-compose-linux-nvidia-cuda.yml
├─ docker-compose-wsl-intel-xpu.yml
├─ Dockerfile.intel-tensorflow-xpu
├─ Dockerfile.intel-pytorch-xpu
├─ Dockerfile.intel-openvino
├─ Dockerfile.intel-nvidia-cuda
├─ Dockerfile.intel-xpu
├─ .env
└─ README.md
```

## 5. 공통 실행 환경

Windows 환경은 WSL 2, Docker Desktop, VS Code, Dev Containers를 기반으로 하며 Intel GPU 환경에서는 `/dev/dxg`와 `/usr/lib/wsl`을 컨테이너에 연결합니다.

## 6. 환경 변수

기존 환경에서는 `SOURCE_PATH`, `DRIVE_PATH`, `AI_COMPUTE_BENCHMARK_PATH`, `PROJECT_NAME`과 framework별 Jupyter host port를 `.env`에서 관리했습니다.

## 7. 포트 설계

컨테이너 내부 Jupyter 포트는 `8888`로 통일하고 host port만 framework별로 분리했습니다.

## 8. Dev Container

각 환경은 `.devcontainer/*/devcontainer.json`에서 대응하는 Docker Compose service를 선택하여 VS Code Dev Container로 실행하도록 구성했습니다.

## 9. Intel GPU / XPU

WSL 2에서 Intel GPU를 사용하기 위해 `/dev/dxg`, `/usr/lib/wsl` 및 Intel OpenCL/Level Zero runtime을 사용합니다.

## 10. NVIDIA CUDA

NVIDIA 환경은 NVIDIA driver와 NVIDIA Container Toolkit을 전제로 별도의 CUDA Docker/Compose 환경으로 관리합니다.

## 11. 이후 변경 방향

v0.1.0 이후 Intel Iris Xe 개발 환경은 중복 Dockerfile을 줄이고 framework를 선택하는 공통 구조로 정리합니다.

```text
FRAMEWORK=pytorch
# 또는
FRAMEWORK=tensorflow
```

공통 Intel GPU runtime 위에 PyTorch XPU와 TensorFlow/ITEX XPU stage를 분리하고 최종 runtime stage에서 선택하는 방식으로 전환합니다.
