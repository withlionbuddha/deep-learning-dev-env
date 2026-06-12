# Deep Learning Dev Environment

> Docker, WSL 2, VS Code Dev Containers 기반의 딥러닝 개발 환경 저장소입니다.  
> TensorFlow XPU, PyTorch XPU, OpenVINO, NVIDIA CUDA 환경을 목적별로 분리하여 관리합니다.

![Docker](https://img.shields.io/badge/Docker-Dev%20Container-blue)
![Python](https://img.shields.io/badge/Python-3.10-yellow)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook%20%7C%20Lab-orange)
![WSL2](https://img.shields.io/badge/WSL-2-green)
![CUDA](https://img.shields.io/badge/NVIDIA-CUDA-76B900)
![Intel](https://img.shields.io/badge/Intel-XPU-0071C5)

---

## 1. 개요

이 저장소는 딥러닝 실습과 개발 환경을 Docker Compose와 VS Code Dev Containers로 분리 관리하기 위한 개발 환경입니다.

기반 저장소는 다음과 같습니다.

```text
Base repository : git@github.com:withlionbuddha/oa-docker.git
Base branch     : experiment/debian-gpu-jupyter-env
Current repo    : git@github.com:withlionbuddha/deep-learning-dev-env.git
```

---

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

---

## 3. 지원 환경

| 구분 | Compose 파일 | Service | Profile | 외부 포트 예시 | 내부 Jupyter 포트 | Dockerfile |
|---|---|---|---|---:|---:|---|
| WSL Intel TensorFlow XPU | `docker-compose-wsl-intel-tensorflow-xpu.yml` | `intel-tensorflow-xpu-dev` | `tensorflow-xpu` | `8889` | `8888` | `Dockerfile.intel-tensorflow-xpu` |
| WSL Intel PyTorch XPU | `docker-compose-wsl-intel-pytorch-xpu.yml` | `intel-pytorch-xpu-dev` | `pytorch-xpu` | `8890` | `8888` | `Dockerfile.intel-pytorch-xpu` |
| WSL Intel OpenVINO | `docker-compose-wsl-intel-openvino.yml` | `intel-openvino-dev` | `openvino` | `8891` | `8888` | `Dockerfile.intel-openvino` |
| WSL NVIDIA CUDA | `docker-compose-wsl-intel-nvidia-cuda.yml` | `intel-nvidia-cuda-dev` | `wsl-nvidia-cuda` | `8892` | `8888` | `Dockerfile.intel-nvidia-cuda` |
| Linux NVIDIA CUDA | `docker-compose-linux-nvidia-cuda.yml` | `linux-nvidia-cuda-dev` | `linux-nvidia-cuda` | `8893` | `8888` | `Dockerfile.intel-nvidia-cuda` |

---

## 4. 저장소 구조

```text
deep-learning-dev-env/
├─ .devcontainer/
│  ├─ wsl-intel-tensorflow-xpu/
│  │  └─ devcontainer.json
│  ├─ wsl-intel-pytorch-xpu/
│  │  └─ devcontainer.json
│  ├─ wsl-intel-openvino/
│  │  └─ devcontainer.json
│  ├─ wsl-intel-nvidia-cuda/
│  │  └─ devcontainer.json
│  └─ linux-nvidia-cuda/
│     └─ devcontainer.json
│
├─ docker-compose-wsl-intel-tensorflow-xpu.yml
├─ docker-compose-wsl-intel-pytorch-xpu.yml
├─ docker-compose-wsl-intel-openvino.yml
├─ docker-compose-wsl-intel-nvidia-cuda.yml
├─ docker-compose-linux-nvidia-cuda.yml
│
├─ Dockerfile.intel-tensorflow-xpu
├─ Dockerfile.intel-pytorch-xpu
├─ Dockerfile.intel-openvino
├─ Dockerfile.intel-nvidia-cuda
│
├─ .env
└─ README.md
```

---

## 5. 사전 요구사항

### Windows + WSL 2 환경

- Windows 10/11
- WSL 2
- Docker Desktop
- VS Code
- VS Code Dev Containers 확장
- NVIDIA GPU 사용 시 Windows용 NVIDIA Driver
- Intel XPU 사용 시 Intel GPU 관련 런타임 확인 필요

### Linux + NVIDIA CUDA 환경

- Linux
- Docker Engine
- NVIDIA Driver
- NVIDIA Container Toolkit
- NVIDIA GPU

---

## 6. 환경 변수 설정

저장소 루트에 `.env` 파일을 생성합니다.

```env
SOURCE_PATH=/mnt/host/f/meori-hr-ai
DRIVE_PATH=/mnt/host/e/drive
PROJECT_NAME=deeplearningspace

TENSORFLOW_XPU_HOST_PORT=8889
PYTORCH_XPU_HOST_PORT=8890
OPENVINO_HOST_PORT=8891
WSL_NVIDIA_CUDA_HOST_PORT=8892
LINUX_NVIDIA_CUDA_HOST_PORT=8893
```

| 변수 | 의미 |
|---|---|
| `SOURCE_PATH` | 컨테이너 작업 폴더로 연결할 호스트 경로 |
| `DRIVE_PATH` | 읽기 전용으로 연결할 보조 드라이브 경로 |
| `PROJECT_NAME` | `/home/workspace/` 아래에 연결할 프로젝트 폴더명 |
| `*_HOST_PORT` | 호스트에서 접속할 외부 포트 |

---

## 7. 포트 설계 기준

컨테이너 내부 Jupyter 포트는 모든 환경에서 `8888`로 통일합니다.

외부 접속 포트만 compose 파일과 `.env`에서 환경별로 다르게 지정합니다.

```yaml
ports:
  - "${TENSORFLOW_XPU_HOST_PORT}:8888"
```

의미는 다음과 같습니다.

```text
호스트 외부 포트 : 컨테이너 내부 포트
```

예시:

| 환경 | 외부 접속 주소 | 컨테이너 내부 Jupyter 포트 |
|---|---|---:|
| TensorFlow XPU | `http://localhost:8889` | `8888` |
| PyTorch XPU | `http://localhost:8890` | `8888` |
| OpenVINO | `http://localhost:8891` | `8888` |
| WSL NVIDIA CUDA | `http://localhost:8892` | `8888` |
| Linux NVIDIA CUDA | `http://localhost:8893` | `8888` |

---

## 8. Dockerfile의 Jupyter 포트

Dockerfile에서는 내부 Jupyter 포트를 `8888`로 통일합니다.

```dockerfile
EXPOSE 8888

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--ServerApp.token=", \
     "--ServerApp.allow_origin=*"]
```

| 설정 | 의미 |
|---|---|
| `EXPOSE 8888` | 이미지가 내부적으로 `8888` 포트를 사용한다는 메타데이터 |
| `--port=8888` | Jupyter Lab 서버가 컨테이너 내부 `8888` 포트에서 실행됨 |
| compose `ports` 왼쪽 | 호스트 외부 접속 포트 |
| compose `ports` 오른쪽 | 컨테이너 내부 Jupyter 포트 |

---

## 9. Dev Container에서 Jupyter 실행

`.devcontainer/*/devcontainer.json`에서 Jupyter를 직접 실행한다면 포트를 `8888`로 맞춥니다.

```json
{
  "postStartCommand": [
    "sh",
    "-c",
    "ln -s /home/drive /home/workspace/drive || true && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.token= --ServerApp.allow_origin=*"
  ],
  "forwardPorts": [8888]
}
```

`overrideCommand: true`가 설정되어 있으면 Dockerfile의 `CMD`가 Dev Container 실행 과정에서 덮어써질 수 있습니다. 이 경우 Jupyter 실행은 `postStartCommand`가 담당합니다.

---

## 10. 저장소 클론

```bash
git clone git@github.com:withlionbuddha/deep-learning-dev-env.git
cd deep-learning-dev-env
```

---

## 11. VS Code Dev Container 실행

VS Code에서 저장소 루트 폴더를 엽니다.

```text
File → Open Folder → deep-learning-dev-env
```

명령 팔레트를 실행합니다.

```text
Ctrl + Shift + P
```

다음 명령을 선택합니다.

```text
Dev Containers: Reopen in Container
```

환경 선택 목록에서 필요한 항목을 하나 선택합니다.

```text
wsl-intel-tensorflow-xpu
wsl-intel-pytorch-xpu
wsl-intel-openvino
wsl-intel-nvidia-cuda
linux-nvidia-cuda
```

---

## 12. Docker Compose 직접 실행

### WSL Intel TensorFlow XPU

```bash
docker compose \
  -f docker-compose-wsl-intel-tensorflow-xpu.yml \
  --profile tensorflow-xpu \
  up -d --build
```

접속 주소:

```text
http://localhost:8889
```

### WSL Intel PyTorch XPU

```bash
docker compose \
  -f docker-compose-wsl-intel-pytorch-xpu.yml \
  --profile pytorch-xpu \
  up -d --build
```

접속 주소:

```text
http://localhost:8890
```

### WSL Intel OpenVINO

```bash
docker compose \
  -f docker-compose-wsl-intel-openvino.yml \
  --profile openvino \
  up -d --build
```

접속 주소:

```text
http://localhost:8891
```

### WSL NVIDIA CUDA

```bash
docker compose \
  -f docker-compose-wsl-intel-nvidia-cuda.yml \
  --profile wsl-nvidia-cuda \
  up -d --build
```

접속 주소:

```text
http://localhost:8892
```

### Linux NVIDIA CUDA

```bash
docker compose \
  -f docker-compose-linux-nvidia-cuda.yml \
  --profile linux-nvidia-cuda \
  up -d --build
```

접속 주소:

```text
http://localhost:8893
```

---

## 13. Python 설치 기준

이 저장소는 `python:3.10-slim-bookworm` 이미지를 기준으로 합니다.

권장 Python 경로:

```text
/usr/local/bin/python
```

Python 패키지는 다음 형식으로 설치하는 것을 권장합니다.

```bash
python -m pip install --no-cache-dir <package-name>
```

`pip install`만 사용하는 방식은 피하는 것이 좋습니다. 컨테이너에 Debian 패키지 기반 Python이 함께 설치된 경우 `pip`, `pip3`, `python`, `python3`가 서로 다른 Python 환경을 가리킬 수 있기 때문입니다.

확인 명령:

```bash
which python
python --version
python -c "import sys; print(sys.executable)"
python -m pip --version
```

---

## 14. Jupyter Kernel 선택

노트북에서 Kernel을 선택할 때는 컨테이너 내부 Python을 선택합니다.

권장 경로:

```text
/usr/local/bin/python
```

커널이 보이지 않으면 컨테이너 터미널에서 다음 명령을 실행합니다.

```bash
python -m ipykernel install --user --name python310-dev --display-name "Python 3.10 Dev"
```

---

## 15. VS Code 확장

Dev Container 내부에서 Python/Jupyter 사용을 위해 다음 확장을 권장합니다.

```json
{
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-toolsai.jupyter",
        "ms-toolsai.jupyter-keymap",
        "ms-toolsai.jupyter-renderers",
        "ms-azuretools.vscode-docker"
      ],
      "settings": {
        "python.defaultInterpreterPath": "/usr/local/bin/python"
      }
    }
  }
}
```

Kernel 선택 시 다음 메시지가 표시될 수 있습니다.

```text
Install / Enable suggested Python Jupyter extension
```

이 경우 Dev Container 내부 VS Code Server에 Python/Jupyter 확장이 없거나 비활성화된 상태입니다.

---

## 16. WSL NVIDIA GPU 설정

WSL 2 + Docker Desktop + NVIDIA GPU 환경에서는 NVIDIA CUDA 기준으로 설정합니다.

권장 compose 설정:

```yaml
environment:
  NVIDIA_VISIBLE_DEVICES: all
  NVIDIA_DRIVER_CAPABILITIES: compute,utility
  PYTHONUNBUFFERED: "1"

deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

NVIDIA CUDA 목적이면 보통 다음 설정은 필요하지 않습니다.

```yaml
devices:
  - /dev/dxg:/dev/dxg

volumes:
  - type: bind
    source: /usr/lib/wsl
    target: /usr/lib/wsl
    read_only: true

environment:
  LD_LIBRARY_PATH: "/usr/lib/wsl/lib"
```

위 설정은 Intel XPU, OpenCL, Level Zero, DirectML 계열 실험에서 필요할 수 있습니다.

---

## 17. Intel XPU 설정

Intel XPU 환경에서는 WSL GPU 장치와 라이브러리 경로가 필요할 수 있습니다.

```yaml
devices:
  - /dev/dxg:/dev/dxg

volumes:
  - type: bind
    source: /usr/lib/wsl
    target: /usr/lib/wsl
    read_only: true

environment:
  LD_LIBRARY_PATH: "/usr/lib/wsl/lib"
```

Intel oneAPI/SYCL runtime이 필요한 경우 다음 오류가 발생할 수 있습니다.

```text
NotFoundError: libsycl.so.8: cannot open shared object file
NotFoundError: libimf.so: cannot open shared object file
```

확인 명령:

```bash
find /opt/intel -name "libsycl.so*" 2>/dev/null
find /opt/intel -name "libimf.so*" 2>/dev/null
ldconfig -p | grep libsycl
ldconfig -p | grep libimf
```

---

## 18. GPU 확인 명령

### NVIDIA GPU

호스트 또는 WSL에서 확인:

```bash
nvidia-smi
```

Docker GPU 확인:

```bash
docker run --rm --gpus all nvidia/cuda:12.5.0-base-ubuntu22.04 nvidia-smi
```

컨테이너 내부 PyTorch CUDA 확인:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### Intel XPU

```bash
clinfo
```

PyTorch XPU 확인:

```bash
python -c "import torch; print(torch.__version__); print(torch.xpu.is_available())"
```

TensorFlow 확인:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

---

## 19. 포트 충돌 해결

다음 오류가 발생할 수 있습니다.

```text
Bind for 0.0.0.0:8889 failed: port is already allocated
```

실행 중인 컨테이너와 포트를 확인합니다.

```bash
docker ps --format "table {{.Names}}\t{{.Ports}}"
```

기존 컨테이너 중지:

```bash
docker stop <container-name>
```

또는 .env 파일의 host port를 변경합니다. 

---

## 20. Docker / WSL 디스크 공간 문제

다음 오류가 발생할 수 있습니다.

```text
No space left on device
OSError: [Errno 5] Input/output error
```

Docker 사용량 확인:

```bash
docker system df
```

WSL 디스크 확인:

```bash
wsl -d docker-desktop -e df -h
```

정리:

```bash
docker builder prune -a
docker image prune -a
docker container prune
```

사용하지 않는 volume까지 정리할 경우:

```bash
docker volume prune
```

중요 데이터가 Docker volume에 저장되어 있다면 삭제 전 확인이 필요합니다.

---

## 21. Docker Desktop WSL Disk 이동 시 주의

Docker Desktop의 `docker_data.vhdx` 이동 중 다음 오류가 발생할 수 있습니다.

```text
The process cannot access the file because it is being used by another process.
```

해결 순서:

```text
1. VS Code Dev Container 창 닫기
2. 실행 중인 컨테이너 중지
3. Docker Desktop 종료
4. wsl --shutdown 실행
5. Docker Desktop 재실행
6. Disk image location 이동 재시도
```

PowerShell 예시:

```powershell
docker ps -q | ForEach-Object { docker stop $_ }
wsl --shutdown
```

---


## 22. 참고

이 저장소는 운영 배포용 이미지가 아니라 학습, 실험, 개발 환경 재현을 위한 Dev Container 구성입니다.

딥러닝 프레임워크와 GPU/XPU 런타임은 버전 충돌이 자주 발생할 수 있으므로 TensorFlow, PyTorch, OpenVINO, CUDA 환경을 목적별 compose 파일과 Dev Container 설정으로 분리하는 방식을 사용합니다.
