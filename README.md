# Deep Learning Dev Environment
[![Docker](https://img.shields.io/badge/Docker-Dev%20Container-blue)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.10-yellow)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook%20%7C%20Lab-orange)](https://jupyter.org/)
[![WSL2](https://img.shields.io/badge/WSL-2-green)](https://learn.microsoft.com/windows/wsl/)
[![Intel](https://img.shields.io/badge/Intel-Iris%20Xe%20XPU-0071C5)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0%2Bxpu-EE4C2C)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-FF6F00)](https://www.tensorflow.org/)
[![oneAPI](https://img.shields.io/badge/oneAPI-2025.0-0071C5)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)

> Windows + WSL2 + Docker Desktop 및 Linux 환경에서 **딥러닝·AI 개발과 하드웨어 가속 실험을 위한 컨테이너 기반 개발 환경**을 구성하는 저장소입니다.
> Intel Iris Xe XPU와 NVIDIA CUDA GPU를 대상으로 PyTorch, TensorFlow, OpenVINO 등의 프레임워크를 실행할 수 있도록 **Dockerfile, Docker Compose, VS Code Dev Container 및 Jupyter 개발 환경**을 목적별로 구성하고 관리합니다.특히 Intel Iris Xe 환경에서는 Level Zero, OpenCL, oneAPI/SYCL 기반 XPU 가속 환경을 구성하고, 프레임워크별 하드웨어 인식과 연산 동작을 검증합니다.
>
> 이전 README는 [`README-v0.1.0.md`](README-v0.1.0.md)에 보관합니다.

---

## PyTorch & TensorFlow Docker Dev Environment for Intel XPU(GPU)

>
> [`README-v0.1.0.md`](README-v0.1.0.md) 에 비해 수정된 내용은 TensorFlow 연산을 수행하기 위한 컴파일러·연산·런타임·드라이버 계층 업데이트 이며, Pytorch 연산을 수행을 위한 라이브러리는 이전과 동일합니다.
>
> 아래에 구성, 라이브러리 사양 및 실제 검증 결과를 상세히 기록합니다.
> ## TensorFlow 라이브러리 구성과 역할 
>
> ```text
> TensorFlow
>     ↓
> Intel Extension for TensorFlow (ITEX)
>     ↓
> SYCL / oneAPI
>     ├─ libsycl.so.8              (XPU 연산 실행 Runtime)
>     ├─ oneMKL
>     │   └─ libmkl_sycl_blas.so.5 (XPU 수치 연산)
>     └─ UMF
>         └─ libumf.so.0           (CPU/XPU 메모리 관리)
>     ↓
> GPU Runtime
>     ├─ Level Zero                (Intel GPU 저수준 제어 및 연산 실행)
>     └─ OpenCL (NEO)              (Intel GPU OpenCL 연산 실행)
>     ↓
> Intel Iris Xe GPU                (실제 연산을 수행하는 GPU 하드웨어)
> ```
>
> ## 1. 검증 하드웨어
>
>> ### Intel Iris Xe 개발 환경
>>
>> | 구분 | 사양 |
>> |---|---|
>> | CPU | Intel Core i7-1260P |
>> | RAM | 16 GB |
>> | GPU | Intel Iris Xe / `Intel(R) Graphics [0x46a6]` |
>> | Host | Windows + WSL2 + Docker Desktop |
>> | Container OS | Ubuntu 22.04 |
>> | Architecture | x86_64 |
>> | Python | 3.10 |
>> | GPU device bridge | `/dev/dxg` |
>> | WSL GPU libraries | `/usr/lib/wsl` |
>>
>> Intel Iris Xe `[0x46a6]`는 컨테이너에서 OpenCL과 Level Zero 양쪽 backend로 확인되었습니다.
>>
>> ```text
>> [level_zero:gpu][level_zero:0] Intel(R) oneAPI Unified Runtime over Level-Zero,
>> Intel(R) Graphics [0x46a6] 12.3.0 [1.3.29735+27]
>>
>> [opencl:gpu][opencl:0] Intel(R) OpenCL Graphics,
>> Intel(R) Graphics [0x46a6] OpenCL 3.0 NEO [24.39.31294]
>> ```
>
> ---
>
> ## 2. PyTorch XPU
>
>> ### 2.1 Framework 사양
>>
>> | 구성 | 버전 / 설정 |
>> |---|---|
>> | PyTorch | `2.8.0+xpu` |
>> | TorchVision | `0.23.0+xpu` |
>> | TorchAudio | `2.8.0+xpu` |
>> | Package index | PyTorch XPU wheel index |
>> | GPU | Intel Iris Xe `[0x46a6]` |
>> | Device | `xpu:0` |
>>
>> ### 2.2 검증 상태
>>
>> PyTorch XPU 환경에서 Intel GPU device 사용 및 행렬곱 연산이 확인되었습니다.
>>
>> ```text
>> Device : Intel(R) Graphics [0x46a6]
>> XPU    : xpu:0
>> ```
>>
>> 1000 × 1000 행렬곱 테스트:
>>
>> ```text
>> 0.00557 s
>> 0.01228 s
>> 0.00539 s
>> average ≈ 0.00775 s
>> ```
>
> ---
>
> ## 3. TensorFlow XPU
>
>> ### 3.1 Framework 사양
>>
>> | 구성 | 버전 / 설정 |
>> |---|---|
>> | TensorFlow | `2.15.0` |
>> | Intel Extension for TensorFlow | `2.15.0.2` |
>> | ITEX library | `2.15.0.2.2` |
>> | oneAPI DPC++ | `2025.0` (`2025.0.4-1519` 확인) |
>> | oneMKL | `2025.0` (`2025.0.1-14` 확인) |
>> | SYCL ABI | `libsycl.so.8` |
>> | oneMKL SYCL BLAS | `libmkl_sycl_blas.so.5` |
>> | Unified Memory Framework | `intel-oneapi-umf-0.9 0.9.1-6` |
>> | UMF runtime | `/opt/intel/oneapi/umf/0.9/lib/libumf.so.0` |
>> | Level Zero | `1.3.29735+27` |
>> | OpenCL NEO | `24.39.31294` |
>> | GPU | Intel Iris Xe `[0x46a6]` |
>> | TensorFlow device | `/physical_device:XPU:0` |
>>
>> ### 3.2 Runtime library path
>>
>> TensorFlow/ITEX XPU runtime에서 사용하는 주요 library path입니다.
>>
>> ```text
>> /opt/intel/oneapi/compiler/2025.0/lib
>> /opt/intel/oneapi/mkl/2025.0/lib
>> /opt/intel/oneapi/mkl/2025.0/lib/intel64
>> /opt/intel/oneapi/umf/0.9/lib
>> /usr/lib/wsl/lib
>> ```
>>
>> UMF 경로는 SYCL Unified Runtime adapter가 `libumf.so.0`를 로드하기 위해 필요합니다.
>>
>> ### 3.3 SYCL 검증
>>
>> ```bash
>> sycl-ls
>> ```
>>
>> ```text
>> [level_zero:gpu][level_zero:0] Intel(R) oneAPI Unified Runtime over Level-Zero,
>> Intel(R) Graphics [0x46a6] 12.3.0 [1.3.29735+27]
>>
>> [opencl:gpu][opencl:0] Intel(R) OpenCL Graphics,
>> Intel(R) Graphics [0x46a6] OpenCL 3.0 NEO [24.39.31294]
>> ```
>>
>> Level Zero backend:
>>
>> ```bash
>> ONEAPI_DEVICE_SELECTOR=level_zero:* sycl-ls
>> ```
>>
>> ```text
>> [level_zero:gpu] Intel(R) oneAPI Unified Runtime over Level-Zero,
>> Intel(R) Graphics [0x46a6] 12.3.0 [1.3.29735+27]
>> ```
>>
>> ### 3.4 TensorFlow / ITEX 검증
>>
>> ```python
>> import tensorflow as tf
>> import intel_extension_for_tensorflow as itex
>>
>> print("TensorFlow:", tf.__version__)
>> print("Devices:", tf.config.list_physical_devices())
>> print("XPU:", tf.config.list_physical_devices("XPU"))
>> ```
>>
>> 검증 결과:
>>
>> ```text
>> Intel Extension for Tensorflow* GPU backend is loaded.
>> Selected platform: Intel(R) oneAPI Unified Runtime over Level-Zero
>> StreamExecutor device (0): Intel(R) Graphics [0x46a6]
>>
>> TensorFlow: 2.15.0
>> Devices: [
>>   PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'),
>>   PhysicalDevice(name='/physical_device:XPU:0', device_type='XPU')
>> ]
>> XPU: [PhysicalDevice(name='/physical_device:XPU:0', device_type='XPU')]
>> ```
>>
>> TensorFlow/ITEX에서는 Intel Iris Xe가 `GPU`가 아니라 **`XPU` device type**으로 노출됩니다.
>
> ---
>
> ## 4. TensorFlow XPU 문제 해결 기록
>
>> TensorFlow 2.15 + ITEX 환경을 Intel Iris Xe에서 동작시키기 위해 SYCL/Unified Runtime dependency를 맞췄습니다.
>>
>> ```text
>> ITEX
>>  └─ SYCL / Unified Runtime
>>      ├─ libsycl.so.8
>>      ├─ libmkl_sycl_blas.so.5
>>      └─ libumf.so.0
>>          ├─ Level Zero → Intel Iris Xe
>>          └─ OpenCL    → Intel Iris Xe
>> ```
>>
>> 주요 해결 사항:
>>
>> 1. ITEX가 요구하는 SYCL 8 ABI에 맞춰 oneAPI DPC++ 2025.0 사용
>> 2. oneMKL 2025.0의 `libmkl_sycl_blas.so.5` runtime 경로 등록
>> 3. Unified Runtime adapter가 요구하는 `libumf.so.0` 경로 등록
>> 4. Compose에서 image의 `LD_LIBRARY_PATH`를 `/usr/lib/wsl/lib` 하나로 덮어쓰지 않도록 수정
>> 5. `/dev/dxg`와 `/usr/lib/wsl`을 WSL2 container에 전달
>> 6. `sycl-ls`에서 Level Zero/OpenCL Intel GPU enumeration 확인
>> 7. TensorFlow/ITEX에서 `/physical_device:XPU:0` 확인
>
> ---
>
> ## 5. TensorFlow 최종 확인 명령
>
>> ```bash
>> ldconfig -p | grep libumf
>> sycl-ls
>>
>> python -c "import tensorflow as tf; import intel_extension_for_tensorflow as itex; print('TensorFlow:', tf.__version__); print('Devices:', tf.config.list_physical_devices()); print('XPU:', tf.config.list_physical_devices('XPU'))"
>> ```
>>
>> ```text
>> libumf.so.0                           PASS
>> SYCL Level Zero Intel Iris Xe         PASS
>> SYCL OpenCL Intel Iris Xe             PASS
>> ITEX GPU backend loaded               PASS
>> TensorFlow /physical_device:XPU:0     PASS
>> ```
>
> ---
>
> ## 6. 실행 구성 및 주요 파일
>
>> ### 6.1 Framework
>>
>> `docker-compose-wsl-intel-irisxe-framework.yml`은 하나의 Intel Iris Xe 환경에서 **PyTorch 또는 TensorFlow를 선택하여 실행**합니다.
>>
>> 선택값은 `.env`의 `FRAMEWORK`이며 Compose가 이를 Docker build argument로 전달합니다.
>>
>> **PyTorch**
>>
>> ```env
>> FRAMEWORK=pytorch
>> ```
>>
>> **TensorFlow **
>>
>> ```env
>> FRAMEWORK=tensorflow
>> ```
>>
>> `FRAMEWORK`를 지정하지 않으면 기본값은 `pytorch`입니다.
>>
>> ### 6.2 Compose → Dockerfile 전달 구조
>>
>> ```text
>> .env
>>  └─ FRAMEWORK=pytorch | tensorflow
>>       ↓
>> docker-compose-wsl-intel-irisxe-framework.yml
>>  └─ build.args.FRAMEWORK
>>       ↓
>> Dockerfile.intel-irisxe-framework
>>  └─ FROM ${FRAMEWORK} AS runtime
>>       ├─ pytorch
>>       └─ tensorflow
>> ```
>>
>> Compose 설정:
>>
>> ```yaml
>> build:
>>   context: .
>>   dockerfile: Dockerfile.intel-irisxe-framework
>>   args:
>>     FRAMEWORK: ${FRAMEWORK:-pytorch}
>> ```
>>
>> Dockerfile 설정:
>>
>> ```dockerfile
>> ARG FRAMEWORK=pytorch
>>
>> # ... common Intel Iris Xe stage ...
>> # ... pytorch stage ...
>> # ... tensorflow stage ...
>>
>> FROM ${FRAMEWORK} AS runtime
>> ```
>>
>> ### 6.3 실행
>>
>> ```bash
>> # .env의 FRAMEWORK 값에 따라 PyTorch 또는 TensorFlow image를 build
>> docker compose \
>>   --env-file .env \
>>   -f docker-compose-wsl-intel-irisxe-framework.yml \
>>   build
>>
>> docker compose \
>>   --env-file .env \
>>   -f docker-compose-wsl-intel-irisxe-framework.yml \
>>   up -d
>> ```
>>
>> ### 6.4 추가/수정된 파일
>>
>> ```text
>> Dockerfile.intel-irisxe-framework
>> docker-compose-wsl-intel-irisxe-framework.yml
>> Dockerfile.intel-opencl-levelzero-xpu-tensorflow
>> Dockerfile.intel-opencl-levelzero-xpu-tensorflow-jupyter
>> docker-compose-wsl-intel-xpu-tensorflow.yml
>> docker-compose-wsl-intel-xpu-tensorflow-jupyter.yml
>> .env.example
>> ```
>>
>> `Dockerfile.intel-irisxe-framework`는 PyTorch/TensorFlow 공통 Intel Iris Xe 개발 환경입니다.
>>
>> `docker-compose-wsl-intel-irisxe-framework.yml` 실행 시 `.env`의 `FRAMEWORK=pytorch|tensorflow` 설정에 따라 사용할 framework가 결정됩니다.
