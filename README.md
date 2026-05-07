# TargetDiff 재현 코드

TargetDiff 원본 코드를 참고해 학습과 샘플링 흐름을 간단히 재작성한 코드입니다. 이 저장소의 핵심 목적은 `62000.pt` checkpoint를 사용해 test set pocket 100개(`data_id=0..99`)에 대해 ligand를 샘플링하는 것입니다.

전체 흐름은 아래와 같습니다.

```text
train_diffusion.py -> checkpoint 생성
sampling.py        -> test set pocket별 ligand 샘플링
```

## 주요 파일

- `config.yml`: 학습 설정 파일입니다.
- `sampling.yml`: 샘플링 설정 파일입니다. 기본 checkpoint는 `62000.pt`입니다.
- `train_diffusion.py`: diffusion model 학습 코드입니다.
- `sampling.py`: 특정 pocket에 대해 ligand를 샘플링하는 코드입니다.
- `dataset.py`, `diffusion.py`, `network.py`, `reconstruct.py`: 데이터 처리, 모델, diffusion, 분자 재구성 핵심 코드입니다.
- `run_sample_test100.sh`: `data_id=0..99` 전체에 대해 샘플링을 실행합니다.
- `run_full_train_and_sample.sh`: 학습을 수행한 뒤 최신 checkpoint로 test100 샘플링을 실행합니다.
- `build_lmdb.py`: pocket/ligand 파일에서 LMDB를 만드는 선택용 유틸리티입니다.
- `evaluate_diffusion.py`: 보조 평가 스크립트입니다. 원본 TargetDiff의 `utils/` 계열 코드와 docking 도구가 필요하므로, 최소 샘플링 재현 경로에는 포함되지 않습니다.

## 필요한 파일

데이터 파일은 용량이 커서 Git에 포함하지 않습니다. 실행 전에 아래 경로에 직접 준비해야 합니다.

```text
data/crossdocked_pocket10_pose_split.pt
data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb
```

샘플링에 사용하는 checkpoint는 저장소에 포함합니다.

```text
logs_diffusion_full/targetdiff_cjkim_full_gpu/checkpoints/62000.pt
```

이 경로들은 `sampling.yml`과 `run_sample_test100.sh`의 기본값과 일치합니다.

## 환경

아래 패키지들이 필요합니다.

```text
torch
torch_geometric
torch_scatter
rdkit
openbabel
lmdb
scipy
numpy
pyyaml
tqdm
tensorboard
```

특히 `torch`, `torch_geometric`, `torch_scatter`는 서로 호환되는 버전으로 설치해야 합니다.

## Test100 샘플링

기본 실행:

```bash
bash run_sample_test100.sh
```

샘플 수, batch size, diffusion step 수를 바꾸려면 환경변수를 사용합니다.

```bash
NUM_SAMPLES=100 BATCH_SIZE=16 NUM_STEPS=1000 bash run_sample_test100.sh
```

결과는 아래 폴더에 저장됩니다.

```text
sampling_results_full_test100/
```

## 학습부터 샘플링까지 실행

학습을 다시 수행한 뒤, 생성된 최신 checkpoint로 test100 샘플링까지 실행하려면 아래 스크립트를 사용합니다.

```bash
bash run_full_train_and_sample.sh
```

예시:

```bash
TRAIN_MAX_ITERS=71000 TRAIN_TAG=cjkim_full_gpu bash run_full_train_and_sample.sh
```

## Git에 포함하지 않는 파일

아래 파일과 폴더는 로컬 데이터 또는 실행 산출물이므로 `.gitignore`에 포함되어 있습니다.

```text
data/
logs_diffusion*/              # 단, 62000.pt는 예외로 포함
sampling_results*/
targetdiff_eval_meta_full_test100/
sampling_runtime*.yml
*.lmdb
```
