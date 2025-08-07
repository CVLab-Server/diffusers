# SDXL Docker 실행 예제

Stable Diffusion XL (SDXL)을 Docker 환경에서 실행하기 위한 프로젝트입니다.
SDXL은 텍스트 프롬프트를 입력받아 고품질의 이미지를 생성하는 최신 AI 모델로, 1024x1024 해상도의 이미지를 생성할 수 있습니다.

## 프로젝트 구조

- `Dockerfile`: NVIDIA PyTorch 공식 이미지 기반 Docker 파일
- `Dockerfile.ubuntu`: Ubuntu 22.04 기반 Docker 파일
- `example.py`: SDXL text-to-image 생성 예제
- `_README.md`: 원본 Diffusers README 파일

## Docker 이미지 빌드

### 방법 1: NVIDIA PyTorch 이미지 사용 (권장)
```bash
docker build -t diffusers-sdxl:pytorch -f Dockerfile .
```

### 방법 2: Ubuntu 이미지 사용 (예비 및 트러블슈팅용으로 제공)
```bash
docker build -t diffusers-sdxl:ubuntu -f Dockerfile.ubuntu .
```

## 예제 실행 방법

### 1. 캡션 파일 준비

프로젝트 루트에 `captions.txt` 파일이 포함되어 있습니다. 이 파일에는 5개의 예제 캡션이 들어있습니다:
- Astronaut in a jungle, cold color palette, muted colors, detailed
- A majestic lion jumping from a big stone at night
- Van Gogh painting of a starry night over mountains
- A futuristic city with flying cars at sunset
- A serene Japanese garden with cherry blossoms

원하는 경우 이 파일을 수정하거나 새로운 캡션 파일을 생성하여 사용할 수 있습니다.

### 2. Docker 컨테이너 실행 및 이미지 생성

#### NVIDIA PyTorch 이미지:
```bash
docker run --gpus all -v $(pwd)/output:/output diffusers-sdxl:pytorch \
    python3 example.py --output /output
```

#### Ubuntu 이미지:
```bash
docker run --gpus all -v $(pwd)/output:/output diffusers-sdxl:ubuntu \
    python3 example.py --output /output
```

## 예제 코드 설명

`example.py`는 다음 작업을 수행합니다:
1. SDXL 모델 로드 (빌드 시 다운로드된 캐시 사용)
2. captions.txt 파일에서 텍스트 프롬프트 읽기
3. 각 프롬프트에 대해 1024x1024 이미지 생성
4. 생성된 이미지를 output 디렉토리에 저장

## 출력 예시 (stdout)
```
Loading SDXL model...
Loading pipeline components...: 100%|██████████| 7/7 [00:00<00:00,  7.22it/s]
Generating image 1/5: Astronaut in a jungle, cold color palette, muted colors, detailed
100%|██████████| 50/50 [00:12<00:00,  3.97it/s]
Saved: /output/Astronaut_in_a_jungle,_cold_color_palette,_muted_colors,_detailed.png
Generating image 2/5: A majestic lion jumping from a big stone at night
100%|██████████| 50/50 [00:12<00:00,  4.08it/s]
Saved: /output/A_majestic_lion_jumping_from_a_big_stone_at_night.png
Generating image 3/5: Van Gogh painting of a starry night over mountains
100%|██████████| 50/50 [00:12<00:00,  4.05it/s]
Saved: /output/Van_Gogh_painting_of_a_starry_night_over_mountains.png
Generating image 4/5: A futuristic city with flying cars at sunset
100%|██████████| 50/50 [00:12<00:00,  3.97it/s]
Saved: /output/A_futuristic_city_with_flying_cars_at_sunset.png
Generating image 5/5: A serene Japanese garden with cherry blossoms
100%|██████████| 50/50 [00:13<00:00,  3.80it/s]
Saved: /output/A_serene_Japanese_garden_with_cherry_blossoms.png
All 5 images generated successfully!
```

## 출력 결과

생성된 이미지는 `output` 디렉토리에 저장됩니다. 파일명은 캡션의 공백을 언더스코어(_)로 치환한 형태로 저장됩니다.

예시:
- `Astronaut in a jungle` → `Astronaut_in_a_jungle.png`
- `A majestic lion jumping` → `A_majestic_lion_jumping.png`
