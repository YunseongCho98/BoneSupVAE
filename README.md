# ConvVAE-based Bone Suppression for Chest X-rays

## 프로젝트 개요

뼈 구조가 제거된 체스트 X-레이 이미지를 생성하는 ConvVAE(Convolutional Variational Autoencoder) 기반의 딥러닝 모델입니다. 원본(X-ray) 이미지에서 뼈 성분을 제거하여 진단 시 폐 영역 시각화를 돕고, 후속 AI 분석의 정확도를 높이기 위한 전처리 도구로 활용됩니다.

## 데이터셋
	• 총 241쌍(pairs)의 흉부 X-레이 이미지로 구성
	• 각 쌍(pairs)은 원본(Original) 이미지 1장과 뼈 제거(BoneSuppression)된 타겟 이미지 1장으로 이루어짐
	• 해상도: 1024 × 1024 픽셀

```
• 디렉토리 구조

BS_dataset_split/
├─ train/
│   ├─ Original/
│   └─ BoneSuppression/ 
└─ val/
    ├─ Original/
    └─ BoneSuppression/
```

	• ChestXrayPairDataset 클래스를 통해 폴더별 파일명을 매칭하여 (input, target, filename) 형태로 로드

## 모델 구조 (ConvVAE)
### Encoder
	• 5개의 Convolutional 블록 (stride=2, padding=1)
	• 각 블록: Conv2d → ReLU
	• 최종 feature map을 flatten 후, 두 개의 선형 레이어로 잠재 벡터 평균(mu)과 로그 분산(logvar) 추출
	• Reparameterization
	• z = mu + eps * exp(0.5 * logvar) 방식으로 잠재 변수 샘플링
    
### Decoder
	• 선형층으로 잠재 벡터 복원 후 5단계 ConvTranspose 연산
	• 인코더의 스킵 연결(skip connections) 활용 (U-Net 스타일)
	• 마지막 출력 채널 Sigmoid 활성화로 [0,1] 범위로 정규화
	• 하이퍼파라미터
	• z_dim = 256 (잠재 차원)
	• base_channels = 64 (기본 채널 수)
	• 입력/출력 채널: 1 (흑백 이미지)

## 학습 과정
### 전처리 & DataLoader

    • transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])

    • train_ds = ChestXrayPairDataset("./BS_dataset_split/train", img_size=512, transform=transform)

    • train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)

### 장치 설정

    MPS(macOS), CUDA, CPU 순으로 자동 감지

### 손실 함수

    • Reconstruction Loss (L1 또는 L2)

	• KL Divergence (잠재 분포의 정규화)

	• 가중치 beta 적용 가능 (beta-VAE)

### loss = loss_fn(x_recon, y, mu, logvar, beta=1.0)

### 최적화

    • 옵티마이저: Adam(model.parameters(), lr=1e-4)
	• Epochs: 사용자 설정 (예: 50)
	• Batch size: 4

### 체크포인트

    • 매 epoch 종료 후 평균 손실이 갱신되면 모델을 저장

### 훈련 실행

    python VAE.py


## 학습
python VAE.py

## 추론 (테스트/샘플 생성용 스크립트 추가 예정)
python test.py

## 샘플

### 원본 이미지
![image](./sample_ori.png)

### 정답 이미지
![image](./sample_gt.png)

### 생성된 이미지
![image](./sample_recon.png)