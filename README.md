![image](https://github.com/user-attachments/assets/70868c54-1dbe-49d8-8879-081562811273)


![image](https://github.com/user-attachments/assets/548b7821-5a04-4daf-8f45-f857bd8244cb)
이 코드는 **Generative Adversarial Network (GAN)**을 구현한 예제입니다. GAN은 **Generator**와 **Discriminator**라는 두 신경망으로 구성되며, 서로 경쟁하면서 훈련됩니다. Generator는 가짜 이미지를 생성하고, Discriminator는 그 이미지가 실제 이미지인지 가짜 이미지인지를 판별합니다. 이 네트워크는 결국 Generator가 매우 사실적인 이미지를 생성할 수 있도록 학습합니다.

### 1. **라이브러리 임포트**
```python
import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
```
- **os**: 파일 경로 관리.
- **torch, torch.nn, torch.optim**: PyTorch 라이브러리로 모델, 손실 함수, 최적화 알고리즘을 정의.
- **matplotlib.pyplot**: 이미지 시각화를 위한 라이브러리.
- **torch.utils.data**: 사용자 정의 데이터셋을 처리할 때 사용.
- **PIL (Python Imaging Library)**: 이미지를 로딩하고 처리하기 위한 라이브러리.
- **tqdm**: 진행 상태 표시를 위한 라이브러리.

### 2. **커스텀 데이터셋 클래스**
```python
class CustomImageDataset(Dataset):
    def __init__(self, image_folder_path, transform=None):
        self.image_paths = [os.path.join(image_folder_path, fname) for fname in os.listdir(image_folder_path)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # 이미지를 RGB로 변환
        
        if self.transform:
            image = self.transform(image)
        
        return image
```
- **CustomImageDataset**: 이미지 폴더에서 이미지를 로딩하고, 변환(transformation)을 적용하는 데이터셋 클래스.
- `__init__`: 이미지 폴더의 경로를 받아 이미지 파일 경로 리스트를 생성.
- `__len__`: 데이터셋의 길이 반환.
- `__getitem__`: 주어진 인덱스에 해당하는 이미지를 로딩하고, 필요하다면 변환을 적용한 후 반환.

### 3. **모델 정의**
#### Generator
```python
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(128 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 128 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = x.view(x.size(0), 128, 7, 7)
        x = torch.relu(self.deconv1(x))
        x = torch.tanh(self.deconv2(x))
        return x
```
- **Generator**: 1채널 (흑백 이미지)을 입력받아 1채널 이미지를 생성하는 네트워크.
    - **Conv2d**: 컨볼루션 층 (이미지에서 특징을 추출).
    - **Linear**: 완전 연결층 (특징을 압축).
    - **ConvTranspose2d**: 전치 컨볼루션 층 (이미지 크기 확장).
    - **ReLU**: 비선형 활성화 함수.
    - **Tanh**: 출력값을 -1과 1 사이로 압축하는 활성화 함수 (생성된 이미지 픽셀 값을 이 범위로 제한).

#### Discriminator
```python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(128 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
```
- **Discriminator**: 이미지를 입력받아 그것이 실제 이미지인지 생성된 가짜 이미지인지를 판별하는 네트워크.
    - **Conv2d**: 입력 이미지에서 특징을 추출.
    - **Linear**: 특징을 압축.
    - **Sigmoid**: 출력값을 0과 1 사이로 변환 (실제 이미지일 확률을 출력).

### 4. **손실 함수 및 최적화**
```python
def adversarial_loss(disc_pred, target):
    return torch.mean((disc_pred - target) ** 2)
```
- **Adversarial Loss**: 실제 이미지에 대한 Discriminator의 예측을 1로, 가짜 이미지에 대한 예측을 0으로 맞추는 손실 함수.
    - 실제 이미지는 `1`, 가짜 이미지는 `0`을 목표로 하는 MSE (Mean Squared Error) 기반 손실 함수 사용.

### 5. **가짜 이미지 저장 함수**
```python
def save_fake_images(epoch, fake_images):
    fake_images = fake_images.detach().cpu()
    grid = torchvision.utils.make_grid(fake_images, normalize=True)
    plt.figure(figsize=(8,8))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(f"Epoch {epoch}")
    plt.axis('off')
    plt.savefig(f'fake_images_epoch_{epoch}.png')
    plt.close()
```
- **save_fake_images**: 훈련 중 생성된 가짜 이미지를 저장하는 함수.

### 6. **데이터셋 및 DataLoader 준비**
```python
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```
- 이미지 변환 함수 정의: 이미지를 1채널의 흑백 이미지로 변환하고, 28x28 크기로 리사이즈 후, 텐서로 변환하고, 정규화.

```python
image_folder_path = '/content/drive/MyDrive/GANFACE/filtered_images'  # 이미지 폴더 경로
train_dataset = CustomImageDataset(image_folder_path, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
```
- `image_folder_path`는 학습 데이터가 들어있는 폴더 경로를 설정하고, 이를 `CustomImageDataset`으로 불러옵니다. 그리고 DataLoader를 사용하여 배치 단위로 데이터를 로딩합니다.

### 7. **모델 초기화 및 최적화기 설정**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator().to(device)
discriminator = Discriminator().to(device)

optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
```
- `device`는 GPU가 있으면 GPU를 사용하도록 설정.
- Generator와 Discriminator 모델을 초기화하고, Adam 옵티마이저를 설정.

### 8. **훈련 함수**
```python
def train(generator, discriminator, dataloader, num_epochs, device):
    for epoch in range(num_epochs):
        for i, real_images in enumerate(tqdm(dataloader)):
            # 훈련 과정
```
- 각 epoch마다 GAN을 훈련시키고, Generator와 Discriminator를 번갈아 업데이트합니다.
- `real_images`: 실제 이미지를 Discriminator에 입력하고, 그에 대한 손실을 계산하여 Discriminator를 업데이트합니다.
- `fake_images`: Generator를 통해 가짜 이미지를 생성하고, 그에 대한 손실을 계산하여 Generator를 업데이트합니다.
- 훈련 중 주기적으로 손실을 출력하고, 일정 epoch마다 가짜 이미지를 저장합니다.

### 9. **모델 저장**
```python
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')
```
- 훈련이 끝난 후, 모델의 가중치를 파일로 저장합니다.

### 요약:
이 코드는 GAN을 사용하여 이미지를 생성하는 모델을 훈련하는 코드입니다. Generator는 가짜 이미지를 생성하고, Discriminator는 그것이 실제 이미지인지 가짜 이미지인지를 판별합니다. 두 네트워크는 서로 경쟁하면서 점점 더 사실적인 이미지를 생성하게 됩니다.
