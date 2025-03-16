import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


# CNN 모델 정의
class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 입력 채널 1, 출력 32
            nn.ReLU(),
            nn.MaxPool2d(2),                           # 14x12 -> 7x6
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 7x6 -> 7x6
            nn.ReLU(),
            nn.MaxPool2d(2)                            # 7x6 -> 3x3 (정수 나눗셈)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 9)   # 1~9까지 총 9개 클래스
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x


def train_model():
    # 전처리: 이미지가 이미 전처리되어 있으나 혹시 모르니 Grayscale 적용
    transform = transforms.Compose([
        transforms.Grayscale(),  # 만약 이미 그레이스케일이면 중복되지만 안전함
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # ImageFolder를 이용해 데이터셋 구성 (디렉토리 구조에 맞게)
    train_dataset = ImageFolder(root='./data/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = DigitCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10  # 에포크 수는 데이터에 따라 조정
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # 학습된 모델 저장
    torch.save(model.state_dict(), 'digit_cnn.pth')
    print("학습 완료 및 모델 저장됨.")


if __name__ == "__main__":
    train_model()
