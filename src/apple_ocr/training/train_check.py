import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from ocr.train_model import DigitCNN


def load_model(model_path='digit_cnn.pth'):
    model = DigitCNN()
    model.load_state_dict(torch.load(
        model_path, map_location=torch.device('cpu')))
    model.eval()  # 평가 모드로 변경
    return model


def test_model(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    print(f"테스트 정확도: {accuracy:.2f}%")


def visualize_predictions(model, dataset, num_samples=5):
    fig, axes = plt.subplots(1, num_samples, figsize=(3*num_samples, 3))
    for ax in axes:
        idx = np.random.randint(0, len(dataset))
        image, label = dataset[idx]
        with torch.no_grad():
            output = model(image.unsqueeze(0))
        pred = output.argmax(dim=1).item()
        # 라벨과 예측 결과에 +1 적용
        ax.imshow(image.squeeze(), cmap='gray')
        ax.set_title(f"Label: {label+1}\nPred: {pred+1}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 학습할 때 사용한 전처리와 동일한 전처리 적용
    transform = transforms.Compose([
        transforms.Grayscale(),  # 컬러인 경우 그레이스케일로 변환
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 테스트 데이터셋 (data/test 폴더가 없는 경우 data/train을 사용해도 됩니다.)
    test_dataset = ImageFolder(root='./data/train', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 모델 로드 및 평가
    model = load_model()
    test_model(model, test_loader)
    visualize_predictions(model, test_dataset)
