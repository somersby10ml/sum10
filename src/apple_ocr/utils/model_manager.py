import torch

from apple_ocr.models.train_model import DigitCNN


class OCRModelManager:
    _instance = None
    _model = None
    _device = None
    
    @classmethod
    def get_model(cls):
        """모델의 싱글톤 인스턴스를 반환합니다. 첫 호출 시에만 모델을 로딩합니다."""
        if cls._model is None:
            cls._load_model()
        return cls._model
    
    @classmethod
    def get_device(cls):
        """현재 사용 중인 디바이스를 반환합니다."""
        if cls._device is None:
            cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls._device
    
    @classmethod
    def _load_model(cls):
        """모델을 로딩하는 내부 메서드"""
        cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Apple OCR model on {cls._device}...")
        
        # 모델 초기화 및 가중치 로딩
        cls._model = DigitCNN().to(cls._device)
        cls._model.load_state_dict(torch.load('./assets/digit_cnn.pth', map_location=cls._device))
        cls._model.eval()
        
        print("Model loaded successfully")
    
    @classmethod
    def is_model_loaded(cls):
        """모델이 이미 로딩되었는지 확인합니다."""
        return cls._model is not None