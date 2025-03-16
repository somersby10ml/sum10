import base64
import io
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from apple_ocr.models.train_model import DigitCNN
from util.img import Ocr


def init_apple_ocr():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitCNN().to(device)
    model.load_state_dict(torch.load('./assets/digit_cnn.pth', map_location=device))
    model.eval()