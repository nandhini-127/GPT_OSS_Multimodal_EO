import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 224
NUM_CLASSES = 10   # EuroSAT has 10 classes

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 3e-4

MODEL_NAME = "swin_tiny_patch4_window7_224"

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"
