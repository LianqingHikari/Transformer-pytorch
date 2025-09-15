import torch


# -------------------------- 数据与清洗超参数 --------------------------
MAX_SEQ_LENGTH = 512
MIN_SENTENCE_LENGTH = 1
MAX_LENGTH_RATIO = 2.0
VALIDATION_SPLIT_RATIO = 0.05


# -------------------------- 分词器超参数 --------------------------
SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]
VOCAB_SIZE = 37000
MIN_FREQUENCY = 2


# -------------------------- 训练超参数（模块级常量） --------------------------
BATCH_SIZE = 16
EPOCHS = 30
GRADIENT_ACCUMULATION = 8
RESUME_CHECKPOINT = None

# 模型结构
D_MODEL = 512
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
D_FF = 2048
DROPOUT = 0.1

# 优化器
LEARNING_RATE = 5e-4
BETA1 = 0.9
BETA2 = 0.98
EPS = 1e-9

# 训练杂项
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOG_DIR = 'logs'
CHECKPOINT_DIR = 'checkpoints'
SAVE_BEST_ONLY = False
CLIP_GRAD_NORM = 5.0
LABEL_SMOOTHING = 0.1


