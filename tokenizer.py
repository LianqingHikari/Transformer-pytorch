# tokenizer.py
# 专注于BPE分词器的训练、加载与验证，包含分词器核心配置
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

# -------------------------- 分词器核心配置（与论文一致） --------------------------
# 特殊标记：<pad>填充、<unk>未知词、<s>句子开始、</s>句子结束（论文标准）
SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]
# BPE词汇表大小（论文明确指定30,000）
VOCAB_SIZE = 37000
# 低频子词过滤阈值（避免词汇表膨胀，仅保留出现≥2次的子词组合）
MIN_FREQUENCY = 2


def train_bpe_tokenizer(data_files: list, save_path: str, special_tokens: list = SPECIAL_TOKENS) -> Tokenizer:
    """
    训练BPE分词器（严格复现论文方法），并保存到指定路径

    Args:
        data_files: 训练数据文件路径列表（单语数据，如["./train.en"]）
        save_path: 分词器保存路径（如"./tokenizers/src_tokenizer_en.json"）
        special_tokens: 需加入词汇表的特殊标记，默认使用论文标准标记

    Returns:
        训练完成的BPE分词器实例

    Raises:
        RuntimeError: 训练或保存过程失败时抛出
    """
    print(f"\n[分词器训练] 开始 | 词汇表大小：{VOCAB_SIZE} | 训练数据：{data_files}")

    # 1. 初始化BPE模型（未知词标记为<unk>，与论文一致）
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    # 2. 预处理配置：按空格拆分（英文/德文通用预处理）
    tokenizer.pre_tokenizer = Whitespace()

    # 3. 训练器配置（完全匹配论文参数）
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=special_tokens,
        min_frequency=MIN_FREQUENCY,
        show_progress=True  # 显示训练进度
    )

    # 4. 执行训练
    try:
        print("[分词器训练] 正在训练...")
        tokenizer.train(data_files, trainer)
    except Exception as e:
        raise RuntimeError(f"[分词器训练] 训练失败：{str(e)}") from e

    # 5. 后处理配置：自动添加<s>和</s>（符合Transformer输入格式）
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",  # 单句格式（源/目标语言单独编码）
        pair="<s> $A </s> $B:1 </s>:1",  # 句对格式（翻译任务输入-输出绑定）
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ],
    )

    # 6. 保存分词器
    try:
        # 确保保存目录存在
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        tokenizer.save(save_path)
        print(f"[分词器训练] 完成 | 保存路径：{save_path} | 词汇表大小：{tokenizer.get_vocab_size()}")
    except Exception as e:
        raise RuntimeError(f"[分词器训练] 保存失败：{str(e)}") from e

    return tokenizer


def load_bpe_tokenizer(load_path: str, special_tokens: list = SPECIAL_TOKENS) -> Tokenizer:
    """
    加载已训练的BPE分词器，并验证完整性（确保符合论文使用要求）

    Args:
        load_path: 已训练分词器的文件路径
        special_tokens: 需验证的特殊标记列表（默认论文标准标记）

    Returns:
        加载并验证通过的BPE分词器实例

    Raises:
        FileNotFoundError: 分词器文件不存在时抛出
        ValueError: 分词器缺少特殊标记或后处理配置时抛出
        RuntimeError: 加载过程失败时抛出
    """
    print(f"\n[分词器加载] 尝试加载 | 路径：{load_path}")

    # 1. 检查文件是否存在
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"[分词器加载] 未找到文件：{load_path}")

    # 2. 加载分词器
    try:
        tokenizer = Tokenizer.from_file(load_path)
    except Exception as e:
        raise RuntimeError(f"[分词器加载] 加载失败：{str(e)}") from e

    # 3. 验证分词器完整性（核心：确保符合论文格式要求）

    # 3.1 验证特殊标记是否完整（修复：跳过<unk>自身的冲突检查）
    unk_token_id = tokenizer.token_to_id("<unk>")  # 提前获取<unk>的ID，避免重复计算
    for token in special_tokens:
        token_id = tokenizer.token_to_id(token)
        # 检查标记是否存在（必须包含所有特殊标记）
        if token_id is None:
            raise ValueError(f"[分词器加载] 缺少必要特殊标记：{token}（需重新训练）")
        # 仅对非<unk>的标记，检查是否与<unk>的ID冲突（<unk>无需与自身对比）
        if token != "<unk>" and token_id == unk_token_id:
            raise ValueError(f"[分词器加载] 标记{token}与<unk>ID冲突（需重新训练）")

    # 3.2 验证后处理配置（确保自动添加<s>和</s>）
    if tokenizer.post_processor is None:
        raise ValueError(f"[分词器加载] 缺少后处理配置（无法自动添加<s>/</s>，需重新训练）")

    print(f"[分词器加载] 成功 | 词汇表大小：{tokenizer.get_vocab_size()}")
    return tokenizer