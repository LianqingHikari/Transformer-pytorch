# data_processing.py
# 动态编码模式：取数时实时对原始句子编码，不提前缓存
import os
import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
# 从tokenizer.py导入分词器相关配置与函数
from tokenizer import SPECIAL_TOKENS, train_bpe_tokenizer, load_bpe_tokenizer
import hashlib
import json

# -------------------------- 数据集核心配置（与论文一致） --------------------------
MAX_SEQ_LENGTH = 512  # 最大序列长度（预留2个位置给<s>/</s>）
MIN_SENTENCE_LENGTH = 1  # 最小句子长度（仅过滤空句）
MAX_LENGTH_RATIO = 2.0  # 句对长度比上限（过滤不合理翻译）
VALIDATION_SPLIT_RATIO = 0.05  # 训练集拆分验证集比例


class WMTTranslationDataset(Dataset):
    """适配WMT数据集格式的翻译数据集类（动态编码：取数时实时分词）"""

    def __init__(self, src_sentences: list, tgt_sentences: list,
                 src_tokenizer, tgt_tokenizer):
        """
        Args:
            src_sentences: 源语言原始句子列表（清洗后）
            tgt_sentences: 目标语言原始句子列表（清洗后，与源语言严格对齐）
            src_tokenizer: 源语言BPE分词器（来自tokenizer.py）
            tgt_tokenizer: 目标语言BPE分词器（来自tokenizer.py）
        """
        # 验证句对数量一致性（核心要求）
        assert len(src_sentences) == len(tgt_sentences), \
            "源语言和目标语言句子数量不匹配"

        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self):
        """数据集总样本数"""
        return len(self.src_sentences)

    def __getitem__(self, idx):
        """动态编码：取数时实时对原始句子分词编码（核心改动）"""
        # 获取原始句子
        src_sent = self.src_sentences[idx]
        tgt_sent = self.tgt_sentences[idx]

        # 实时编码源语言（分词器自动添加<s>和</s>，符合论文格式）
        src_encoded = self.src_tokenizer.encode(src_sent)
        src_ids = torch.tensor(src_encoded.ids, dtype=torch.long)

        # 实时编码目标语言（用于Teacher Forcing，与论文一致）
        tgt_encoded = self.tgt_tokenizer.encode(tgt_sent)
        tgt_ids = torch.tensor(tgt_encoded.ids, dtype=torch.long)

        # 返回编码结果+序列长度（供collate_fn排序用）
        return {
            'src': src_ids,
            'tgt': tgt_ids,
            'src_len': len(src_ids),
            'tgt_len': len(tgt_ids)
        }


def load_and_clean_parallel_corpus(src_path: str, tgt_path: str) -> tuple:
    """加载并清洗平行语料（仅保留原始句子，不做编码，与论文预处理标准一致）

    含缓存机制：将清洗后的原始句对缓存至同目录的 cache/ 下，避免重复清洗。
    缓存键包含文件名与核心过滤参数（MAX_SEQ_LENGTH/MAX_LENGTH_RATIO/MIN_SENTENCE_LENGTH）。
    """
    # 构建缓存路径
    base_dir = os.path.dirname(src_path)
    cache_dir = os.path.join(base_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    cache_meta = {
        'src': os.path.basename(src_path),
        'tgt': os.path.basename(tgt_path),
        'MAX_SEQ_LENGTH': MAX_SEQ_LENGTH,
        'MAX_LENGTH_RATIO': MAX_LENGTH_RATIO,
        'MIN_SENTENCE_LENGTH': MIN_SENTENCE_LENGTH
    }
    cache_key = hashlib.md5(json.dumps(cache_meta, sort_keys=True).encode('utf-8')).hexdigest()[:16]
    cache_file = os.path.join(cache_dir, f"clean_{cache_meta['src']}__{cache_meta['tgt']}__{cache_key}.pt")

    if os.path.exists(cache_file):
        try:
            data = torch.load(cache_file, map_location='cpu')
            if isinstance(data, dict) and 'src_sents' in data and 'tgt_sents' in data:
                print(f"加载清洗缓存：{cache_file} | 句对数={len(data['src_sents'])}")
                return data['src_sents'], data['tgt_sents']
        except Exception as e:
            print(f"警告：读取缓存失败，将重新清洗。原因：{e}")
    src_sents = []
    tgt_sents = []

    # 检查文件是否存在
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"源语言文件不存在：{src_path}")
    if not os.path.exists(tgt_path):
        raise FileNotFoundError(f"目标语言文件不存在：{tgt_path}")

    # 计算源文件总行数（用于tqdm进度条）
    def file_line_count(file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f)

    total_lines = file_line_count(src_path)

    with open(src_path, 'r', encoding='utf-8', errors='ignore') as src_f, \
            open(tgt_path, 'r', encoding='utf-8', errors='ignore') as tgt_f:

        for line_idx, (src_line, tgt_line) in tqdm(enumerate(zip(src_f, tgt_f), 1), desc="平行语料清洗中...",total=total_lines):
            src_line = src_line.strip()
            tgt_line = tgt_line.strip()

            # 1. 过滤空句子（论文中唯一明确的过滤规则）
            if not src_line or not tgt_line:
                continue

            # 2. 过滤超长序列（预留2个位置给<s>和</s>，避免编码后超出模型最大长度）
            src_token_count = len(src_line.split())  # 按空格粗统计，避免提前分词
            tgt_token_count = len(tgt_line.split())
            if src_token_count > MAX_SEQ_LENGTH - 2 or tgt_token_count > MAX_SEQ_LENGTH - 2:
                continue

            # 3. 过滤极端长度比例的句对（避免标注错误数据）
            len_ratio = max(src_token_count / tgt_token_count, tgt_token_count / src_token_count)
            if len_ratio > MAX_LENGTH_RATIO:
                continue

            # 直接保留原始句子（不编码）
            src_sents.append(src_line)
            tgt_sents.append(tgt_line)

    print(f"平行语料清洗完成：共保留{len(src_sents)}个有效句对（仅原始句子，未编码）")

    # 写入缓存
    try:
        torch.save({'meta': cache_meta, 'src_sents': src_sents, 'tgt_sents': tgt_sents}, cache_file)
        print(f"已缓存清洗结果至：{cache_file}")
    except Exception as e:
        print(f"警告：写入缓存失败（不影响训练）：{e}")
    return src_sents, tgt_sents


def split_train_validation(src_train: list, tgt_train: list,
                           split_ratio: float = VALIDATION_SPLIT_RATIO) -> tuple:
    """从训练集拆分验证集（拆分原始句子列表，不涉及编码）"""
    print(f"\n从训练集拆分验证集：拆分比例={split_ratio}")
    # 绑定源-目标句对，避免拆分后错位
    combined_pairs = list(zip(src_train, tgt_train))
    # 固定种子确保可复现
    random.seed(42)
    random.shuffle(combined_pairs)

    # 计算拆分索引
    split_idx = int(len(combined_pairs) * (1 - split_ratio))
    train_pairs = combined_pairs[:split_idx]
    val_pairs = combined_pairs[split_idx:]

    # 还原为单独的原始句子列表
    src_train_split, tgt_train_split = zip(*train_pairs)
    src_val_split, tgt_val_split = zip(*val_pairs)

    print(f"拆分完成：训练集{len(src_train_split)}句对，验证集{len(src_val_split)}句对")
    return list(src_train_split), list(tgt_train_split), list(src_val_split), list(tgt_val_split)


def collate_fn(batch: list) -> dict:
    """批处理函数（逻辑不变，仍按长度排序+填充，符合论文动态批处理策略）"""
    # 按源语言序列长度降序排序（优化填充效率，减少padding数量）
    batch.sort(key=lambda x: x['src_len'], reverse=True)

    # 提取编码后的序列
    src_seqs = [item['src'] for item in batch]
    tgt_seqs = [item['tgt'] for item in batch]

    # 填充至批次内最大长度（序列长度在前，批次在后，符合Transformer输入格式）
    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_seqs, batch_first=False, padding_value=0  # <pad>的ID为0（分词器默认配置）
    )
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        tgt_seqs, batch_first=False, padding_value=0
    )

    # 构建填充掩码（1=有效token，0=padding，用于注意力层屏蔽无效token）
    src_mask = (src_padded != 0).float()
    tgt_mask = (tgt_padded != 0).float()

    return {
        'src': src_padded,  # 形状: (max_src_len, batch_size)
        'tgt': tgt_padded,  # 形状: (max_tgt_len, batch_size)
        'src_mask': src_mask,
        'tgt_mask': tgt_mask
    }


def get_wmt_dataloaders(data_dir: str, batch_size: int = 32,
                        use_predefined_val: bool = True) -> dict:
    """
    获取符合论文标准的WMT数据集加载器（动态编码模式：取数时实时分词）
    """
    # 1. 定义文件路径（遵循WMT数据集命名规范）
    file_paths = {
        'train_src': os.path.join(data_dir, 'train.en'),  # 英语训练集（原始文本）
        'train_tgt': os.path.join(data_dir, 'train.de'),  # 德语训练集（原始文本）
        #'train_src': os.path.join(data_dir, 'newstest2014.en'),  # 英语训练集（原始文本）
        #'train_tgt': os.path.join(data_dir, 'newstest2014.de'),  # 德语训练集（原始文本）
        'test_src': os.path.join(data_dir, 'newstest2014.en'),  # 英语测试集（原始文本）
        'test_tgt': os.path.join(data_dir, 'newstest2014.de'),  # 德语测试集（原始文本）
        'val_src': os.path.join(data_dir, 'newstest2013.en'),  # 英语验证集（原始文本）
        'val_tgt': os.path.join(data_dir, 'newstest2013.de')  # 德语验证集（原始文本）
    }

    # 2. 检查数据集目录是否存在
    if not os.path.exists(data_dir):
        raise NotADirectoryError(f"数据集目录不存在：{data_dir}")

    # 3. 加载/训练分词器（逻辑不变，仍从tokenizer.py调用，确保编码规则统一）
    tokenizer_dir = os.path.join(data_dir, 'tokenizers')
    os.makedirs(tokenizer_dir, exist_ok=True)
    src_tokenizer_path = os.path.join(tokenizer_dir, 'src_tokenizer_en.json')  # 英语分词器
    tgt_tokenizer_path = os.path.join(tokenizer_dir, 'tgt_tokenizer_de.json')  # 德语分词器

    ## 3.1 源语言（英语）分词器：优先加载，无则训练
    try:
        src_tokenizer = load_bpe_tokenizer(src_tokenizer_path)
    except FileNotFoundError:
        print("未找到英语分词器，开始训练...")
        src_tokenizer = train_bpe_tokenizer(
            data_files=[file_paths['train_src']],  # 用英语原始训练集训练
            save_path=src_tokenizer_path
        )

    ## 3.2 目标语言（德语）分词器：优先加载，无则训练
    try:
        tgt_tokenizer = load_bpe_tokenizer(tgt_tokenizer_path)
    except FileNotFoundError:
        print("未找到德语分词器，开始训练...")
        tgt_tokenizer = train_bpe_tokenizer(
            data_files=[file_paths['train_tgt']],  # 用德语原始训练集训练
            save_path=tgt_tokenizer_path
        )

    # 4. 处理训练集（加载原始文本→清洗→创建动态编码数据集）
    print("\n" + "=" * 50)
    print("开始处理训练集...")
    # 4.1 加载并清洗原始训练集（仅保留原始句子）
    src_train_raw, tgt_train_raw = load_and_clean_parallel_corpus(
        file_paths['train_src'], file_paths['train_tgt']
    )

    # 5. 处理验证集（优先newstest2013，无则从训练集拆分，均为原始句子）
    print("\n" + "=" * 50)
    print("开始处理验证集（动态编码模式）...")
    if use_predefined_val and os.path.exists(file_paths['val_src']) and os.path.exists(file_paths['val_tgt']):
        # 5.1 使用预定义验证集（newstest2013，原始文本）
        print("使用预定义验证集（newstest2013）...")
        src_val_raw, tgt_val_raw = load_and_clean_parallel_corpus(
            file_paths['val_src'], file_paths['val_tgt']
        )
        # 注意：使用预定义验证集时，训练集保持不变
    else:
        # 5.2 从训练集拆分验证集（拆分原始句子列表）
        print("未找到newstest2013，从训练集拆分验证集...")
        src_train_raw, tgt_train_raw, src_val_raw, tgt_val_raw = split_train_validation(
            src_train_raw, tgt_train_raw
        )
        # 拆分后，src_train_raw 和 tgt_train_raw 已经被更新为拆分后的训练集

    # 4.2 创建动态编码数据集（传入原始句子+分词器，取数时实时编码）
    train_dataset = WMTTranslationDataset(
        src_sentences=src_train_raw,
        tgt_sentences=tgt_train_raw,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer
    )

    # 5.3 创建验证集（动态编码）
    val_dataset = WMTTranslationDataset(
        src_sentences=src_val_raw,
        tgt_sentences=tgt_val_raw,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer
    )

    # 6. 处理测试集（newstest2014，原始文本+动态编码）
    print("\n" + "=" * 50)
    print("开始处理测试集（动态编码模式）...")
    # 6.1 加载并清洗原始测试集
    src_test_raw, tgt_test_raw = load_and_clean_parallel_corpus(
        file_paths['test_src'], file_paths['test_tgt']
    )
    # 6.2 创建测试集（动态编码）
    test_dataset = WMTTranslationDataset(
        src_sentences=src_test_raw,
        tgt_sentences=tgt_test_raw,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer
    )

    # 7. 创建数据加载器（逻辑不变，批处理仍按长度排序+填充）
    print("\n" + "=" * 50)
    print("开始创建数据加载器...")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=collate_fn,
        shuffle=True, num_workers=4, pin_memory=True  # pin_memory加速GPU数据传输
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=collate_fn,
        shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=collate_fn,
        shuffle=False, num_workers=4, pin_memory=True
    )

    print(f"数据加载器创建完成（动态编码模式）：")
    print(f"- 训练集：{len(train_loader)}个批次（每批{batch_size}个句对，取数时实时编码）")
    print(f"- 验证集：{len(val_loader)}个批次")
    print(f"- 测试集：{len(test_loader)}个批次")

    # 返回加载器和分词器（推理时需用分词器解码）
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'src_tokenizer': src_tokenizer,
        'tgt_tokenizer': tgt_tokenizer
    }


# 示例用法：验证动态编码流程
if __name__ == "__main__":
    # 替换为你的WMT2014英德数据集目录（需包含train.en/train.de等原始文本文件）
    DATA_DIR = "./dataset/WMT2014EngGer"

    try:
        # 获取动态编码模式的数据加载器
        dataloaders = get_wmt_dataloaders(
            data_dir=DATA_DIR,
            batch_size=1,
            use_predefined_val=True  # 优先使用newstest2013作为验证集
        )

        # 验证第一个训练批次（动态编码结果）
        print("\n" + "=" * 50)
        print("验证训练集批次（动态编码模式）：")
        for batch in dataloaders['train']:
            # 打印张量形状（与预编码模式一致，符合Transformer输入）
            print(f"源语言序列形状: {batch['src'].shape} → (max_src_len, batch_size)")
            print(f"目标语言序列形状: {batch['tgt'].shape} → (max_tgt_len, batch_size)")
            print(f"源语言掩码形状: {batch['src_mask'].shape}（与源序列形状一致）")

            # 解码示例句子（验证动态编码正确性）
            src_tokenizer = dataloaders['src_tokenizer']
            tgt_tokenizer = dataloaders['tgt_tokenizer']
            # 取第一个样本的ID解码为文本
            src_sample_ids = batch['src'][:, 0].numpy()
            tgt_sample_ids = batch['tgt'][:, 0].numpy()
            src_sample_text = src_tokenizer.decode(src_sample_ids)
            tgt_sample_text = tgt_tokenizer.decode(tgt_sample_ids)

            print(f"\n第一个样本（源语言-英语）：{src_sample_text}")
            print(f"第一个样本（目标语言-德语）：{tgt_sample_text}")
            break  # 仅验证第一个批次

        print("\n" + "=" * 50)
        print("动态编码模式数据处理流程验证完成，符合论文规范！")

    except Exception as e:
        print(f"\n数据处理失败：{str(e)}")