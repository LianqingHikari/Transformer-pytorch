# data_processing.py
# 动态编码模式：取数时实时对原始句子编码，不提前缓存
import os
import torch
import random
import pickle  # 新增：用于序列化/反序列化缓存数据
from typing import Optional  # 新增：用于可选参数的类型注解
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
# 从tokenizer.py导入分词器相关配置与函数
from tokenizer import SPECIAL_TOKENS, train_bpe_tokenizer, load_bpe_tokenizer

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


def load_and_clean_parallel_corpus(
    src_path: str,
    tgt_path: str,
    cache_dir: Optional[str] = None  # 新增：缓存目录（默认使用源文件所在目录）
) -> tuple:
    """加载并清洗平行语料（支持缓存：首次处理保存，后续直接加载）"""
    src_sents = []
    tgt_sents = []

    # -------------------------- 新增：缓存路径处理 --------------------------
    # 1. 确定缓存目录（默认用源文件所在目录，用户可自定义）
    if cache_dir is None:
        cache_dir = os.path.dirname(src_path)  # 源文件所在目录
    os.makedirs(cache_dir, exist_ok=True)  # 确保缓存目录存在（不存在则创建）

    # 2. 生成唯一缓存文件名（基于源/目标文件的文件名，避免不同文件缓存冲突）
    src_filename = os.path.splitext(os.path.basename(src_path))[0]  # 源文件前缀（如"train.en"→"train"）
    tgt_filename = os.path.splitext(os.path.basename(tgt_path))[0]  # 目标文件前缀（如"train.de"→"train"）
    cache_filename = f"{src_filename}_vs_{tgt_filename}_cleaned.pkl"  # 缓存文件名（如"train_vs_train_cleaned.pkl"）
    cache_path = os.path.join(cache_dir, cache_filename)  # 完整缓存路径

    # 3. 检查缓存是否存在：存在则加载（加载失败则重新处理）
    if os.path.exists(cache_path):
        print(f"\n[缓存机制] 发现缓存文件：{cache_path}")
        try:
            with open(cache_path, "rb") as f:
                src_sents, tgt_sents = pickle.load(f)  # 反序列化加载缓存
            print(f"[缓存机制] 缓存加载成功！共包含 {len(src_sents)} 个有效句对")
            return src_sents, tgt_sents  # 直接返回加载的缓存数据
        except Exception as e:
            print(f"[缓存机制] 缓存文件损坏或加载失败：{str(e)}，将重新处理数据")

    # -------------------------- 原有逻辑：数据清洗 --------------------------
    # 检查原始文件是否存在
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"源语言文件不存在：{src_path}")
    if not os.path.exists(tgt_path):
        raise FileNotFoundError(f"目标语言文件不存在：{tgt_path}")

    # 计算源文件总行数（用于tqdm进度条）
    def file_line_count(file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f)

    total_lines = file_line_count(src_path)

    # 逐行读取并清洗数据
    with open(src_path, 'r', encoding='utf-8', errors='ignore') as src_f, \
         open(tgt_path, 'r', encoding='utf-8', errors='ignore') as tgt_f:

        for line_idx, (src_line, tgt_line) in tqdm(
            enumerate(zip(src_f, tgt_f), 1),
            desc=f"[数据清洗] 处理 {src_filename} vs {tgt_filename}...",
            total=total_lines
        ):
            src_line = src_line.strip()
            tgt_line = tgt_line.strip()

            # 1. 过滤空句子
            if not src_line or not tgt_line:
                continue

            # 2. 过滤超长序列（预留2个位置给<s>/</s>）
            src_token_count = len(src_line.split())  # 按空格粗统计（避免提前分词）
            tgt_token_count = len(tgt_line.split())
            if src_token_count > MAX_SEQ_LENGTH - 2 or tgt_token_count > MAX_SEQ_LENGTH - 2:
                continue

            # 3. 过滤极端长度比例的句对
            len_ratio = max(src_token_count / tgt_token_count, tgt_token_count / src_token_count)
            if len_ratio > MAX_LENGTH_RATIO:
                continue

            # 保留清洗后的原始句子
            src_sents.append(src_line)
            tgt_sents.append(tgt_line)

    # -------------------------- 新增：保存缓存 --------------------------
    print(f"\n[缓存机制] 数据清洗完成，共保留 {len(src_sents)} 个有效句对")
    try:
        with open(cache_path, "wb") as f:
            pickle.dump((src_sents, tgt_sents), f)  # 序列化保存清洗后的数据
        print(f"[缓存机制] 缓存已保存至：{cache_path}（下次可直接加载）")
    except Exception as e:
        print(f"[缓存机制] 缓存保存失败：{str(e)}（但数据清洗结果已正常返回）")

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
    获取符合论文标准的WMT数据集加载器（动态编码模式：取数时实时分词，支持数据缓存）
    Args:
        data_dir: 数据集根目录（需包含train.en/train.de等原始文本文件）
        batch_size: 批次大小（默认32，符合Transformer训练常规设置）
        use_predefined_val: 是否使用预定义验证集（newstest2013），默认True
    Returns:
        dict: 包含训练/验证/测试加载器，及源/目标语言分词器
    """
    # 1. 定义文件路径（严格遵循WMT2014英德数据集命名规范）
    file_paths = {
        'train_src': os.path.join(data_dir, 'train.en'),  # 英语训练集（原始文本）
        'train_tgt': os.path.join(data_dir, 'train.de'),  # 德语训练集（原始文本）
        'test_src': os.path.join(data_dir, 'newstest2014.en'),  # 英语测试集（原始文本）
        'test_tgt': os.path.join(data_dir, 'newstest2014.de'),  # 德语测试集（原始文本）
        'val_src': os.path.join(data_dir, 'newstest2013.en'),  # 英语验证集（newstest2013）
        'val_tgt': os.path.join(data_dir, 'newstest2013.de')   # 德语验证集（newstest2013）
    }

    # 2. 基础校验：检查数据集目录是否存在
    if not os.path.exists(data_dir):
        raise NotADirectoryError(f"数据集目录不存在：{data_dir}，请确认路径正确")

    # 3. 初始化关键目录（分词器目录+数据缓存目录）
    # 3.1 分词器保存目录（统一放在数据集目录下的tokenizers子目录）
    tokenizer_dir = os.path.join(data_dir, 'tokenizers')
    os.makedirs(tokenizer_dir, exist_ok=True)  # 目录不存在则创建
    # 3.2 数据缓存目录（统一放在数据集目录下的cache子目录，集中管理缓存文件）
    cache_root = os.path.join(data_dir, 'cache')
    os.makedirs(cache_root, exist_ok=True)  # 确保缓存目录存在

    # 4. 加载/训练分词器（优先加载已有分词器，无则用训练集训练，保证编码一致性）
    # 4.1 源语言（英语）分词器：路径+加载/训练逻辑
    src_tokenizer_path = os.path.join(tokenizer_dir, 'src_tokenizer_en.json')
    try:
        src_tokenizer = load_bpe_tokenizer(src_tokenizer_path)
        print(f"✅ 成功加载英语分词器：{src_tokenizer_path}")
    except FileNotFoundError:
        print(f"❌ 未找到英语分词器，将用英语训练集（{file_paths['train_src']}）训练...")
        src_tokenizer = train_bpe_tokenizer(
            data_files=[file_paths['train_src']],  # 仅用英语训练集原始文本训练
            save_path=src_tokenizer_path,
            special_tokens=SPECIAL_TOKENS  # 传入特殊token配置（来自tokenizer.py）
        )
        print(f"✅ 英语分词器训练完成，已保存至：{src_tokenizer_path}")

    # 4.2 目标语言（德语）分词器：路径+加载/训练逻辑
    tgt_tokenizer_path = os.path.join(tokenizer_dir, 'tgt_tokenizer_de.json')
    try:
        tgt_tokenizer = load_bpe_tokenizer(tgt_tokenizer_path)
        print(f"✅ 成功加载德语分词器：{tgt_tokenizer_path}")
    except FileNotFoundError:
        print(f"❌ 未找到德语分词器，将用德语训练集（{file_paths['train_tgt']}）训练...")
        tgt_tokenizer = train_bpe_tokenizer(
            data_files=[file_paths['train_tgt']],  # 仅用德语训练集原始文本训练
            save_path=tgt_tokenizer_path,
            special_tokens=SPECIAL_TOKENS  # 统一特殊token配置
        )
        print(f"✅ 德语分词器训练完成，已保存至：{tgt_tokenizer_path}")

    # 5. 处理训练集（带缓存：优先加载缓存，无则清洗并保存缓存）
    print("\n" + "=" * 60)
    print("📥 开始处理训练集（英→德）...")
    # 5.1 调用带缓存的语料清洗函数（传入缓存目录，自动处理缓存逻辑）
    src_train_raw, tgt_train_raw = load_and_clean_parallel_corpus(
        src_path=file_paths['train_src'],
        tgt_path=file_paths['train_tgt'],
        cache_dir=cache_root  # 关键：启用缓存机制
    )
    # 5.2 创建动态编码数据集（取数时实时分词，不提前缓存编码结果）
    train_dataset = WMTTranslationDataset(
        src_sentences=src_train_raw,
        tgt_sentences=tgt_train_raw,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer
    )
    print(f"✅ 训练集创建完成：共{len(train_dataset)}个有效句对（动态编码模式）")

    # 6. 处理验证集（两种模式：预定义newstest2013 / 从训练集拆分，均支持缓存）
    print("\n" + "=" * 60)
    print("📥 开始处理验证集...")
    if use_predefined_val and os.path.exists(file_paths['val_src']) and os.path.exists(file_paths['val_tgt']):
        # 6.1 模式1：使用预定义验证集（newstest2013，推荐，符合论文评估标准）
        print(f"ℹ️ 使用预定义验证集：newstest2013（英→德）")
        src_val_raw, tgt_val_raw = load_and_clean_parallel_corpus(
            src_path=file_paths['val_src'],
            tgt_path=file_paths['val_tgt'],
            cache_dir=cache_root  # 启用缓存
        )
    else:
        # 6.2 模式2：从训练集拆分验证集（无预定义验证集时降级使用）
        print(f"ℹ️ 未找到newstest2013，将从训练集拆分验证集（拆分比例={VALIDATION_SPLIT_RATIO}）")
        src_train_raw, tgt_train_raw, src_val_raw, tgt_val_raw = split_train_validation(
            src_train=src_train_raw,
            tgt_train=tgt_train_raw,
            split_ratio=VALIDATION_SPLIT_RATIO
        )
        # 注意：拆分后训练集原始句子变化，需重新创建训练集
        train_dataset = WMTTranslationDataset(
            src_sentences=src_train_raw,
            tgt_sentences=tgt_train_raw,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer
        )
        print(f"ℹ️ 拆分后训练集：{len(train_dataset)}个句对，验证集：{len(src_val_raw)}个句对")

    # 6.3 创建验证集（动态编码模式）
    val_dataset = WMTTranslationDataset(
        src_sentences=src_val_raw,
        tgt_sentences=tgt_val_raw,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer
    )
    print(f"✅ 验证集创建完成：共{len(val_dataset)}个有效句对（动态编码模式）")

    # 7. 处理测试集（固定使用newstest2014，支持缓存）
    print("\n" + "=" * 60)
    print("📥 开始处理测试集（英→德，newstest2014）...")
    # 7.1 调用带缓存的语料清洗函数
    src_test_raw, tgt_test_raw = load_and_clean_parallel_corpus(
        src_path=file_paths['test_src'],
        tgt_path=file_paths['test_tgt'],
        cache_dir=cache_root  # 启用缓存
    )
    # 7.2 创建测试集（动态编码模式）
    test_dataset = WMTTranslationDataset(
        src_sentences=src_test_raw,
        tgt_sentences=tgt_test_raw,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer
    )
    print(f"✅ 测试集创建完成：共{len(test_dataset)}个有效句对（动态编码模式）")

    # 8. 创建数据加载器（动态批处理：按序列长度排序+填充，优化GPU效率）
    print("\n" + "=" * 60)
    print(f"🚀 开始创建数据加载器（批次大小={batch_size}）...")
    # 8.1 训练集加载器（shuffle=True，训练时打乱数据）
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,  # 自定义批处理函数（排序+填充+掩码）
        shuffle=True,
        num_workers=4,  # 多线程加载（根据CPU核心数调整，建议≤CPU核心数）
        pin_memory=True,  # 锁定内存，加速GPU数据传输（需配合GPU使用）
        drop_last=False  # 不丢弃最后一个不完整批次（避免数据浪费）
    )
    # 8.2 验证集加载器（shuffle=False，评估时固定顺序）
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    # 8.3 测试集加载器（shuffle=False，测试时固定顺序）
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    # 9. 打印加载器统计信息（方便用户确认数据规模）
    print(f"\n📊 数据加载器创建完成（动态编码+缓存模式）：")
    print(f"- 训练集：{len(train_loader)} 个批次（总计 {len(train_dataset)} 个句对）")
    print(f"- 验证集：{len(val_loader)} 个批次（总计 {len(val_dataset)} 个句对）")
    print(f"- 测试集：{len(test_loader)} 个批次（总计 {len(test_dataset)} 个句对）")
    print(f"- 缓存目录：{cache_root}（下次运行将优先加载缓存）")
    print(f"- 分词器目录：{tokenizer_dir}（编码规则已固定）")

    # 10. 返回结果（加载器+分词器，分词器用于后续推理时的解码）
    return {
        'train': train_loader,    # 训练集加载器
        'val': val_loader,        # 验证集加载器
        'test': test_loader,      # 测试集加载器
        'src_tokenizer': src_tokenizer,  # 英语分词器（源语言）
        'tgt_tokenizer': tgt_tokenizer   # 德语分词器（目标语言）
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