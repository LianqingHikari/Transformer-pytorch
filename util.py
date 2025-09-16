import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from torch import Tensor
import torch
from typing import Tuple
import torch.nn as nn

def calculate_bleu_score(predicted, target, tokenizer):
    """按token级别计算BLEU。

    改进：
    - 在首个 </s> 处截断预测与参考，避免EOS后的内容影响
    - 过滤 <pad>/<s>/</s>
    - 根据实际序列长度动态设置n-gram权重（长度<4时不被4-gram稀释）
    - 空序列时用占位符兜底，避免BLEU报错
    """
    predictions = []
    references = []

    specials = {'<pad>', '<s>', '</s>'}

    def ids_to_clean_tokens(id_list):
        toks = [tokenizer.id_to_token(int(i)) for i in id_list]
        if '</s>' in toks:
            toks = toks[:toks.index('</s>')]
        toks = [t for t in toks if t not in specials]
        return toks if toks else ['<unk>']

    for pred, tgt in zip(predicted, target):
        pred_ids = pred.tolist() if hasattr(pred, 'tolist') else list(pred)
        tgt_ids = tgt.tolist() if hasattr(tgt, 'tolist') else list(tgt)

        pred_tokens = ids_to_clean_tokens(pred_ids)
        tgt_tokens = ids_to_clean_tokens(tgt_ids)

        predictions.append(pred_tokens)
        references.append([tgt_tokens])

    # 动态n-gram权重：n = min(4, min_len)
    if len(predictions) == 0:
        return 0.0
    # 估计有效长度（使用参考长度更稳定）
    ref_lens = [len(ref[0]) for ref in references]
    min_len = max(1, min(ref_lens))
    n = min(4, min_len)
    weights = tuple([1.0 / n] * n + [0.0] * (4 - n))

    # 对极短句不做平滑，长度>=4时使用平滑以提升稳定性
    smoothing = None if n < 4 else SmoothingFunction().method4
    bleu_score = corpus_bleu(references, predictions, weights=weights, smoothing_function=smoothing)
    return bleu_score * 100


def plot_training_curves(train_losses, val_losses, val_bleus, out_path: str = 'training_curves.png'):
    """绘制训练/验证损失与BLEU曲线并保存。"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    if train_losses:
        plt.plot(train_losses, label='训练损失')
    if val_losses:
        plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()

    plt.subplot(1, 2, 2)
    if val_bleus:
        plt.plot(val_bleus, label='验证BLEU分数')
        plt.xlabel('Epoch')
        plt.ylabel('BLEU分数')
        plt.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    return out_path


def generate_mask(src: Tensor, tgt: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """生成注意力掩码，用于屏蔽填充位置和未来信息"""
    src_seq_len = src.size(0)
    tgt_seq_len = tgt.size(0)

    # 源序列自注意力掩码（全零，允许关注所有位置）
    src_mask = torch.zeros(src_seq_len, src_seq_len).float().to(src.device)

    # 目标序列自注意力掩码（防止关注未来的词）
    tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len) * float('-inf'), diagonal=1).to(tgt.device)

    # 交叉注意力掩码（解码器->编码器），形状为(tgt_seq_len, src_seq_len)
    # 全零表示允许解码器关注编码器的所有位置
    cross_mask = torch.zeros(tgt_seq_len, src_seq_len).float().to(src.device)

    # 填充掩码，假设0是填充符号
    src_pad_mask = (src == 0).transpose(0, 1)  # 形状: (batch_size, src_seq_len)
    tgt_pad_mask = (tgt == 0).transpose(0, 1)  # 形状: (batch_size, tgt_seq_len)

    return src_mask, tgt_mask, cross_mask, src_pad_mask, tgt_pad_mask

def count_parameters(model: nn.Module) -> int:
    """计算模型的总参数量"""
    return sum(p.numel() for p in model.parameters())