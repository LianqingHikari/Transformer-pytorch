import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from torch import Tensor
import torch
from typing import Tuple
import torch.nn as nn


def calculate_bleu_score(predicted, target, tokenizer):
    specials = {'<pad>', '<s>', '</s>', '<unk>'}  # 扩展特殊标记
    refs = []
    hyps = []

    for pred, tgt in zip(predicted, target):
        # 转换token ID到文本并过滤特殊标记
        pred_tokens = [
            tokenizer.id_to_token(int(i))
            for i in (pred.tolist() if hasattr(pred, 'tolist') else pred)
            if tokenizer.id_to_token(int(i)) not in specials
        ]
        tgt_tokens = [
            tokenizer.id_to_token(int(i))
            for i in (tgt.tolist() if hasattr(tgt, 'tolist') else tgt)
            if tokenizer.id_to_token(int(i)) not in specials
        ]
        # 遇到</s>提前停止（如果存在）
        try:
            pred_tokens = pred_tokens[:pred_tokens.index('</s>')]
        except ValueError:
            pass
        try:
            tgt_tokens = tgt_tokens[:tgt_tokens.index('</s>')]
        except ValueError:
            pass

        if tgt_tokens:  # 跳过空参考句
            refs.append([tgt_tokens])
            hyps.append(pred_tokens)

    # 语料库级BLEU计算（标准4-gram权重）
    smoothing = SmoothingFunction().method1  # 常用平滑方法
    score = corpus_bleu(refs, hyps, smoothing_function=smoothing)
    return score * 100  # 转换为百分比形式


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