#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
from typing import List

import config as C
from model import Transformer
from data_processing import get_wmt_dataloaders


def ids_trim_eos(ids: List[int], tokenizer) -> List[int]:
    """在首个</s>截断，并移除尾部padding(0)。"""
    # 去掉末尾连续的0
    while ids and ids[-1] == 0:
        ids.pop()
    # 在</s>处截断
    try:
        eos_id = tokenizer.token_to_id("</s>")
        if eos_id in ids:
            cut = ids.index(eos_id)
            ids = ids[:cut]
    except Exception:
        pass
    return ids


def decode_ids(ids: List[int], tokenizer) -> str:
    """安全解码：先截断，再decode。"""
    trimmed = ids_trim_eos(list(ids), tokenizer)
    if not trimmed:
        return ""
    try:
        return tokenizer.decode(trimmed)
    except Exception:
        # 失败则退回逐token拼接
        toks = [tokenizer.id_to_token(int(i)) for i in trimmed]
        return " ".join(toks)


def main():
    device = C.DEVICE if hasattr(C, 'DEVICE') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据与分词器
    dataloaders = get_wmt_dataloaders(
        data_dir='./dataset/WMT2014EngGer',
        batch_size=1,
        use_predefined_val=True
    )
    val_loader = dataloaders['val']
    tgt_tokenizer = dataloaders['tgt_tokenizer']
    src_tokenizer = dataloaders['src_tokenizer']

    src_vocab_size = src_tokenizer.get_vocab_size()
    tgt_vocab_size = tgt_tokenizer.get_vocab_size()

    # 模型构建
    model = Transformer(
        num_encoder_layers=C.NUM_ENCODER_LAYERS,
        num_decoder_layers=C.NUM_DECODER_LAYERS,
        d_model=C.D_MODEL,
        num_heads=C.NUM_HEADS,
        d_ff=C.D_FF,
        input_vocab_size=src_vocab_size,
        target_vocab_size=tgt_vocab_size,
        dropout=C.DROPOUT
    ).to(device)

    # 加载权重
    # ckpt_path = os.path.join(C.CHECKPOINT_DIR if hasattr(C, 'CHECKPOINT_DIR') else 'checkpoints', 'epoch_4.pth')
    # if not os.path.exists(ckpt_path):
    #     raise FileNotFoundError(f"未找到检查点：{ckpt_path}")
    # checkpoint = torch.load(ckpt_path, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 取验证集第一个batch
    batch = next(iter(val_loader))
    src = batch['src'].to(device)  # (src_len, 1)
    tgt = batch['tgt'].to(device)  # (tgt_len, 1)

    # 准备解码配置
    start_symbol = tgt_tokenizer.token_to_id("<s>")
    end_symbol = tgt_tokenizer.token_to_id("</s>")
    max_len = C.MAX_SEQ_LENGTH if hasattr(C, 'MAX_SEQ_LENGTH') else 128

    # 推理
    with torch.no_grad():
        pred_ids_batched = model.greedy_decode(
            src=src,
            max_len=max_len,
            start_symbol=start_symbol,
            end_symbol=end_symbol,
            device=device,
            pad_symbol=0
        )  # (batch, decoded_len)

    # 仅batch=1，取第一个样本
    pred_ids = pred_ids_batched[0].tolist()

    # 参考标签（去掉首token <s> 更符合阅读习惯）
    tgt_ids_full = tgt[:, 0].tolist()
    # 去掉开头<s>
    if tgt_ids_full and tgt_ids_full[0] == start_symbol:
        tgt_ids_ref = tgt_ids_full[1:]
    else:
        tgt_ids_ref = tgt_ids_full

    pred_text = decode_ids(pred_ids, tgt_tokenizer)
    ref_text = decode_ids(tgt_ids_ref, tgt_tokenizer)

    print("===== 推理结果（验证集第一个样本） =====")
    print(f"源文本: {decode_ids(src[:,0].tolist(), src_tokenizer)}")
    print(f"参考译文: {ref_text}")
    print(f"模型译文: {pred_text}")


if __name__ == "__main__":
    main()


