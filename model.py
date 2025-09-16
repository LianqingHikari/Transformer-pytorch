import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from typing import Optional, Tuple, List
from util import generate_mask

class PositionalEncoding(nn.Module):
    """位置编码模块，为输入序列添加位置信息"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 计算位置编码
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # 不参与训练的参数

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: 输入张量，形状为 (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == self.embed_dim, "嵌入维度必须能被头数整除"

        # 线性变换层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            attn_mask: Optional[Tensor] = None,
            padding_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            query: 查询张量，形状为 (seq_len_q, batch_size, embed_dim)
            key: 键张量，形状为 (seq_len_k, batch_size, embed_dim)
            value: 值张量，形状为 (seq_len_v, batch_size, embed_dim)
            attn_mask: 注意力掩码，形状为 (seq_len_q, seq_len_k)
            padding_mask: 填充掩码，形状为 (batch_size, seq_len_k)

        Returns:
            输出张量和注意力权重
        """
        batch_size = query.size(1)

        # 线性变换并分拆多头
        query = self.q_proj(query).view(-1, batch_size, self.num_heads, self.head_dim).transpose(0, 2)
        key = self.k_proj(key).view(-1, batch_size, self.num_heads, self.head_dim).transpose(0, 2)
        value = self.v_proj(value).view(-1, batch_size, self.num_heads, self.head_dim).transpose(0, 2)

        # 计算注意力分数 (num_heads, batch_size, seq_len_q, seq_len_k)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 应用注意力掩码
        if attn_mask is not None:
            # 确保掩码形状正确并添加到注意力分数上
            # 从 (seq_len_q, seq_len_k) 变为 (1, 1, seq_len_q, seq_len_k)
            attn_scores = attn_scores + attn_mask.unsqueeze(0).unsqueeze(0)

        # 应用填充掩码
        if padding_mask is not None:
            # 调整填充掩码形状以适应注意力分数
            # 从 (batch_size, seq_len_k) 变为 (1, batch_size, 1, seq_len_k)
            padding_mask = padding_mask.unsqueeze(0).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(padding_mask, float('-inf'))

        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力
        output = torch.matmul(attn_weights, value)

        # 拼接多头结果
        output = output.transpose(0, 2).contiguous().view(-1, batch_size, self.embed_dim)

        # 最终线性变换
        output = self.out_proj(output)

        # 调整注意力权重的维度顺序
        attn_weights = attn_weights.transpose(0, 1).contiguous()

        return output, attn_weights


class PositionWiseFeedForward(nn.Module):
    """位置-wise前馈网络"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # 使用GELU激活函数，比ReLU效果更好

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    """编码器层"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # dropout层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
            self,
            x: Tensor,
            src_mask: Tensor,
            src_pad_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # 自注意力子层
        attn_output, attn_weights = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=src_mask,
            padding_mask=src_pad_mask
        )
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # 前馈子层
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x, attn_weights


class DecoderLayer(nn.Module):
    """解码器层"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)  # 掩蔽自注意力
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)  # 编码器-解码器注意力
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # dropout层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
            self,
            x: Tensor,
            enc_output: Tensor,
            tgt_mask: Tensor,
            cross_mask: Tensor,  # 新增：交叉注意力掩码
            tgt_pad_mask: Tensor,
            src_pad_mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # 掩蔽自注意力子层
        attn_output1, attn_weights1 = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=tgt_mask,
            padding_mask=tgt_pad_mask
        )
        x = x + self.dropout1(attn_output1)
        x = self.norm1(x)

        # 编码器-解码器注意力子层（使用交叉注意力掩码）
        attn_output2, attn_weights2 = self.cross_attn(
            query=x,
            key=enc_output,
            value=enc_output,
            attn_mask=cross_mask,  # 改为使用交叉注意力掩码
            padding_mask=src_pad_mask
        )
        x = x + self.dropout2(attn_output2)
        x = self.norm2(x)

        # 前馈子层
        ff_output = self.feed_forward(x)
        x = x + self.dropout3(ff_output)
        x = self.norm3(x)

        return x, attn_weights1, attn_weights2


class Encoder(nn.Module):
    """完整的编码器"""

    def __init__(
            self,
            num_layers: int,
            d_model: int,
            num_heads: int,
            d_ff: int,
            input_vocab_size: int,
            dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model

        # 词嵌入层
        self.embedding = nn.Embedding(input_vocab_size, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 编码器层堆叠
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(
            self,
            src: Tensor,
            src_mask: Tensor,
            src_pad_mask: Tensor
    ) -> Tuple[Tensor, List[Tensor]]:
        # 词嵌入并缩放
        x = self.embedding(src) * math.sqrt(self.d_model)

        # 添加位置编码
        x = self.pos_encoder(x)

        # 通过所有编码器层
        attn_weights_list = []
        for layer in self.layers:
            x, attn_weights = layer(x, src_mask, src_pad_mask)
            attn_weights_list.append(attn_weights)

        return x, attn_weights_list


class Decoder(nn.Module):
    """完整的解码器"""

    def __init__(
            self,
            num_layers: int,
            d_model: int,
            num_heads: int,
            d_ff: int,
            target_vocab_size: int,
            dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model

        # 词嵌入层
        self.embedding = nn.Embedding(target_vocab_size, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 解码器层堆叠
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(
            self,
            tgt: Tensor,
            enc_output: Tensor,
            tgt_mask: Tensor,
            cross_mask: Tensor,  # 新增：交叉注意力掩码
            tgt_pad_mask: Tensor,
            src_pad_mask: Tensor
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        # 词嵌入并缩放
        x = self.embedding(tgt) * math.sqrt(self.d_model)

        # 添加位置编码
        x = self.pos_encoder(x)

        # 通过所有解码器层
        self_attn_weights_list = []
        cross_attn_weights_list = []
        for layer in self.layers:
            x, self_attn_weights, cross_attn_weights = layer(
                x, enc_output, tgt_mask, cross_mask, tgt_pad_mask, src_pad_mask
            )
            self_attn_weights_list.append(self_attn_weights)
            cross_attn_weights_list.append(cross_attn_weights)

        return x, self_attn_weights_list, cross_attn_weights_list


class Transformer(nn.Module):
    """完整的Transformer模型"""

    def __init__(
            self,
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 6,
            d_model: int = 512,
            num_heads: int = 8,
            d_ff: int = 2048,
            input_vocab_size: int = 50000,
            target_vocab_size: int = 50000,
            dropout: float = 0.1
    ):
        super().__init__()

        # 编码器
        self.encoder = Encoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            input_vocab_size=input_vocab_size,
            dropout=dropout
        )

        # 解码器
        self.decoder = Decoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            target_vocab_size=target_vocab_size,
            dropout=dropout
        )
        self.decoder.embedding = self.encoder.embedding

        # 最终输出层
        self.fc_out = nn.Linear(d_model, target_vocab_size)
        self.fc_out.weight = self.encoder.embedding.weight

    def forward(
            self,
            src: Tensor,
            tgt: Tensor
    ) -> Tuple[Tensor, List[Tensor], List[Tensor], List[Tensor]]:
        # 生成掩码，新增交叉注意力掩码
        src_mask, tgt_mask, cross_mask, src_pad_mask, tgt_pad_mask = generate_mask(src, tgt)

        # 编码器输出
        enc_output, enc_attn_weights = self.encoder(src, src_mask, src_pad_mask)

        # 解码器输出（传入交叉注意力掩码）
        dec_output, dec_self_attn_weights, dec_cross_attn_weights = self.decoder(
            tgt, enc_output, tgt_mask, cross_mask, tgt_pad_mask, src_pad_mask
        )

        # 最终输出
        output = self.fc_out(dec_output)

        return output, enc_attn_weights, dec_self_attn_weights, dec_cross_attn_weights

    def greedy_decode(
            self,
            src: Tensor,
            max_len: int,
            start_symbol: int,
            end_symbol: int,
            device: torch.device,
            pad_symbol: int = 0
    ) -> Tensor:
        """贪婪解码（按样本处理EOS，已结束的样本后续强制填充EOS）

        返回形状: (batch_size, decoded_len)
        """
        self.eval()
        batch_size = src.size(1)

        # 初始目标序列，仅<BOS>
        tgt = torch.full((1, batch_size), start_symbol, dtype=src.dtype, device=device)

        with torch.no_grad():
            # 编码源序列（一次）
            src_mask, _, _, src_pad_mask, _ = generate_mask(src, src)
            enc_output, _ = self.encoder(src, src_mask, src_pad_mask)

            # 跟踪每个样本是否已生成<EOS>
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for _ in range(max_len - 1):
                # 动态构造当前tgt所需掩码
                _, tgt_mask, cross_mask, _, tgt_pad_mask = generate_mask(src, tgt)

                # 解码并取最后一步预测
                dec_output, _, _ = self.decoder(tgt, enc_output, tgt_mask, cross_mask, tgt_pad_mask, src_pad_mask)
                logits = self.fc_out(dec_output)  # (tgt_len, batch, vocab)
                next_token = logits[-1].argmax(dim=-1)  # (batch,)

                # 已完成的样本强制输出<EOS>
                next_token = torch.where(
                    finished, torch.as_tensor(end_symbol, device=device, dtype=next_token.dtype), next_token
                )

                # 追加到序列
                tgt = torch.cat([tgt, next_token.unsqueeze(0)], dim=0)

                # 更新完成标记
                finished = finished | (next_token == end_symbol)

                # 所有样本均完成则提前停止
                if finished.all():
                    break

        return tgt.transpose(0, 1)


# 示例用法
if __name__ == "__main__":
    # 模型参数
    input_vocab_size = 37000
    target_vocab_size = 37000
    d_model = 512
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    d_ff = 2048
    dropout = 0.1

    # 初始化模型
    transformer = Transformer(
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        dropout=dropout
    )


    def count_parameters(model: nn.Module) -> int:
        """计算模型的总参数量"""
        return sum(p.numel() for p in model.parameters())

    print(count_parameters(transformer))

    # 随机生成示例数据 (seq_len, batch_size)
    src_seq_len = 10
    tgt_seq_len = 15
    batch_size = 2
    src = torch.randint(1, input_vocab_size, (src_seq_len, batch_size))  # 避免0，0作为填充符
    tgt = torch.randint(1, target_vocab_size, (tgt_seq_len, batch_size))

    # 前向传播
    output, enc_attn, dec_self_attn, dec_cross_attn = transformer(src, tgt)

    # 打印输出形状
    print(f"输入源序列形状: {src.shape}")
    print(f"输入目标序列形状: {tgt.shape}")
    print(f"输出序列形状: {output.shape}")
    print(f"编码器注意力权重形状: {len(enc_attn)} 层, 每层形状 {enc_attn[0].shape}")
    print(f"解码器自注意力权重形状: {len(dec_self_attn)} 层, 每层形状 {dec_self_attn[0].shape}")
    print(f"解码器交叉注意力权重形状: {len(dec_cross_attn)} 层, 每层形状 {dec_cross_attn[0].shape}")
