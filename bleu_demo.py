#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最小BLEU验证脚本：
 - 使用一个简易分词器（仅实现 id_to_token）
 - 构造几组小样例，验证 util.calculate_bleu_score 的行为
用法：python bleu_demo.py
"""

from util import calculate_bleu_score


class FakeTokenizer:
    def __init__(self):
        # 简单词表与特殊符号
        # 0:<pad>, 1:<unk>, 2:<s>, 3:</s>
        self.id2tok = {
            0: "<pad>",
            1: "<unk>",
            2: "<s>",
            3: "</s>",
            4: "I",
            5: "love",
            6: "NLP",
            7: "Deep",
            8: "Learning",
        }

    def id_to_token(self, idx: int) -> str:
        return self.id2tok.get(idx, "<unk>")


def main():
    tokenizer = FakeTokenizer()

    # 案例1：完全匹配（短句，动态权重应给到1/3-gram，分数接近100）
    preds_1 = [[4, 5, 6]]                 # I love NLP
    tgts_1  = [[4, 5, 6]]                 # I love NLP
    bleu_1 = calculate_bleu_score(preds_1, tgts_1, tokenizer)

    # 案例2：部分匹配（有2-gram匹配，BLEU介于0与100之间）
    preds_2 = [[4, 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4, 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]]                 # I love NLP
    tgts_2  = [[4, 5, 7, 8,10,11,12,13,14,15,16,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4, 5, 7, 8,10,11,12,13,14,15,16,5,5,5,5,5,5,5,5,5,5,5,5,5,5]]              # I love Deep Learning
    bleu_2 = calculate_bleu_score(preds_2, tgts_2, tokenizer)

    # 案例3：完全不匹配（BLEU应很低，接近0）
    preds_3 = [[7, 8]]                    # Deep Learning
    tgts_3  = [[4, 5, 6]]                 # I love NLP
    bleu_3 = calculate_bleu_score(preds_3, tgts_3, tokenizer)

    # 案例4：包含特殊符号，应被过滤后再计算，等价于完全匹配
    preds_4 = [[4, 5, 6]]
    tgts_4  = [[4, 5, 6]]
    bleu_4 = calculate_bleu_score(preds_4, tgts_4, tokenizer)

    # 案例5：4词完全匹配（应更稳定地接近100）
    preds_5 = [[4, 5, 7, 8]]              # I love Deep Learning
    tgts_5  = [[4, 5, 7, 8]]
    bleu_5 = calculate_bleu_score(preds_5, tgts_5, tokenizer)

    print("BLEU验证示例：")
    print(f"1) 完全匹配:           {bleu_1:.2f}")
    print(f"2) 部分匹配:           {bleu_2:.2f}")
    print(f"3) 完全不匹配:         {bleu_3:.2f}")
    print(f"4) 含特殊符号的匹配:   {bleu_4:.2f}")
    print(f"5) 4词完全匹配:         {bleu_5:.2f}")


if __name__ == "__main__":
    main()


