import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def calculate_bleu_score(predicted, target, tokenizer):
    """按token级别计算BLEU，基于id_to_token映射并过滤特殊符号。"""
    predictions = []
    references = []

    specials = {'<pad>', '<s>', '</s>'}

    for pred, tgt in zip(predicted, target):
        pred_ids = pred.tolist()
        tgt_ids = tgt.tolist()

        pred_tokens = [tokenizer.id_to_token(int(i)) for i in pred_ids]
        tgt_tokens = [tokenizer.id_to_token(int(i)) for i in tgt_ids]

        pred_tokens = [tok for tok in pred_tokens if tok not in specials]
        tgt_tokens = [tok for tok in tgt_tokens if tok not in specials]

        predictions.append(pred_tokens)
        references.append([tgt_tokens])

    smoothie = SmoothingFunction().method4
    bleu_score = corpus_bleu(references, predictions, smoothing_function=smoothie)
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


