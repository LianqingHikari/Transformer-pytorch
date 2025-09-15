import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from model import Transformer,generate_mask
from data_processing import get_wmt_dataloaders,MAX_SEQ_LENGTH
from tqdm import tqdm
from util import calculate_bleu_score, plot_training_curves

class TrainingConfig:
    def __init__(self):
        self.batch_size = 16  # 单GPU批大小
        self.epochs = 30  # 总训练轮次
        self.gradient_accumulation = 8  # 梯度累积步数，总批大小=32*8=256
        self.resume_checkpoint = None  # 用于续训的检查点路径，如"checkpoints/epoch_5.pth"
        self.d_model = 512
        self.num_heads = 8
        self.num_encoder_layers = 6
        self.num_decoder_layers = 6
        self.d_ff = 2048
        self.dropout = 0.1

        # Adam参数
        self.learning_rate = 5e-4
        self.beta1 = 0.9
        self.beta2 = 0.98
        self.eps = 1e-9

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_dir = 'logs'
        self.checkpoint_dir = 'checkpoints'
        self.save_best_only = False  # 续训时建议保存所有epoch的检查点
        self.clip_grad_norm = 5.0
        self.label_smoothing = 0.1


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, smoothing=0.0, ignore_index=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing

    def forward(self, logits, target):
        logits = logits.reshape(-1, self.vocab_size)
        target = target.reshape(-1)

        one_hot = torch.zeros_like(logits).scatter(1, target.unsqueeze(1), 1)
        smoothed = one_hot * self.confidence + (1 - self.confidence) / self.vocab_size

        mask = (target != self.ignore_index).float()
        loss = -torch.sum(smoothed * torch.log_softmax(logits, dim=1), dim=1)
        loss = (loss * mask).sum() / mask.sum()

        return loss


## moved to util.calculate_bleu_score


def train_epoch(model, dataloader, loss_fn, optimizer, config, epoch):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        src = batch['src'].to(config.device)
        tgt = batch['tgt'].to(config.device)
        tgt_input = tgt[:-1, :]
        tgt_output = tgt[1:, :]

        output, _, _, _ = model(src, tgt_input)
        loss = loss_fn(output, tgt_output) / config.gradient_accumulation
        loss.backward()

        if (batch_idx + 1) % config.gradient_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        # 累计损失（注意：这里损失已经除以累积步数，所以直接累加）
        total_loss += loss.item() * config.gradient_accumulation

        if (batch_idx + 1) % (50 * config.gradient_accumulation) == 0:
            avg_loss = total_loss / (batch_idx + 1)
            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch + 1}/{config.epochs}, Batch {batch_idx + 1}/{len(dataloader)}, '
                  f'Loss: {avg_loss:.4f}, LR: {current_lr:.6f}, Time: {elapsed:.2f}s')

            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('train/loss', avg_loss, global_step)

    return total_loss / len(dataloader)


## decode逻辑已迁移至 model.Transformer.greedy_decode


def validate(model, dataloader, loss_fn, tgt_tokenizer, config, epoch):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    start_time = time.time()

    # 获取开始和结束符号的ID
    start_symbol = tgt_tokenizer.token_to_id("<s>")
    end_symbol = tgt_tokenizer.token_to_id("</s>")

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Validating Epoch {epoch + 1}", unit="batch")

        for batch_idx, batch in enumerate(pbar):
            src = batch['src'].to(config.device)
            tgt = batch['tgt'].to(config.device)
            tgt_input = tgt[:-1, :]
            tgt_output = tgt[1:, :]

            # 计算损失（仍使用教师强制，用于监控训练进度）
            output, _, _, _ = model(src, tgt_input)
            loss = loss_fn(output, tgt_output)
            total_loss += loss.item()

            # 自回归生成预测序列
            predictions = model.greedy_decode(
                src, MAX_SEQ_LENGTH, start_symbol, end_symbol, config.device
            )
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(tgt_output.transpose(0, 1).cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    bleu_score = calculate_bleu_score(all_predictions, all_targets, tgt_tokenizer)
    val_time = time.time() - start_time

    print(f'\n===== Epoch {epoch + 1} 验证结果 =====')
    print(f'验证损失: {avg_loss:.4f}, BLEU分数: {bleu_score:.2f}, 耗时: {val_time:.2f}s')
    print('=================================\n')

    return avg_loss, bleu_score


def main():
    config = TrainingConfig()
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    global writer
    writer = SummaryWriter(config.log_dir)

    print(f"使用设备: {config.device}")
    print(f"批大小: {config.batch_size}, 梯度累积: {config.gradient_accumulation}, "
          f"等效总批大小: {config.batch_size * config.gradient_accumulation}")

    # 加载数据
    dataloaders = get_wmt_dataloaders(
        data_dir="./dataset/WMT2014EngGer",
        batch_size=config.batch_size,
        use_predefined_val=True
    )
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    src_tokenizer = dataloaders['src_tokenizer']
    tgt_tokenizer = dataloaders['tgt_tokenizer']

    src_vocab_size = src_tokenizer.get_vocab_size()
    tgt_vocab_size = tgt_tokenizer.get_vocab_size()

    # 初始化模型、损失函数和优化器
    model = Transformer(
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        input_vocab_size=src_vocab_size,
        target_vocab_size=tgt_vocab_size,
        dropout=config.dropout
    ).to(config.device)

    loss_fn = LabelSmoothingLoss(
        vocab_size=tgt_vocab_size,
        smoothing=config.label_smoothing,
        ignore_index=0
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps
    )

    # 初始化训练状态
    start_epoch = 0
    best_bleu = 0.0
    train_losses = []
    val_losses = []
    val_bleus = []

    # 检查是否需要从检查点续训
    if config.resume_checkpoint and os.path.exists(config.resume_checkpoint):
        print(f"从检查点 {config.resume_checkpoint} 恢复训练...")
        checkpoint = torch.load(config.resume_checkpoint, map_location=config.device)

        # 恢复模型参数
        model.load_state_dict(checkpoint['model_state_dict'])
        # 恢复优化器状态（保证学习率和动量等参数连续）
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # 恢复训练进度
        start_epoch = checkpoint['epoch']  # 从下一个epoch开始
        best_bleu = checkpoint.get('best_bleu', best_bleu)
        # 恢复损失历史（用于绘图）
        if 'train_losses' in checkpoint:
            train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            val_losses = checkpoint['val_losses']
        if 'val_bleus' in checkpoint:
            val_bleus = checkpoint['val_bleus']

        print(f"成功恢复至 epoch {start_epoch}，将从 epoch {start_epoch + 1} 继续训练")

    # 开始训练（从start_epoch开始）
    print(f"\n开始训练，总轮次: {config.epochs}，起始轮次: {start_epoch + 1}")
    for epoch in range(start_epoch, config.epochs):
        # 训练当前epoch
        print(f'\n----- Epoch {epoch + 1}/{config.epochs} 训练开始 -----')
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, config, epoch)
        train_losses.append(train_loss)

        # 验证当前epoch
        print(f'\n----- Epoch {epoch + 1}/{config.epochs} 验证开始 -----')
        val_loss, val_bleu = validate(model, val_loader, loss_fn, tgt_tokenizer, config, epoch)
        val_losses.append(val_loss)
        val_bleus.append(val_bleu)

        # 记录日志
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/bleu', val_bleu, epoch)

        # 保存完整训练状态（包含所有需要恢复的信息）
        checkpoint_path = os.path.join(config.checkpoint_dir, f'epoch_{epoch + 1}.pth')
        torch.save({
            'epoch': epoch + 1,  # 当前已完成的epoch
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_bleu': best_bleu,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_bleus': val_bleus
        }, checkpoint_path)
        print(f'已保存 epoch {epoch + 1} 检查点至 {checkpoint_path}')

        # 更新最佳模型
        if val_bleu > best_bleu:
            best_bleu = val_bleu
            best_checkpoint_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_bleu': val_bleu
            }, best_checkpoint_path)
            print(f'更新最佳模型 (BLEU: {best_bleu:.2f}) 至 {best_checkpoint_path}')

        # 如果只保存最佳模型，删除当前epoch检查点
        if config.save_best_only and os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

    # 训练结束后测试
    print("\n加载最佳模型进行测试集评估...")
    best_checkpoint = torch.load(os.path.join(config.checkpoint_dir, 'best_model.pth'),
                                 map_location=config.device)
    model.load_state_dict(best_checkpoint['model_state_dict'])

    test_loss, test_bleu = validate(model, test_loader, loss_fn, tgt_tokenizer, config, epoch=-1)
    print(f'\n最终测试集结果: 损失: {test_loss:.4f}, BLEU分数: {test_bleu:.2f}')

    # 绘制训练曲线
    out_path = plot_training_curves(train_losses, val_losses, val_bleus, out_path='training_curves.png')
    print(f"训练曲线已保存至 {out_path}")

    writer.close()


if __name__ == "__main__":
    main()
