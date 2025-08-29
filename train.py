# train.py
"""
训练脚本（已修复：避免在全部为 PAD 的时间步产生 NaN，并修复变长 greedy decode 导致 cat 失败）
说明与要点：
- scheduled sampling 分支的 loss 按所有有效 token（非 PAD）累加后再 / valid_count。
- greedy_decode 固定输出为 (batch, max_len)。
"""

import os
import random
import math
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from multi_head_attention import Transformer  # 确保在同一目录

# -----------------------
# 配置区
# -----------------------
VOCAB_SIZE = 13
PAD_IDX = 0
SOS_IDX = 11
EOS_IDX = 12

MAX_LEN = 12
BATCH_SIZE = 64
D_MODEL = 128
N_HEADS = 8
D_FF = 256
NUM_LAYERS = 3
DROPOUT = 0.1

NUM_EPOCHS = 30
SEED = 42

# 数值/稳定性开关
DEBUG_NAN = True
SAFE_LR = 5e-4
EOS_WEIGHT = 2.0
CLIP_NORM = 1.0

USE_SCHEDULED_SAMPLING = True
SS_START_PROB = 1.0
SS_END_PROB = 0.4
SS_DECAY_EPOCHS = 12

USE_NOAM = False
BASE_LR = 1.0
WARMUP_STEPS = 400

CHECKPOINT_DIR = "./checkpoints"
OVERFIT_SMALL = False
OVERFIT_SAMPLES = 32
OVERFIT_EPOCHS = 200

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# 可复现设置
# -----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------
# Toy 数据集
# -----------------------
class SortDataset(Dataset):
    def __init__(self, num_samples: int, max_len: int):
        super().__init__()
        self.samples = []
        for _ in range(num_samples):
            l = random.randint(2, max_len - 2)
            seq = random.sample(range(1, VOCAB_SIZE - 2), k=l)
            src = seq + [PAD_IDX] * (max_len - len(seq))
            tgt = sorted(seq)
            trg_in = [SOS_IDX] + tgt[:-1] + [PAD_IDX] * (max_len - len(tgt))
            trg_out = tgt + [EOS_IDX] + [PAD_IDX] * (max_len - len(tgt) - 1)
            assert len(src) == max_len and len(trg_in) == max_len and len(trg_out) == max_len
            self.samples.append((src, trg_in, trg_out))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, trg_in, trg_out = self.samples[idx]
        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(trg_in, dtype=torch.long),
            torch.tensor(trg_out, dtype=torch.long),
        )


# -----------------------
# Noam scheduler（保留）
# -----------------------
def get_noam_scheduler(optimizer, d_model, warmup=WARMUP_STEPS):
    def lr_lambda(step: int):
        step = max(1, step)
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup ** -1.5))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# -----------------------
# 工具函数
# -----------------------
def exact_match(preds: torch.Tensor, targets: torch.Tensor) -> float:
    # preds, targets: (N, T) 整齐对齐的张量（包含 PAD）
    # 仅在 targets 非 PAD 的位置进行比对；严格匹配要求所有这些位置相等
    mask = (targets != PAD_IDX)
    if mask.numel() == 0 or mask.sum().item() == 0:
        return 0.0
    # 对每序列判断在有效位置是否全部相等
    eq = (preds == targets)
    eq_masked = eq | (~mask)  # 在 PAD 位置视为 True
    seq_exact = eq_masked.all(dim=1).float()
    return seq_exact.mean().item()


def truncate_at_eos(seq, eos_token):
    """
    支持 1D Tensor/list 或 2D Tensor/list-of-lists：
    - 1D -> 返回 list （前缀到 eos 包含 eos）
    - 2D -> 返回 list of lists
    """
    # Tensor 的情况
    if isinstance(seq, torch.Tensor):
        if seq.dim() == 1:
            out = []
            for t in seq.tolist():
                out.append(int(t))
                if int(t) == eos_token:
                    break
            return out
        elif seq.dim() == 2:
            results = []
            for i in range(seq.size(0)):
                row = seq[i]
                results.append(truncate_at_eos(row, eos_token))
            return results
    else:
        # list 的情况
        if len(seq) == 0:
            return []
        if isinstance(seq[0], (list, tuple)):
            return [truncate_at_eos(s, eos_token) for s in seq]
        else:
            out = []
            for t in seq:
                out.append(int(t))
                if int(t) == eos_token:
                    break
            return out


def greedy_decode(model, src: torch.Tensor, max_len: int, sos_token: int, eos_token: int, pad_token: int, device):
    """
    对你的 Transformer 实现的 greedy 解码器，保证返回 (batch, max_len) 的 tensor（不包含初始 SOS）。
    - 若提前生成 EOS，之后位置填 pad_token
    - 返回类型：torch.LongTensor 在指定 device 上
    """
    model.eval()
    with torch.no_grad():
        batch_size = src.size(0)
        src = src.to(device)
        # encoder side
        # 使用 model 中的组件直接计算（兼容原有 Transformer）
        src_emb = model.pos_enc(model.tok_embed(src) * math.sqrt(model.d_model))
        src_mask = model.make_src_mask(src)
        enc_output, _ = model.encoder(src_emb, src_mask=src_mask)

        ys = torch.full((batch_size, 1), sos_token, dtype=torch.long, device=device)  # 包含初始 sos
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        outputs = torch.full((batch_size, max_len), pad_token, dtype=torch.long, device=device)

        for i in range(max_len):
            # decode with current ys
            trg_emb = model.pos_enc(model.tok_embed(ys) * math.sqrt(model.d_model))
            trg_mask = model.make_trg_mask(ys)
            dec_output, _ = model.decoder(trg_emb, enc_output, trg_mask=trg_mask, src_mask=src_mask)
            logits = model.fc_out(dec_output)  # (B, seq_len, V)
            next_logits = logits[:, -1, :]     # (B, V)
            next_tok = next_logits.argmax(dim=-1)  # (B,)
            outputs[:, i] = next_tok
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            finished |= (next_tok == eos_token)
            if finished.all():
                break
        return outputs.cpu()


# -----------------------
# 主训练流程（含 mask-safe loss 计算）
# -----------------------
def train():
    if DEBUG_NAN:
        torch.autograd.set_detect_anomaly(True)

    set_seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = DEVICE
    print("使用设备:", device)
    if USE_SCHEDULED_SAMPLING:
        print(f"Scheduled Sampling ON: start={SS_START_PROB}, end={SS_END_PROB}, decay={SS_DECAY_EPOCHS}")

    # 数据
    if OVERFIT_SMALL:
        train_ds = SortDataset(num_samples=OVERFIT_SAMPLES, max_len=MAX_LEN)
        val_ds = SortDataset(num_samples=OVERFIT_SAMPLES, max_len=MAX_LEN)
        num_epochs = OVERFIT_EPOCHS
    else:
        train_ds = SortDataset(num_samples=2000, max_len=MAX_LEN)
        val_ds = SortDataset(num_samples=200, max_len=MAX_LEN)
        num_epochs = NUM_EPOCHS

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # sanity print
    for src_dbg, trg_in_dbg, trg_out_dbg in train_loader:
        print("DEBUG Shapes:", src_dbg.shape, trg_in_dbg.shape, trg_out_dbg.shape)
        print("示例 src[0]:", src_dbg[0].tolist())
        print("示例 trg_in[0]:", trg_in_dbg[0].tolist())
        print("示例 trg_out[0]:", trg_out_dbg[0].tolist())
        break

    # 模型
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_LEN,
        dropout=DROPOUT,
        pad_idx=PAD_IDX,
        pre_norm=False,
    ).to(device)

    # loss weight for EOS (注意放在 device 上或在使用时移动)
    weight = torch.ones(VOCAB_SIZE, device=device)
    weight[EOS_IDX] = EOS_WEIGHT
    # base criterion retained for potential use
    base_criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, weight=weight, reduction='mean')

    if USE_NOAM:
        optimizer = optim.Adam(model.parameters(), lr=BASE_LR)
        scheduler = get_noam_scheduler(optimizer, D_MODEL, warmup=WARMUP_STEPS)
    else:
        optimizer = optim.Adam(model.parameters(), lr=SAFE_LR)
        scheduler = None

    best_val_exact = 0.0

    # scheduled sampling schedule
    if USE_SCHEDULED_SAMPLING:
        ss_start = SS_START_PROB
        ss_end = SS_END_PROB
        ss_epochs = SS_DECAY_EPOCHS
    else:
        ss_start = 1.0
        ss_end = 1.0
        ss_epochs = 1

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        train_token_correct = 0
        train_token_total = 0
        train_seq_exact_correct = 0
        train_seq_total = 0

        if USE_SCHEDULED_SAMPLING:
            frac = min(1.0, (epoch - 1) / max(1, ss_epochs - 1))
            tf_prob = ss_start + (ss_end - ss_start) * frac
        else:
            tf_prob = 1.0

        train_eos_pred_count = 0
        train_eos_target_count = 0

        for step, (src, trg_in, trg_out) in enumerate(train_loader, start=1):
            src = src.to(device)
            trg_in = trg_in.to(device)
            trg_out = trg_out.to(device)

            optimizer.zero_grad()

            if USE_SCHEDULED_SAMPLING:
                # sequence-level training with scheduled sampling
                batch_size = src.size(0)
                current = torch.full((batch_size, 1), SOS_IDX, dtype=torch.long, device=device)
                # accumulate per-token loss (sum) and count of valid tokens
                loss_sum = torch.tensor(0.0, device=device)
                valid_token_count = 0
                preds_for_batch = torch.full((batch_size, MAX_LEN), PAD_IDX, dtype=torch.long, device=device)

                for t in range(MAX_LEN):
                    logits, _ = model(src, current)  # (B, seq_len, V)
                    if DEBUG_NAN:
                        if torch.isnan(logits).any() or torch.isinf(logits).any():
                            print(f"NaN/Inf in logits at epoch {epoch} step {step} t={t}")
                            raise RuntimeError("NaN/Inf in logits")

                    next_logits = logits[:, -1, :]  # (batch, vocab)
                    targets_t = trg_out[:, t]       # (batch,)

                    # per-token losses (reduction='none'), ignore_index so PAD positions get 0 but we'll mask explicitly
                    losses_t = F.cross_entropy(next_logits, targets_t, weight=weight.to(device), ignore_index=PAD_IDX, reduction='none')  # (batch,)

                    mask = (targets_t != PAD_IDX)
                    if mask.any():
                        valid_losses = losses_t[mask]
                        loss_sum = loss_sum + valid_losses.sum()
                        valid_token_count += valid_losses.numel()

                    sample_pred = next_logits.argmax(dim=-1)
                    preds_for_batch[:, t] = sample_pred

                    use_teacher = (torch.rand(batch_size, device=device) < tf_prob)
                    next_input = torch.where(use_teacher, targets_t, sample_pred)
                    current = torch.cat([current, next_input.unsqueeze(1)], dim=1)

                    train_eos_pred_count += (sample_pred == EOS_IDX).sum().item()
                    train_eos_target_count += (targets_t == EOS_IDX).sum().item()

                # If no valid tokens in this batch
                if valid_token_count == 0:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
                else:
                    loss = loss_sum / valid_token_count

                if DEBUG_NAN and (torch.isnan(loss) or torch.isinf(loss)):
                    print(f"NaN/Inf in aggregated loss at epoch {epoch} step {step}: loss={loss}")
                    raise RuntimeError("NaN/Inf in aggregated loss")

                loss.backward()
                if DEBUG_NAN:
                    for name, p in model.named_parameters():
                        if p.grad is not None:
                            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                                print(f"NaN/Inf in grad for {name} at epoch {epoch} step {step}")
                                raise RuntimeError("NaN/Inf in gradients")
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                total_loss += float(loss.detach().cpu().item())

                with torch.no_grad():
                    # Only count non-PAD tokens for token accuracy
                    mask_all = (trg_out != PAD_IDX)
                    if mask_all.any():
                        train_token_correct += ((preds_for_batch == trg_out) & mask_all).sum().item()
                        train_token_total += mask_all.sum().item()

                    # sequence exact: only count sequences that have any valid token
                    seq_mask = mask_all.any(dim=1)
                    if seq_mask.any():
                        # for masked positions consider them equal by default
                        eq_masked = (preds_for_batch == trg_out) | (~mask_all)
                        seq_exact = eq_masked.all(dim=1)
                        train_seq_exact_correct += seq_exact[seq_mask].sum().item()
                        train_seq_total += seq_mask.sum().item()

            else:
                # batch teacher forcing: compute logits for whole sequence
                logits, _ = model(src, trg_in)  # (batch, trg_len_pred, vocab)
                trg_len_pred = logits.size(1)
                trg_target_aligned = trg_out[:, :trg_len_pred]  # (batch, trg_len_pred)

                # compute per-position losses with reduction='none', then mask and average safely
                logits_flat = logits.reshape(-1, logits.size(-1))  # (B*T, V)
                targets_flat = trg_target_aligned.reshape(-1)      # (B*T,)

                losses_flat = F.cross_entropy(logits_flat, targets_flat, weight=weight.to(device), ignore_index=PAD_IDX, reduction='none')  # (B*T,)
                # create mask for non-PAD targets
                mask_flat = (targets_flat != PAD_IDX)
                if mask_flat.sum().item() == 0:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
                else:
                    loss = losses_flat[mask_flat].mean()

                if DEBUG_NAN and (torch.isnan(loss) or torch.isinf(loss)):
                    print(f"NaN/Inf in teacher-forcing loss at epoch {epoch} step {step}: loss={loss}")
                    raise RuntimeError("NaN/Inf in loss (teacher forcing)")

                loss.backward()

                if DEBUG_NAN:
                    for name, p in model.named_parameters():
                        if p.grad is not None:
                            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                                print(f"NaN/Inf in grad for {name} at epoch {epoch} step {step}")
                                raise RuntimeError("NaN/Inf in grad (teacher forcing)")

                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                total_loss += float(loss.detach().cpu().item())

                with torch.no_grad():
                    pred_tokens = logits.argmax(dim=-1)
                    mask_all = (trg_target_aligned != PAD_IDX)
                    if mask_all.any():
                        train_token_correct += ((pred_tokens == trg_target_aligned) & mask_all).sum().item()
                        train_token_total += mask_all.sum().item()
                    # sequence exact
                    seq_mask = mask_all.any(dim=1)
                    if seq_mask.any():
                        eq_masked = (pred_tokens == trg_target_aligned) | (~mask_all)
                        seq_exact = eq_masked.all(dim=1)
                        train_seq_exact_correct += seq_exact[seq_mask].sum().item()
                        train_seq_total += seq_mask.sum().item()

                    train_eos_pred_count += (pred_tokens == EOS_IDX).sum().item()
                    train_eos_target_count += (trg_target_aligned == EOS_IDX).sum().item()

            if step % 50 == 0:
                avg_loss = total_loss / step
                tr_tok_acc = train_token_correct / max(1, train_token_total)
                tr_seq_acc = train_seq_exact_correct / max(1, train_seq_total)
                eos_rate = train_eos_pred_count / max(1, train_token_total)
                cur_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch} Step {step}/{len(train_loader)} avg_loss={avg_loss:.6f} "
                      f"train_tok_acc={tr_tok_acc:.4f} train_seq_acc={tr_seq_acc:.4f} "
                      f"tf_prob={tf_prob:.3f} eos_rate={eos_rate:.4f} lr={cur_lr:.6f}")

        avg_train_loss = total_loss / len(train_loader)
        train_tok_acc_epoch = train_token_correct / max(1, train_token_total)
        train_seq_acc_epoch = train_seq_exact_correct / max(1, train_seq_total)
        train_eos_rate = train_eos_pred_count / max(1, train_token_total)

        # 验证
        model.eval()
        with torch.no_grad():
            preds_all = []
            tars_all = []
            token_acc_total = 0
            token_count = 0
            val_eos_pred_count = 0
            val_eos_target_count = 0

            examples_to_print = 6
            printed = 0

            for src, trg_in, trg_out in val_loader:
                src = src.to(device)
                trg_out = trg_out.to(device)

                pred_seq = greedy_decode(model, src, max_len=MAX_LEN, sos_token=SOS_IDX, eos_token=EOS_IDX, pad_token=PAD_IDX, device=device)
                # pred_seq: (B, MAX_LEN) on CPU (as implemented)

                preds_all.append(pred_seq)
                tars_all.append(trg_out.cpu())

                # count token acc only on non-PAD target positions
                mask_nonpad = (trg_out.cpu() != PAD_IDX)
                token_acc_total += ((pred_seq == trg_out.cpu()) & mask_nonpad).sum().item()
                token_count += mask_nonpad.sum().item()

                val_eos_pred_count += (pred_seq == EOS_IDX).sum().item()
                val_eos_target_count += (trg_out.cpu() == EOS_IDX).sum().item()

                # 打印示例（truncate）
                if printed < examples_to_print:
                    p_list = truncate_at_eos(pred_seq, eos_token=EOS_IDX)
                    t_list = truncate_at_eos(trg_out.cpu(), eos_token=EOS_IDX)
                    for i in range(min(len(p_list), len(t_list))):
                        print(f"VAL EXAMPLE idx={printed}: SRC={src[i].cpu().tolist()}")
                        print(f"  TARGET (truncated) = {t_list[i]}")
                        print(f"  PRED   (truncated) = {p_list[i]}")
                        printed += 1
                        if printed >= examples_to_print:
                            break

            preds_all = torch.cat(preds_all, dim=0)
            tars_all = torch.cat(tars_all, dim=0)

            strict_exact = exact_match(preds_all, tars_all)
            pred_trunc = truncate_at_eos(preds_all, eos_token=EOS_IDX)
            tar_trunc = truncate_at_eos(tars_all, eos_token=EOS_IDX)
            eos_based_match = sum(1 for a, b in zip(pred_trunc, tar_trunc) if a == b) / max(1, len(pred_trunc))
            token_acc = token_acc_total / max(1, token_count)
            val_eos_rate = val_eos_pred_count / max(1, token_count)

        print(f"[Epoch {epoch}] train_loss={avg_train_loss:.6f} train_tok_acc={train_tok_acc_epoch:.4f} "
              f"train_seq_acc={train_seq_acc_epoch:.4f} train_eos_rate={train_eos_rate:.4f} "
              f"val_exact_strict={strict_exact:.4f} val_exact_eos={eos_based_match:.4f} token_acc={token_acc:.4f} "
              f"val_eos_rate={val_eos_rate:.4f}")

        if strict_exact > best_val_exact:
            best_val_exact = strict_exact
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_epoch{epoch}_strict{strict_exact:.4f}.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
                "val_exact_strict": strict_exact,
            }, ckpt_path)
            print(f"✅ 保存最优模型: {ckpt_path}")

    print("训练完成！Best strict val exact:", best_val_exact)


if __name__ == "__main__":
    train()
