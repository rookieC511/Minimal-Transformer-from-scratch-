# Minimal-Transformer-from-scratch-
项目概要

这是一个用于教学与快速实验的最小可运行 Transformer 实现（PyTorch）。主要目标不是和大型库竞争性能，而是帮助你自己实现并理解 Transformer 的细节，同时包含若干实用改进以提高稳定性、可调试性和可复现性。

功能亮点：

从头实现 Multi-Head Attention、Scaled Dot-Product Attention、Positional Encoding、FFN、Encoder/Decoder 层等。

更鲁棒的 mask 构造（padding mask + 下三角 mask）。

Attention dropout / projection dropout。

可选 Pre-Norm / Post-Norm 架构。

支持返回注意力权重，便于可视化与调试。

训练脚本包含：scheduled sampling、Noam 学习率、EOS 权重、梯度裁剪、避免 PAD-only 时间步产生 NaN 的安全 loss 计算。

简单的 greedy decode，用于验证和演示。

文件结构
.
├─ multi_head_attention.py    # Transformer 实现（模型/层/编码/解码/pos enc）
├─ train.py                   # 训练脚本（数据，训练循环，验证，greedy decode）
├─ checkpoints/               # 最佳模型会被保存到这里（train.py 会自动创建）
└─ README.md                  # 说明（就是这个文件）

环境依赖

Python 3.8+

PyTorch（支持 CUDA）
示例（仅供参考）：

pip install torch torchvision

快速开始

在仓库根目录运行（默认配置见 train.py 顶部常量）：

python train.py


输出会显示训练/验证日志，最佳模型自动保存到 ./checkpoints/。

配置（train.py 中的常量）

脚本顶部定义了一组常量，直接修改即可控制训练行为：

VOCAB_SIZE, PAD_IDX, SOS_IDX, EOS_IDX：词汇表与特殊 token

MAX_LEN, BATCH_SIZE, D_MODEL, N_HEADS, D_FF, NUM_LAYERS, DROPOUT：模型/训练超参

NUM_EPOCHS, SEED：训练轮次与随机种子

DEBUG_NAN：如果为 True，会启用 PyTorch 的 anomaly 检测并在出现 NaN/Inf 时抛出更早的错误以便调试

USE_SCHEDULED_SAMPLING, SS_START_PROB, SS_END_PROB, SS_DECAY_EPOCHS：是否启用 scheduled sampling（按 epoch 线性衰减 teacher forcing 概率）

USE_NOAM, BASE_LR, WARMUP_STEPS：是否使用 Noam 学习率调度

EOS_WEIGHT：CrossEntropy 中对 EOS 类别的权重放大

CLIP_NORM：梯度裁剪阈值

OVERFIT_SMALL, OVERFIT_SAMPLES：调试用小数据过拟合开关（便于确认模型能学习）

如果想改成 CLI 参数，可以在 train.py 中加入 argparse（当前实现用常量，方便教学快速修改）。

关键设计点（解释与原因）

位置编码（PositionalEncoding）
使用经典的 sin / cos 周期函数，保持实现简单且能让模型感知相对/绝对位置信息。将位置编码注册为 buffer，不参与训练。

多头注意力（MultiHeadAttention）

明确分离线性映射（Q,K,V）与多头拆分/合并步骤，便于理解与调试。

在 scaled_dot_product_attention 中对打分值除以 sqrt(d_k) 来稳定梯度。

使用 masked_fill(~mask, -1e9) 屏蔽不关心的位置（pad 或 future），并在 softmax 后对 attention 做 dropout。

Mask 的构造

make_src_mask：对 PAD 位置做 mask（1 表示有效 token）；

make_trg_mask：同时包含 pad mask 与下三角 causal mask，保证解码器在生成时看不到未来位置。

mask 的形状与 attention score 的形状匹配（(B, 1, 1, L) / (1,1,L,L)），并保证 mask 在正确 device 上（使用输入的 .device）。

避免 PAD-only 时间步产生 NaN

在逐步生成（scheduled sampling）时，对每个时间步只对非 PAD 的 target 计算 loss；若某个时间步全部为 PAD，则跳过该时间步的损失累加，避免除 0 导致 NaN。

在 teacher-forcing（batch）路径，也使用 ignore_index=PAD_IDX 或显式 mask 以安全平均。

Pre-Norm vs Post-Norm

EncoderLayer / DecoderLayer 支持 pre_norm 参数。Pre-Norm 对于深层 Transformer 更稳定，但论文原版使用 Post-Norm。这里提供二者以便实验比较。

可视化与调试

MultiHeadAttention 的 forward 可返回 attention 权重，Encoder/Decoder 层会收集这些权重，Transformer.forward 返回一个字典{"encoder":..., "decoder":...}，方便做热图或对齐可视化。

训练稳定性

Attention dropout + proj dropout + grad clipping + optional Noam scheduler，共同提升训练稳定性；

DEBUG_NAN 开关帮助在开发阶段尽早捕获数值不稳定。

Early stopping / checkpoints

脚本会在验证 val_exact_strict 提升时保存最佳模型到 ./checkpoints/。建议再加一个 early-stopping（patience）逻辑以自动停止训练（避免资源浪费）。示例逻辑（伪码）：

best_val = -1.0
patience = 5
wait = 0

if strict_exact > best_val + min_delta:
    best_val = strict_exact
    wait = 0
    # 保存 checkpoint
else:
    wait += 1
    if wait >= patience:
        print("Early stopping")
        break


（如果你愿意我可以直接把这段合并进 train.py。）

评估与输出说明

train.py 默认会在每个 epoch 打印训练/验证指标（loss、token/seq 精度、EOS 比率等），并打印若干验证样例（truncated）以便人工检查输出内容。

greedy_decode：用于验证时的贪心解码。注意它返回的预测序列长度可能因 EOS 出现位置不同而变化（不包含初始 SOS）。如果你要把所有 batch 预测拼接为一个张量以做批量统计，请在拼接前做 padding 或按最大长度统一截断/补齐，否则 torch.cat 会因尺寸不一致报错（见常见错误）。

常见错误与调试建议

RuntimeError: Sizes of tensors must match except in dimension 0（torch.cat）

原因：greedy_decode 返回的每个样本预测长度不一致（EOS 出现在不同位置），在你试图 torch.cat(preds_all, dim=0) 时会报错。

解决办法：在收集预测时将它们 pad 到相同长度，或在 greedy_decode 固定输出长度为 max_len（例如总是返回 (B, max_len)，未生成的位置用 PAD 填充）。当前实现里可以修改 greedy_decode 使其返回固定宽度，或者在验证收集阶段用 torch.nn.utils.rnn.pad_sequence 做 padding。

NaN / Inf 出现

打开 DEBUG_NAN = True，此时脚本会早期抛出并打印导致 NaN 的位置（如 logits、梯度）。常见原因是除以 0，或在 softmax 前有极大/极小值（mask 未正确对齐）。检查 mask 的 dtype/shape 和 device。

训练/验证指标波动

如果 val_exact_strict 在小数据上波动较大，可能是数据分布过简单或随机采样差异导致。建议使用更大的验证集并固定 SEED 做复现。

可尝试的实验（建议）

关掉 USE_SCHEDULED_SAMPLING 对比：TF vs SS 对模型生成多样性的影响。

切换 pre_norm=True/False 测试深层网络的稳定性差异。

改变 MAX_LEN（更长）测试模型泛化到更长序列的能力。

增大 vocab / 更复杂任务（例如复制/逆序/算数）观察学习曲线差异。

可视化 attention 权重，对齐输入/输出查看模型是否学到合理的对齐策略。

可复现性

使用 SEED 固定随机种子；

若使用 GPU，可能还需设置：

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


将 DEBUG_NAN 打开可捕获数值异常。

许可与贡献

该仓库默认采用 MIT 许可证（若需其他许可证请修改）。欢迎提交 issue / PR 来改进实现、增加命令行参数、或加入更丰富的 demo。

联系 / 支持

如果你把仓库发到 GitHub 并希望我写 README 的特定细节（例如加上 badge、示例图、attention 可视化脚本、CI 配置等），把链接发给我或贴出你想要的内容，我可以继续帮你完善 README 或添加示例 notebook / demo。
