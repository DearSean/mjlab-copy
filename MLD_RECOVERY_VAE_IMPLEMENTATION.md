# MLD 恢复动作潜空间下一阶段实施说明

## 1. 当前基础

当前数据模块已经完成以下工作：

- 77 个 LaFAN BVH 均可解析为右手系、Z-up 的规范人体运动。
- 191 个恢复候选已全部人工审核，91 个接受、100 个拒绝。
- 接受片段已编译为 20 Hz、85 维的机器人无关运动学序列。
- 85D 表示保留关键点轨迹、速度、朝向、连续离地高度、mask 和恢复进度，
  不包含从 BVH 阈值推断的二值接触标签。
- 数据按 recording 分组，当前恢复片段分布为 Train 62、Validation 12、
  Test 17，没有同一同步录制跨集合泄漏。
- 完整 LaFAN 和恢复片段均可通过以下命令确定性重建：

  ```bash
  uv run python -m mjlab.tasks.velocity.scripts.build_recovery_dataset \
    --dataset-root /path/to/full/lafan1
  ```

本机生成的数据位于 `artifacts/recovery/dataset/`，该目录不提交 Git；
`dataset_manifest.json` 保存输入、schema、划分和产物哈希。

## 2. 下一阶段目标

实现一个轻量 MLD 风格的 VAE，学习 LaFAN 人体运动的机器人无关潜空间，
随后在人工筛选的恢复片段上微调，使潜空间重点覆盖倒地、翻身、支撑、
站起以及直接进入行走/跑动的动作变化。

这个阶段只建立人体语义动作潜空间，不生成机器人关节角，不进行机器人
动作重定向，也不训练 PPO。原 `motion-latent-diffusion` 项目仅作为算法
参考；训练核心使用当前 mjlab 环境中的现代 PyTorch 重新实现，不直接依赖
旧版 PyTorch Lightning、CLIP、SMPL 或 HumanML3D 工程环境。

## 3. 数据事实与长度策略

当前人工片段在 20 Hz 下的实际长度为：

| 集合 | 最短 | 中位数 | 最长 |
|---|---:|---:|---:|
| Train | 16 帧 / 0.80 s | 40 帧 / 2.00 s | 74 帧 / 3.70 s |
| Validation | 11 帧 / 0.55 s | 23.5 帧 / 1.18 s | 35 帧 / 1.75 s |
| Test | 28 帧 / 1.40 s | 48 帧 / 2.40 s | 76 帧 / 3.80 s |

因此不能强制把恢复片段裁成 4--8 秒。恢复数据保留原始人工边界，并使用
padding mask 组成 batch；完整 LaFAN 预训练可随机采样 40--160 帧的较长
窗口。第一阶段生成和重建以约 0.5--4 秒为验收范围，4--8 秒作为后续扩展，
不在当前数据规模下作硬性承诺。

## 4. 实现模块

### 4.1 数据加载与窗口采样

新增 `recovery_prior/data.py`，提供：

- `RecoveryKinematicDataset`：以 `allow_pickle=False` 读取 train、validation、
  test NPZ 和 normalizer。
- `MotionWindowSampler`：依据 `motion_offsets` 采样完整 LaFAN 窗口，绝不
  跨越 clip 边界。
- `RecoverySequenceDataset`：依据 `recovery_offsets` 读取完整人工片段，
  返回长度、padding mask、初始姿态、终点类型、outcome 和人工事件索引。
- batch 内按最长序列 padding；padding 值在归一化空间中为 0，所有损失
  必须乘有效帧 mask。
- 归一化只能使用 `normalizer.npz` 中的 Train Mean/Std，Validation/Test
  不允许重新统计。
- 完整恢复定义为 `outcome=success` 且终点为 `stationary` 或
  `locomotion`；其他接受片段作为局部恢复数据，不施加完整终点损失。

编译器还需补充每个 clip/片段的 `nominal_height_m`，用于把归一化关键点
误差还原为毫米指标。

**验收标准**

- 随机抽样 100,000 个窗口无一次跨 clip。
- 同一 seed 的窗口索引、padding mask 和 batch 顺序完全一致。
- 91 个恢复片段均可加载，人工事件索引均位于对应有效区间或明确为 `-1`。
- 所有归一化张量有限，Validation/Test 不修改 normalizer。

### 4.2 轻量 MLD VAE

新增 `recovery_prior/vae.py`，默认结构为：

- 输入：`[B, T, 85]` 归一化运动和 `[B, T]` 有效帧 mask。
- 每帧线性投影到 256 维并加入位置编码。
- 4--6 层、4 heads、FFN 1024 的 Transformer encoder。
- MLD-1 风格单 latent token，输出 `mean/logvar`，latent 形状为
  `[B, 1, 256]`。
- Transformer decoder 接收 latent、目标长度和 padding mask，重建
  `[B, T, 85]`。
- mask 特征不作为主要连续重建目标；人体数据中固定为 1，后续机器人
  投影时作为输入有效性描述。

VAE 不使用文本、CLIP、SMPL、机器人名称或机器人关节顺序。

### 4.3 损失函数

总损失定义为：

```text
L = w_pose * L_pose
  + w_velocity * L_velocity
  + w_clearance * L_clearance
  + w_progress * L_progress
  + w_rotation * L_rotation
  + beta * L_KL
```

- `L_pose`：root 高度和 9 个语义点相对位置的 masked SmoothL1。
- `L_velocity`：root、关键点线速度和 root 角速度的 masked SmoothL1。
- `L_clearance`：8 个连续离地高度的 masked SmoothL1，不使用二值接触损失。
- `L_progress`：恢复进度的 masked SmoothL1；完整 LaFAN 普通动作降低权重。
- `L_rotation`：躯干 6D 朝向的重建和正交约束。
- `L_KL`：标准高斯 KL，使用 warm-up 或 cyclical schedule，避免早期 posterior
  collapse。

局部恢复片段参与轨迹重建，但不要求到达站立终点；完整恢复片段才可增加
终点高度、朝向和终点类型辅助损失。

### 4.4 两阶段训练

第一阶段在完整 LaFAN Train split 上预训练 VAE：

- 随机窗口 40--160 帧。
- 首先建立可靠重建和连续潜空间，不突出恢复动作。
- 8 GB GPU 默认 batch 8--16，通过 gradient accumulation 达到有效 batch 64。
- 使用混合精度、gradient clipping 和可选 activation checkpointing。

第二阶段在恢复数据上微调：

- Train 中 62 个片段全部使用，完整和局部恢复分层采样。
- 混入部分普通 LaFAN 窗口，避免潜空间只保留少量起身样本并发生遗忘。
- `support_only/partial` 只约束已有轨迹；完整恢复才使用终点辅助目标。
- Validation 只用于超参数和 early stopping，Test 在配置冻结前不得查看。

第一版不加入扩散 denoiser。只有 VAE 重建、插值和潜空间覆盖通过验收后，
再实现 initial-state/terminal-conditioned latent diffusion。

### 4.5 训练入口、配置和检查点

新增建议入口：

```bash
uv run python -m mjlab.tasks.velocity.scripts.train_recovery_vae \
  --dataset-dir artifacts/recovery/dataset \
  --output-dir artifacts/recovery/models/vae
```

每个检查点必须包含：

- encoder、decoder、optimizer 和 scheduler 状态。
- 85D schema 版本与 feature schema SHA-256。
- dataset manifest 和 normalizer SHA-256。
- 完整训练配置、Git commit、seed、epoch 和全局 step。
- Train/Validation 指标以及最佳模型选择依据。

resume 后 RNG、dataloader sampler 和优化器必须连续。

## 5. VAE 验收标准

- Validation/Test 重建 MPJPE 不超过 50 mm。
- root height MAE 不超过 20 mm。
- 连续离地高度 MAE 不超过人体标称高度的 2%。
- 6D 朝向投影后有效且无 NaN/Inf，重建骨架骨长漂移不超过 1%。
- 对 91 个恢复片段均可完成 encode/decode，padding 区域不计入损失。
- 仰卧、俯卧、左右侧卧和 `other` 均在训练潜空间中有样本覆盖。
- latent 插值的关键点轨迹连续，不出现明显瞬移、地面深度穿透或姿态爆炸。
- 固定 checkpoint、输入和 seed 的重建输出一致。
- 在约 8 GB GPU 上可训练，峰值显存目标不超过 7.5 GB。

## 6. 本阶段明确不做的内容

- 不训练扩散 denoiser 或在线 DDIM 采样器。
- 不把人体动作重定向成 RL_BOY/G1 qpos。
- 不让 Velocity actor 跟踪人体参考帧。
- 不使用人工或自动二值人体接触标签作为 VAE 监督。
- 不将 MLD 放入 PPO 在线推理。
- 不开始 SMP/PPO 联合训练；SMP 要在 VAE 和后续条件扩散稳定后实现。

## 7. 后续接口

VAE 验收后，冻结 encoder/decoder 并实现条件 latent diffusion：条件包括初始
1--4 帧、初始姿态、目标终点类型和目标长度。生成的仍是 85D 人体语义轨迹。
随后借鉴 MimicKit SMP，在相同 85D 短窗口上训练 score prior，并把 score
成本作为 Velocity PPO 的阶段门控辅助奖励；实际机器人接触继续由 MuJoCo
物理和 RL 奖励学习。
