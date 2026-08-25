# G1 Transformer–PPO–Diffusion 联合起身方案（离线预训练—仿真联合实施版）

## 1. 文档目的与结论

本文给出一套能够在当前 mjlab 项目中逐步落地的 Unitree G1 自主起身方案。
方案以 `lafan-g1-getup/` 中已经重定向到 G1 的高精度轨迹为主要动作先验，
以 G1 velocity 环境的默认屈膝站姿为最终目标。核心不是逐条模仿完整轨迹，
也不是要求人工精确切分每个起身子阶段，而是把所有轨迹采样为统一的短时
窗口，学习局部状态转移，再由 PPO 的站立回报和 MuJoCo 连续动力学把局部
规律组合成完整恢复；只有出现实证断链时才显式连接跨轨迹片段：

- **窗口编码器**：把任意有效短窗映射到共享恢复潜空间，同时保留窗口末端
  状态、运动方向、持续时间和支撑趋势；
- **Transformer**：根据最近若干窗口的 latent 建模时间上下文，不依赖硬编码
  的“翻身/手撑/膝撑”等阶段序号；
- **Diffusion**：低风险接法是冻结窗口 motion prior 作为 PPO 奖励；后置
  研究接法才根据当前上下文和站立目标生成多模态 successor latent；
- **可选的可达图与目标条件 critic**：当局部重规划仍会循环或断链时，
  再把同一轨迹的真实相邻窗口组织成图，并用仿真确认跨轨迹连接；
- **PPO**：在 MuJoCo 闭环动力学中把路标转化为 29-DoF 关节动作，并负责
  接触、平衡、力矩约束以及候选跨片段连接的物理确认；
- **G1 velocity 策略目标**：稳定起身后平滑进入站立或速度跟踪，而不是把
  “到达某条动作最后一帧”当作成功。

结论是：该方案在本项目中可实现，但不能把 Diffusion 当成缺失轨迹的真实
补帧器，也不能直接把不同文件中 latent 距离较近的窗口当成真实连续动作。
推荐先实现“**统一短窗潜空间 + 离线 Transformer + 冻结路标 + 目标条件
PPO + 仿真数据周期更新**”的主链。Diffusion 是确定性路标有效后的多模态
增强，可达图则是出现断链后再加入的后置增强，两者都不是 PPO 起步前置项。

具体顺序是：先用现有 63 条状态轨迹离线训练 Transformer；并行完成 G1
recovery 环境和无先验 PPO；然后冻结 Transformer 接入确定性 waypoint；
待 PPO 产生足够的成功/失败物理数据后，在 PPO 外部周期更新 Transformer。
Diffusion prior 可离线并行训练、在 PPO 基线后冻结接入；Diffusion planner
则最后才训练并启用。三者不在同一个 PPO minibatch 中端到端更新。

## 2. 当前项目与数据基础

### 2.1 G1 控制基础

当前项目已经具备：

- G1 29-DoF 模型、位置执行器和逐关节 action scale；
- G1 flat/rough velocity 环境；
- 关节位置、关节速度、重力投影、历史动作等本体观测；
- 足部接触传感器、姿态奖励、速度奖励和域随机化；
- ObservationManager 历史窗口；
- RSL-RL PPO 训练、checkpoint、play 和导出框架；
- RL_BOY recovery 中已经实现的姿态阶段估计、恢复辅助、成功保持和阶段
  门控逻辑，可复用算法思想，但不能照搬机器人参数。

G1 velocity 实际使用
`src/mjlab/asset_zoo/robots/unitree_g1/g1_constants.py` 中的
`KNEES_BENT_KEYFRAME` 作为默认状态，而不是 `HOME_KEYFRAME`。其主要值为：

| 项目 | 默认值 |
|---|---:|
| pelvis 高度 | 0.76 m |
| 左右 hip pitch | -0.312 rad |
| 左右 knee | 0.669 rad |
| 左右 ankle pitch | -0.363 rad |
| elbow | 0.6 rad |
| joint velocity | 0 |

这是一种便于平衡和接管 velocity 的屈膝站姿，适合作为起身后的目标中心。

### 2.2 `lafan-g1-getup` 数据事实

以当前 `lafan-g1-getup/index.json` 和 CSV 静态审计结果为准：

- 63 条 G1 CSV；按人工 `support_complete` 截去原 locomotion 跑动尾段后，
  当前共 3129 帧；
- 每帧 36 列：root position 3、root quaternion `xyzw` 4、G1 joint 29；
- 轨迹来自 6 个 source recording、5 个 subject；
- 30 条标记为 `success`，33 条标记为 `partial`；
- 原有 13 条 `locomotion` 已保留 `support_complete` 帧并删除之后的跑动，
  同步改标为 `support_only`；当前为 32 条 `stationary`、31 条
  `support_only`，不再包含 locomotion 尾段；
- 严格定义为 `success` 且终点为 `stationary` 的完整恢复共有 18 条；
- 62 条 `terminal_safe=true`，1 条为 false；
- 所有轨迹均有人工确认的 `support_start` 和 `support_complete`；
- 43/63 条轨迹从局部第 0 帧就已经进入 `support_start`，说明数据对可靠
  支撑阶段覆盖较好，但对“倒地静止到首次建立支撑”覆盖不足；
- 23 条轨迹恰好结束在 `support_complete`，没有之后的站立收敛尾段；
- 裁剪前的完整目录备份为根目录
  `lafan-g1-getup_before_loco_trim_20260825.tar.gz`；
- 当前帧率未写入 manifest。已有转换脚本默认输入 30 Hz，但正式编译前必须
  从数据来源确认，不能把默认值当作数据事实。

这些数据的正确定位不是“63 条完整起身示范”，而是：

1. 18 条完整恢复示范；
2. 大量高质量局部支撑与姿态转换示范；
3. 若干右删失轨迹：它们已有部分是有效正样本，但没有观测到最终站立；
4. 不足以独立监督所有倒地初态到默认站姿的完整动力学连接。

### 2.3 现有配置中阻碍起身的部分

当前 velocity 通用配置包含 `fell_over` 终止：机器人倾角超过约 70° 就结束
episode。恢复训练必须移除它，否则策略没有机会从地面起身。

当前 G1 完整碰撞配置中，脚部碰撞为 `condim=3`，其他碰撞几何主要为
`condim=1`。手、前臂、膝/小腿需要在起身中提供切向摩擦，因此必须为
recovery 任务建立专用碰撞配置并进行物理验证。

## 3. 任务定义

### 3.1 最终目标不是单帧，而是稳定站立吸引域

最终目标以 `KNEES_BENT_KEYFRAME` 为中心，但定义成容许小范围扰动的状态
分布 `G_stand`：

```text
G_stand = {
  q       接近 default_joint_pos，
  pelvis  接近默认高度，
  torso   基本竖直，
  dq      足够小，
  v_base  足够小，
  contact 双脚形成稳定支撑，
  unsafe  无持续非预期碰撞、无明显滑移和超限
}
```

推荐第一版 flat terrain 成功判据：

- pelvis 相对地面高度不低于默认高度的 85%；
- torso uprightness 不低于 0.90；
- base 线速度和角速度低于配置阈值；
- 双脚接触稳定，手、膝、躯干不再持续承重；
- 关节位置在软限位内，且总体接近默认屈膝站姿；
- 上述条件连续保持 0.5 s。

具体数值应通过初始 PPO 基线校准，不能用单个关节角误差替代动力学稳定性。

### 3.2 阶段标签只作为辅助信息

核心模型不要求把每条轨迹人工精确切分为固定的起身阶段，也不按阶段分别
训练网络。训练样本从全部轨迹的有效时间范围随机抽取，只要保留
`clip_id`、时间顺序、窗口长度和边界即可。

为了奖励门控、课程、评测和候选连接过滤，环境内部仍可连续估计：

1. **fallen**：高度低、躯干不竖直，尚未建立有效支撑；
2. **support/recovering**：手、肘、膝、脚等正在建立和转换支撑；
3. **upright-unstable**：已经接近站高，但仍有较大运动或非足部承重；
4. **ready/velocity**：稳定站立，逐步恢复 velocity 命令。

这些值优先从 MuJoCo 状态连续计算，可以复用 `PosturePhaseEstimator` 的
高度、uprightness、motion stability 和 walk gate 思路，并增加 G1 支撑
接触状态。人工 `support_start/support_complete` 只用于：

- 辅助分类/回归损失；
- 采样权重和数据审计；
- RSI reset 范围；
- 评测关键事件误差；
- 跨轨迹候选边的保守过滤。

它们不是 Actor 的必需观测，也不是局部 successor 训练的硬分段条件。即使
阶段标签有几帧偏差，只要窗口内部运动和真实邻接正确，主要训练目标仍成立。

### 3.3 控制频率与时间定义

当前 velocity 环境通常以 50 Hz policy 频率运行。原 CSV 很可能为 30 Hz，
所以所有窗口、保持时间和 Diffusion horizon 必须用秒定义，再由编译器映射
到目标帧率。禁止直接把“10 帧”在 30 Hz 数据和 50 Hz 仿真之间等同使用。

建议：

- policy 频率：50 Hz；
- Transformer 历史：0.4–0.8 s；
- Diffusion 短期路标：0.2–0.4 s；
- 每 0.1–0.2 s 重规划一次，可保留上一规划并采用 receding horizon；
- PPO rollout 初始使用 48–96 steps/env，使 rollout 覆盖 0.96–1.92 s；
- 成功保持：0.5 s。

### 3.4 统一短窗与局部 successor 定义

将每条轨迹切成可重叠短窗，而不是把整条轨迹压成一个 latent：

```text
W_i,t      = x_i[t-L+1:t]
W_i,t+K    = x_i[t+K-L+1:t+K]
z_i,t      = E_ref(W_i,t, valid_mask, duration)
```

其中 `i` 是 clip，`L` 是历史窗口长度，`K` 是 successor horizon。训练时可
混合 8、10、15、20 帧等长度，但必须输入 `valid_mask`、实际持续时间和
预测 horizon。第一版也可以固定为 10 帧或 20 帧，以减少变量长度带来的
调试成本。

latent 必须保留两类信息：

- `z_state`：窗口末端姿态和 root 状态；
- `z_motion`：窗口内速度、运动方向，并通过辅助头预测接触趋势。

可使用一个拼接 latent，也可使用两个 token。不能只重建平均姿态，否则
“正在撑起”和“正在跌落”的相似姿势会被错误映射到同一点。

首版 64D deployable input 不含精确接触。离线接触只能由 FK 高度/速度近似成
辅助标签，在线精确接触只给 critic、奖励和评测；若以后确认实机有可靠足部
接触估计，再发布新的 feature schema 并同步修改维度与 parity test。

同一 clip 内不越界的 `W_i,t → W_i,t+K` 是唯一可以直接当作真值的 successor
边。不同 clip 的窗口无论多相似，都只能先成为候选连接，不能直接加入正样本。

## 4. 总体架构

```text
离线主链（先完成）
63 条 G1 状态轨迹 → 短窗/真实未来窗 → E_ref + T_ref_context + T_succ
                                                           │
                                                           ▼ 确定性 z_goal
                                      ↑
                         解析式 G_stand 长期条件

在线执行主链
G1 历史 → E_ref/T_ref_context/T_succ → z_goal → goal-conditioned PPO → MuJoCo
                                                    │
                                      成功、失败、超时窗口 replay
                                                    │
交替更新（PPO rollout 之间）                         ▼
offline anchor + elite sim success → 更新 T_ref_context/T_succ → 准入 → 新版本
全部 sim attempt → OOD/risk（以及后置 P_H/D_H）

后置增强
滚动窗口 reward ← frozen Diffusion prior（C6a，PPO baseline 后）
同一 z_goal 接口 ← Diffusion 多候选（C6b/C6c，确定性基线通过后）
              ← GAS/SoRB 式可达图（C7，观察到断链/循环后才加入）
```

部署最小模型是 `E_ref/T_ref_context/T_succ + Actor`。可选的
`E_actor_history` 只是 PPO 侧小型 adapter，不负责定义 latent 坐标。
若 Diffusion 的闭环增益足以覆盖延迟，再部署轻量 denoiser 和候选选择器；
可达图只在实验证明需要 stitching 时部署。最终也可以把 planner 蒸馏进
Actor，避免在线运行 Diffusion 或图搜索。

### 4.1 必需组件、接入时机与方法出处

完整研究链路包含下表组件，但不是所有组件都在首版同时启用。标记为
“后置”的组件只有在前一版基线通过后才实现。

| 编号 | 组件 | 首版是否必需 | 产物/职责 | 主要参考 |
|---|---|---|---|---|
| C0 | G1 数据编译器与窗口 sampler | 必需 | 50 Hz 特征、mask、同 clip 真实 successor pair | 本项目数据 schema；Trajectory Transformer 的序列样本组织 |
| C1 | 解析式站立目标 sampler | 必需 | 围绕 `KNEES_BENT_KEYFRAME` 采样终端目标，不训练单独站立模型 | HoST、HumanUP 的任务奖励与恢复课程思想 |
| C2 | 离线 Window Transformer | 必需 | `E_ref/T_ref_context/T_succ`、OOD/不确定度 | Trajectory Transformer、Online Decision Transformer |
| C3 | G1 Recovery PPO 环境 | 必需 | 接触闭环、起身奖励、随机 fallen reset、安全约束 | HoST、HumanUP、项目现有 RL_BOY recovery |
| C4 | Latent waypoint command/adapter | 必需 | 把 Transformer/Diffusion 的 `z_goal` 送入 Actor/Critic | GPC 的 token-conditioned controller；SPiRL 的离线 skill prior→下游 RL |
| C5 | Offline/Sim Replay 与版本化 trainer | 主链条件项 | S3 有增益后进行仿真分流和 Transformer 周期更新 | Online Decision Transformer 的离线→在线顺序；面向 PPO 的工程适配 |
| C6a | Frozen Diffusion motion prior | 推荐增强 | 用窗口 score/SDS 形成运动先验奖励，不生成 action | SMP 及 MimicKit 官方实现 |
| C6b | Latent Diffusion successor | 后置研究项 | 多候选短期 successor 与 receding-horizon 重规划 | SSD、Diffuser、Diffusion Policy |
| C6c | Candidate goal/risk critic | C6b 多候选时必需 | 估计 K 步可跟踪、安全和最终站立概率 | SSD goal-conditioned critic；PPO 仿真标注 |
| C7 | Stitching/Reachability | 后置可选 | 有限时域可达 critic、跨 clip 候选与图搜索 | GAS、SoRB、QRL；SSD 的 diffusion stitching |
| C8 | Velocity 接管 | 最终必需 | 从稳定起身平滑恢复 velocity command | 项目现有 posture/walk gate；HumanUP 的部署精炼思想 |
| C9 | 评测与模型准入 | 必需 | 固定初态、消融、安全指标、checkpoint promotion | 本项目工程实现 |

这里的“参考”表示复用已验证的方法结构，不代表把外部仓库直接复制进 mjlab。
外部实现的依赖、许可证、仿真器、机器人 DoF 和训练算法都要单独核对。例如
Online Decision Transformer 官方仓库以离线/在线 state-action-return 序列
为主，并不是 RSL-RL PPO 插件；常见 mixed-replay 方法又多为 off-policy。
当前 CSV 没有 action、旧策略 log-prob 或 return，因此离线窗口只能进入
Transformer、Diffusion 或 critic 的辅助训练，绝不能塞进 PPO surrogate
minibatch。

### 4.2 首版最小链路与完整链路

可运行 MVP 到 C4；用户提出的“仿真数据继续更新 Transformer”由条件项 C5
实现，只有 C4 A/B 证明有增益后才开启：

```text
C0 数据窗口 + C1 解析式站立目标
  ├─→ C2 离线 Transformer
  └─→ C3 Recovery PPO baseline
          └─→ C4 冻结 Transformer waypoint + PPO
                 └─→ S3 有增益时 C5 仿真数据更新 Transformer
```

在这条链路已经能提高起身成功率后，才增加：

```text
C6a frozen Diffusion prior reward（低风险）
  或 C6b Diffusion successor + 多候选时 C6c critic（研究路线）
  → 必要时 C7 跨片段 stitching/reachability

C4/C5 最佳策略 ──→ C8 velocity 接管（不依赖 C6/C7）
```

因此 Transformer 和 Diffusion 的接入顺序不是并列的：Transformer 先离线
预训练并以确定性 successor 形式进入 PPO。Diffusion 有两种互不混淆的接法：
C6a 可作为冻结的窗口自然度奖励低风险接入；C6b 才会替换/扩展确定性
successor head，必须等低层 executor 和候选 critic 可用。即使最终证明二者
都没有额外收益，前面的数据、Transformer 和 PPO 工作仍然有效。

## 5. 数据编译与监督规则

### 5.1 建立版本化 G1 recovery dataset

不能让训练代码直接遍历 CSV 并依赖隐式顺序。应新增确定性编译步骤，将
`lafan-g1-getup` 转成统一 NPZ 或 shard：

- 明确记录输入 FPS、输出 FPS=50、坐标系、单位和 quaternion 顺序；
- quaternion 从 CSV `xyzw` 转为项目内部使用的 `wxyz`；
- root XY 平移归一化到 clip 起点，并做 heading/yaw 归一化；
- quaternion 用 SLERP 重采样，位置和关节角使用时间一致的插值；
- 通过 G1 FK 计算关键 body 的位置、速度和离地高度；
- 记录每个 clip 的来源、subject、recording、原始帧范围和哈希；
- 输出 `valid_mask`、`support_mask`、`terminal_mask`、`terminal_safe`、
  `has_standing_terminal` 和事件索引；
- 输出 `clip_id`、窗口起止时间、持续时间和可验证的同 clip successor 索引；
- 检查 NaN/Inf、关节硬限位、速度尖峰、地面穿透、接触冲击和 quaternion
  连续性；
- 编译可重复，相同输入与配置必须得到相同产物哈希。

现有 `src/mjlab/scripts/csv_to_npz.py` 可以复用插值、SLERP、速度估计和 G1
FK，但要改成批量、本地、确定性编译器，不能继续依赖固定 `/tmp` 输出和
W&B 上传作为核心数据路径。

### 5.2 防止数据泄漏

数据有效多样性主要来自 6 个 source recording，而不是 63 条彼此独立的
轨迹。Train/Validation/Test 必须按 source recording 分组；同一 recording
切出的 clip、同一时段的相邻窗口不能跨集合。随机按窗口划分会严重高估
泛化能力。

### 5.3 不完整轨迹的损失规则

不完整轨迹采用右删失处理：

- 有效范围内所有局部转移都可参与 Transformer 和局部 Diffusion 训练；
- sampler 可以在所有 clip 间混合窗口组成 batch，不要求逐条完整输入；
- `support_start/support_complete` 只参与可选辅助任务、采样和评测，不用于
  硬切断主要 successor 训练；
- clip 边界以后不存在的帧不补零、不复制最后一帧、不插值到默认站姿；
- `partial` 或 `support_only` 不计算“必须到达站立”的 terminal loss；
- 只有 `success` 且 `stationary` 的 18 条完整轨迹计算真实终端连接损失；
- `terminal_safe=false` 的轨迹默认不作为正向控制目标，可用于失败/质量判别；
- 当前数据已删除 locomotion 尾段；后续 PPO 产生并经物理验证的动态接管
  轨迹可以作为进入 velocity 的正样本，不必强迫先完全静止。

### 5.4 真实未来窗优先，跨轨迹 stitching 后置

首版不需要先训练恢复图。统一潜空间只保留同一 clip 内、不越界的真实时间
边，用 receding-horizon Transformer + PPO 检验局部窗口能否自然串成起身。
只有出现以下实证问题时才启用 C7：长期循环在若干支撑姿态、partial 片段
利用率低，或跨轨迹切换明显能增加 held-out 起身成功率。

C7 应按 GAS、SoRB 和 QRL 的形式定义成与当前低层执行策略相关的有限时域
目标条件距离/可达 critic，而不是孤立的“姿态相似二分类器”：

```text
P_H(w, g | pi_goal) = P(pi_goal 在 H 步内安全到达目标窗 g | 当前窗 w)
D_H(w, g | pi_goal) = 预计安全到达步数或其保守上界
```

如外部接口需要二值 `R_link`，它只是 `P_H` 的阈值化结果。模型必须输入
姿态、base/joint 速度、角动量代理、支撑接触和 horizon；起身具有方向性，
所以 `D_H(a,b)` 不应强制等于 `D_H(b,a)`。

标签规则必须保守：

- 同一 clip 的未来窗是“已观测运动学可达”正样本，但还不是 G1 PPO 的
  动力学可控证明；
- 时间倒序只能在违反方向、时间或接触条件时作为规则负样本；
- 随机跨 clip 配对属于未标注候选，不能因“没有观察到”就标为负样本；
- 真正强负样本来自 PPO/MuJoCo 的超时、危险接触、不可达和执行失败；
- 真正确认边必须由当前版本 `pi_goal` 在物理仿真中多次成功执行。

当 executor 已可跟踪局部目标后，再按下列顺序建立图：

```text
同 clip 真实时间边 → 训练/校准 P_H 或 D_H
跨 clip 状态近邻  → 仅作 candidate edge
PPO/MuJoCo 成功    → confirmed edge
G_stand             → terminal node set
```

这不是某篇论文可原样复制的单一算法。GAS 提供“TDR→构图→目标条件低层策略
→图搜索”的直接主线，SoRB 提供 replay 图与 OOD 假捷径的警告，QRL 提供
方向性 quasimetric；将其用于 G1 接触恢复并增加物理确认，是本项目的工程
适配。若无图版本已经可靠起身，则不实现这一复杂度。

### 5.5 解析式站立目标，不前置训练站立模型

第一版不需要为了建立 `G_stand` 专门训练一个稳定站立模型。当前 G1 的
`JointPositionAction` 使用 default offset，零 action 本身就以
`KNEES_BENT_KEYFRAME` 为中心；可以直接构造：

```text
q_goal       = clip(q_default + epsilon_q, soft_joint_limits)
dq_goal      = small_noise around 0
root_up_goal = upright with small roll/pitch noise
height_goal  = 0.76 m + small_noise
v_goal       = small_noise around 0
```

最小实验只加关节噪声即可启动。但它只定义“希望接近的关节姿态”，不能单独
证明机器人已经稳定站立。推荐同时加入很小的 root roll/pitch、root/joint
速度扰动；每个采样状态经 `forward`/短时 settle 后过滤穿透、失衡和非双脚
支撑，训练后期再加入小推扰。以上都只是同一 PPO 的 reset 分布，不是另一个
站立模型。通过以下解析判据定义成功：

- pelvis 高度和 torso uprightness 达标；
- 双脚稳定接触；
- 非足部不再持续承重；
- base/joint 速度足够低；
- 关节接近默认屈膝姿态且没有超限；
- 连续保持 0.3–0.5 s。

用于 Transformer/Diffusion 的 `z_stand` 可由一个重复 10–20 帧的解析目标
窗口得到：`q_rel≈0`、joint/base angular velocity 接近零、projected gravity
直立。双脚支撑仍由环境成功判据检查，不伪装成 64D Transformer 输入。全局
XY 和绝对 yaw 应忽略或按当前机器人重新对齐。

当 PPO 已能站稳后，可以自动收集真实成功 episode 的最后 0.5–1 s，形成
`terminal_replay`，用于把解析目标扩充为更真实、仍以默认姿态为中心的站立
分布。这个 replay 是训练副产物，不是前置条件，也不需要单独训练站立
policy。它仍然不能提供
“支撑末端到站立”的缺失中间监督。

项目中可直接复用 `reset_joints_by_offset` 与 `reset_root_state_uniform` 构造
上述分布，`variable_posture` 和 `upright` 可作为终端奖励基础。现有
`PosturePhaseEstimator.is_ready` 只阈值化 `height * uprightness`，没有把已
计算的 motion stability 或双脚接触纳入布尔条件，不能直接当 recovery 成功
判据；应新增本节的严格保持 term，而不是训练另一个模型。

### 5.6 仿真成功轨迹回灌

PPO 发现的连接轨迹只有通过以下过滤后才能加入 replay：

- 完成稳定站立或安全进入 velocity；
- 无硬限位违规和持续地面穿透；
- 峰值力矩、速度、功率和接触力在设定范围；
- 无明显手/膝滑移和高频 action 抖动；
- 初始状态和恢复路线具有足够多样性。

还要把 `B_sim_success` 再分成普通成功和 `elite_success`。按仰卧/俯卧/侧卧、
reset 难度和 domain-randomization 桶分别比较，使用“成功保持、起身时间、
峰值冲击、力矩/功率、action rate”的 Pareto 前沿或桶内 top-quality 分位数。
只有 `elite_success` 的真实局部未来窗默认更新无条件 desired `T_succ/D_plan`。
普通成功可用于覆盖率、critic、OOD，或在显式加入 return/quality condition 后
训练；否则简单自蒸馏当前策略会把低质量习惯固化进 prior。

replay 必须标记来源为 `retargeted_demo` 或 `sim_validated`，评测时分别报告，
避免把模型自己生成的数据当成人工真值。

回灌不只保存整条成功 episode，还要保存：

- 执行前窗口、目标窗口和实际到达窗口；
- planner/可选 `P_H/D_H` 预测、实际成功/失败和所用时间；
- 是否形成新的跨 clip 确认边；
- 接触、力矩、功率、滑移和冲击统计；
- 失败原因和候选来源。

成功样本可更新 desired-successor Transformer 和后续 Diffusion；失败、超时
和危险样本更新 OOD/risk，并在 C7 启用后校准 `P_H/D_H`。失败状态
不能作为“下一段应该这样走”的正监督，也不能被简单丢弃。

## 6. Transformer 表征

### 6.1 Actor 可部署观测

离线 CSV 没有 action，也没有可直接观测的真实接触力。为保证离线/在线特征
一致，首版 Transformer 每个 50 Hz frame 建议只使用：

- projected gravity / torso orientation 的可部署表示；
- base angular velocity；
- 29 个 joint position 相对默认姿态；
- 29 个 joint velocity；

这恰好形成 64D 的统一 proprio 特征。`last_action`、velocity command、
planner age 和 `z_goal` 可以作为 Actor head 的额外输入，但不能混入离线
Transformer 后再假装 CSV 中存在。可可靠估计的足部接触或 base linear
velocity 可在 feature schema v2 中加入，并需要离线/仿真 parity 测试。

绝对 pelvis 高度、精确全身接触力和仿真真值 base linear velocity 只能在
实机确实具备相应估计器时进入 actor；否则只放入 privileged critic 或作为
离线辅助标签。不能让训练期 Transformer 依赖部署时不存在的输入。

### 6.2 结构建议

第一版可使用：

- 单窗口长度：固定 10 或 20 帧起步；验证后可随机使用 8–20 帧；
- 上下文长度：最近 2–4 个窗口或等价的 20–40 帧历史；
- model dim：128 或 256；
- 3–4 层 causal Transformer；
- 4 heads；
- `z_state/z_motion` 总 latent：64 或 128 维；
- 显式输入 window duration、successor horizon 和有效长度 mask；
- reset 时清空历史并使用有效长度 mask，禁止跨 episode 污染。

建议把“窗口编码”和“窗口序列上下文”区分开：

- `E_ref(W)`：双向编码一个已经完成的短窗口，得到固定结构 latent；
- `T_ref_context(z_t-n:t)`：因果编码最近的 latent token，输出 prior 条件
  `h_ref`；
- `E_actor_history(W_<=t)`：可选的 PPO 侧小型 history adapter；它不定义
  Diffusion target 或 latent reward 坐标。

在线路径不能使用当前时刻之后的帧。`z_state` 要能重建窗口末端状态，
`z_motion` 要能重建速度、方向，并通过辅助头区分接触趋势，从结构上防止
相似姿态的上升与下降运动被折叠到同一点。

### 6.3 四个职责分离并明确冻结规则

不建议一个不断被 PPO 更新的编码器同时承担以下三件事：

1. actor 的策略特征；
2. Diffusion 的训练坐标系；
3. latent 奖励的度量坐标系。

否则 reward 坐标会随 PPO 一起漂移。推荐把职责明确拆开：

- `E_ref`：定义窗口 latent 坐标；一个发布版本内完全冻结，供 target、距离、
  Diffusion 和 replay 使用；
- `T_ref_context`：根据历史 latent 产生 frozen-prior context；只在 S1 或 S4
  auxiliary step 更新，PPO rollout 中冻结；
- `T_succ`：根据 `h_ref`、horizon 和 `z_stand` 预测 K 步后的期望 endpoint
  latent；先离线训练，之后可用仿真 elite-success 周期更新；
- `E_actor_history`：可选 PPO 侧 adapter，输入 Actor/Critic，可以随 PPO 更新，
  但其输出不作为 replay key、Diffusion target 或 waypoint 距离。

S3 最稳妥配置是冻结 `E_ref/T_ref_context/T_succ`，只训练 Actor/Critic 和可选
adapter。S4 也优先只在 rollout 之间更新 `T_ref_context/T_succ`，保持
`E_ref v0` 永久冻结。如果后来必须更新 `E_ref`，要发布新 schema/model 版本
并离线重编码所有 replay、Diffusion target 和图节点；禁止在同一 rollout 内
让 latent 奖励坐标漂移。

### 6.4 离线预训练到底学什么

当前 CSV 只有 root/joint 状态序列，没有 G1 在 MuJoCo 中执行这些动作时的
action、torque、reward、return 或旧策略 log-prob。因此可以先训练
Transformer，但它学的是**状态窗口表征和未来窗口先验**，不是可直接输出
电机动作的 Decision Transformer policy。低层 action 必须由 PPO 在仿真中学。

可在所有安全有效窗口上训练：

- 同 clip 下一段 latent/状态预测；
- masked joint/body state reconstruction；
- support phase 分类或连续支撑强度回归；
- recovery progress 排序；
- time-to-support / time-to-ready 的 censored regression；
- 重叠窗口的一致性；
- 同一轨迹相邻窗口的时间对比一致性；
- 运动方向和速度重建，以及从 FK clearance 推导的接触趋势辅助预测；
- 可选的 Temporal Distance Representation：预测同 clip 两窗的时间距离，
  为后续 GAS 式构图提供初始化。

训练顺序建议为：先用重建、重叠一致性和真实 successor 学出稳定潜空间，
再训练 causal `T_ref_context/T_succ`。所有 future target 都必须来自同 clip 的
真实后续帧；partial/support-only 到文件末端即停止监督。辅助阶段标签只在
训练期使用，不应成为实机部署所需的人工输入。

### 6.5 到默认站姿的 critic 是后置项，不是 Transformer 前置项

`z_stand` 长期条件、解析式 progress reward 和有限 episode，已经足够启动
确定性 Transformer + PPO 基线。若闭环评测确认局部模型会循环或选择死路，
再增加：

```text
V_to_stand(z)
  = P(从当前窗口最终能够安全到达 G_stand)
```

稳定站立目标集合是值为 1 的终端，失败和超时 rollout 提供低值样本。对
仿真边和确认边进行目标传播：

```text
V(z_t) ≈ success_now + gamma * V(z_next)
```

18 条完整轨迹只能提供运动学锚点；`V_to_stand` 是否校准，最终要由当前 PPO
的 rollout 决定。尚无连接证据的 partial 片段只贡献局部转移，不被强行赋予
高终端价值。

### 6.6 Transformer 从离线到仿真的更新协议

用户提出的“先离线训练一轮，能引导起身后再用仿真数据更新”应作为默认
协议。这里的“一轮”不是固定 epoch 数，而是通过 held-out recording 和闭环
准入条件定义：

| 时期 | `E_ref` | `T_ref_context/T_succ` | PPO | 使用的数据 |
|---|---|---|---|---|
| T0 离线预训练 | 训练后冻结 v0 | 训练 v0 | 不参与 | 63 条真实有效短窗 |
| T1 PPO 基线 | 冻结 v0 | 不接入 | 训练 | 纯仿真 on-policy |
| T2 冻结引导 | 冻结 v0 | 冻结 v0 | 训练 goal-conditioned executor | RSI + fallen reset；先真值路标后预测路标 |
| T3 交替更新 | 冻结 v0 | PPO rollout 之间更新 | 每个 prior 版本内正常训练 | 固定离线 anchor + 筛选后的 sim replay |
| T4 可选重表征 | 发布 v1 后冻结 | 随 v1 重训 | 从兼容 checkpoint 继续 | 全量重编码并通过回归评测 |

T2 使用 teacher-forcing 课程：RSI 环境先跟踪 demonstration 的真实下一窗，
然后逐步混入 `T_succ` 自己预测的路标，最后扩大到随机 fallen reset。对明显
超出离线分布的倒地状态，OOD gate 应降低/关闭 latent waypoint reward，退回
解析式高度、upright、支撑和 `G_stand` 奖励，避免模型自信地给出伪路标。

T3 不在每个 PPO minibatch 里更新 Transformer，而采用宏周期：

```text
冻结 prior vN → 收集若干 PPO on-policy rollout → 写辅助 replay
→ 暂停/结束本轮 rollout → 混合训练 T_ref_context/T_succ candidate
→ 固定评测与安全准入 → 通过则发布 prior vN+1，否则保留 vN
```

辅助 replay 至少分三类：

- `B_offline_anchor`：原始真实窗口，始终保留以防灾难性遗忘；
- `B_sim_success`：通过严格成功判据的 transition；其中只有按初态桶筛出的
  `elite_success` 默认更新无条件 desired successor prior；
- `B_sim_attempt`：全部成功、失败、超时和危险 transition，只用于 OOD、risk
  以及后置 `P_H/D_H`；失败动作不能训练成期望 successor。

每个 batch 可从 offline-heavy 开始，再逐步增加 sim 比例；比例由固定
validation 决定，不设成方法定理。新版本至少要同时满足：离线 held-out
future-window 误差不显著退化、仿真 waypoint 可跟踪率提高、固定 fallen
初态成功率不下降、安全成本不过阈值。PPO surrogate 始终只使用当前策略的
on-policy rollout；旧 replay 只训练这些辅助模型。

## 7. Diffusion 的两种接法与明确时机

“Diffusion 什么时候接入”取决于它承担什么功能。这里必须把两个不同模型
分开，不能把同一个 diffusion loss 同时解释成奖励和 planner。

### 7.1 C6a：冻结的窗口 motion prior（推荐先做）

参考 SMP 的公开做法，可用全部有效局部窗口离线训练小型 motion diffusion，
再冻结为 PPO 的 score/SDS 运动先验奖励：

```text
D_prior: noisy completed state-window → denoising score
r_prior: 当前策略刚执行出的窗口在该 motion prior 下的相容度
r_total: r_getup_task + lambda_prior * r_prior - safety_costs
```

它可以和 T0 Transformer 离线预训练并行完成，但要等 PPO-only 环境基线能
起身后才以小权重启用。它不预测“下一片段”，不输出 action，也不需要知道
每个片段的阶段；因此所有高质量局部支撑窗口都能贡献自然运动约束。SMP
官方 MimicKit 就是“先训练 prior，后冻结作为新 policy 的奖励”。AMP 是较早
的对抗式 motion-prior PPO 基线，可作为对照，但需要随 policy 训练判别器。

这一路线最接近已有论文和开源代码，也最适合回答“局部片段怎样共同影响
完整起身”：PPO 的站立任务回报负责全局方向，MuJoCo 产生连续物理状态，
Diffusion 只约束每个滚动窗口仍处在合理支撑运动分布。它不能单独保证最终
站立，所以必须保留 PPO-only、PPO+prior 两组消融。

### 7.2 C6b：条件短期 latent successor（确定性基线后再做）

若确定性 `T_succ` 已经能提高起身，但因同一状态有多条合理路线而出现平均
动作，再训练多候选规划模型：

```text
z_next ~ D_plan(noisy_future_latent, diffusion_step,
                h_ref, z_stand, successor_horizon)
```

其 target 只能是同一 clip 的真实未来窗，或按初态桶筛出的 PPO
`elite_success` rollout 真实未来窗：

```text
z_t       = E_ref(W_i,t)
h_ref     = T_ref_context(z_i,t-n:t)
condition = h_ref + z_stand + K
target    = E_ref(W_i,t+K),  t+K < clip_end
```

推荐两个模型版本：`D_plan-v0` 在冻结 `E_ref/T_ref_context` 后用离线局部 pair
预训练，但暂不接 PPO；`D_plan-v1` 只混合 `B_offline_anchor + elite_success`
真实仿真 pair 再训练，固定后才接入 rollout。普通成功只有加入显式
return/quality condition 后才可进入。partial/support-only 末端不生成伪终端
pair。

这一路线主要参考 SSD 的“短 subtrajectory diffusion + goal-conditioned critic
+ receding horizon”和 Diffuser/Diffusion Policy 的短期生成思想。但 SSD 的
原始数据包含 state-action 且规模远大于本项目；本方案只让 diffusion 生成
latent/state 路标，action 仍由 PPO executor 输出，是明确的工程适配。

### 7.3 在线采样、候选选择与准入门槛

接入顺序固定为：单一确定性 `T_succ` → 单候选 `D_plan` → 多候选。单候选
必须在相同采样延迟预算下不弱于确定性基线，才进入多候选实验。多候选阶段
必须已经有由 PPO rollout 训练的 goal-conditioned critic；否则没有可信依据
判断哪个样本更可执行、更接近站立。

```text
score(z_next) = log p_diffusion(z_next | h_ref, z_stand)
              + beta_goal * Q_goal(s_t, z_next, G_stand)
              - beta_risk * C_risk(s_t, z_next)
```

C7 尚未启用时，`Q_goal` 只使用 PPO 的任务 critic/waypoint 可跟踪头，不做
跨 clip 图搜索；启用 C7 后才加入保守的 `P_H/D_H`。如果所有候选低于阈值，
退回确定性 waypoint 或 PPO-only 控制。初始可用 4–8 步 DDIM，每 5–10 个
policy step 重规划，采样延迟必须计入 50 Hz 控制预算。

C6c 必须单独收集监督，不能把 PPO state-value 直接冒充候选 action-value：

1. 从同一批保存状态在并行环境中采样 1–4 个候选，由固定 executor 各执行
   K 步；
2. 记录 `y_track`（是否到达 endpoint）、`y_safe`（K 步内是否无超限/冲击）
   和 `y_stand`（继续执行到 episode 末是否进入并保持 `G_stand`）；
3. 训练 `Q_goal(s,z_goal)` 预测 `y_stand`/折扣 task return，训练
   `C_risk(s,z_goal)` 预测危险概率；按倒地类型、planner 和 executor 版本分层；
4. 用 held-out candidate 做可靠性校准，并用 ensemble/温度缩放暴露不确定度；
5. 一个 rollout 周期内冻结 critic。executor 或 planner 大幅更新后重新采集、
   重新校准，OOD 候选直接 fallback。

准入标准不是离线 diffusion loss，而是相对确定性 `T_succ` 的 held-out
future coverage、固定 fallen 初态闭环成功率、起身时间、安全成本和推理延迟。
没有稳定增益就保留 C6a 奖励，删除在线 C6b planner。

### 7.4 默认站姿如何进入 Diffusion

`z_stand` 由重复 10–20 帧的解析默认目标窗口及有界噪声编码得到，不需要先
训练站立 policy 或收集站立 buffer。PPO 成功后，可把安全 terminal window
加入经验终端分布，让目标从解析点扩展成真实吸引域。

默认站姿只能作为长期 condition 或 inpainting endpoint，不能把每条 partial
轨迹的最后一帧直接连接到它。任何 inpainting 结果仍须由 MuJoCo/PPO 验证，
不能视为真实 action 或动力学监督。

### 7.5 明确不采用的首版做法

- Diffusion 直接输出 29-DoF torque/action；
- 从第一轮 PPO 起同时反向更新 Transformer、Diffusion 和 Actor；
- 给 partial 轨迹补默认站姿或对缺失区间线性插值；
- 把跨 clip latent 近邻直接标成真实 successor；
- 使用随 PPO 漂移的 encoder 定义 Diffusion target；
- 一次生成 4–8 s 完整恢复并要求低层 policy 盲目跟踪。

## 8. PPO 控制与联合目标

### 8.1 Actor 和 Critic

Actor：

```text
a_t ~ pi_phi(a | h_ref, optional h_actor, z_goal,
             velocity_command, planner_age)
```

其中 `planner_age` 表示当前 waypoint 已执行多久，避免 Actor 不知道路标的
时间相位；同时输入明确的 successor horizon。action 仍使用项目现有 G1
关节位置 action scale。

Critic 可以使用 privileged 状态：

- root height 和线速度真值；
- 全身接触和接触力；
- posture phase；
- 关节力矩/功率；
- 当前 planner 候选质量和目标；
- 后置的 `P_H/D_H`、`V_to_stand` 和 risk 分数；
- 域随机化参数。

Actor 不能依赖实机不可获得的 privileged 状态。

### 8.2 因果路标奖励

原始“在时刻 `t` 比较未来 `x[t+1:t+K]`”的奖励无法由当前 RewardManager
直接计算。第一版明确采用 **K 步 endpoint latent**，不假设已有
latent-to-window decoder：

```text
z_goal = T_succ(h_ref, z_stand, K)       # 或后置 D_plan 单候选
z_cur  = E_ref(截至当前时刻的完整历史窗)
r_waypoint_step = clip(d(z_prev,z_goal) - d(z_cur,z_goal))
r_waypoint_K    = 1[d(z_actual_K,z_goal) < epsilon_goal]
```

同一 `z_goal` 连续 K 步输入 Actor/Critic；每步用“到 endpoint 的距离改善”提供
因果稠密 shaping，K 步结束再计算到达奖励并重规划。这只依赖已经发生的历史，
无需回看未来 PPO storage。若以后确实需要逐帧 waypoint sequence，必须新增
显式 trajectory head/`window_decoder.py`、对应同 clip 序列监督和独立消融，
不能把一个 endpoint predictor 当作已能输出 K 个目标。

每个 horizon 结束后，将实际窗口 `W'` 编码为 `z_actual`，记录：

```text
planned edge: z_t -> z_goal
actual edge:  z_t -> z_actual
tracking:     distance(z_actual, z_goal)
```

所有结果写入 `B_sim_attempt`；满足 latent 到达、动力学安全和后续可继续恢复
时才进入 `B_sim_success`。C7 启用后，后者还可成为 confirmed edge。失败不能
只按 tracking error 判断：即使到达目标姿态，若产生过大冲击、滑移或不可
持续接触，也应作为风险负样本。

### 8.3 奖励结构

总奖励建议写成连续状态门控形式，不依赖人工阶段边界：

```text
r = w_progress * r_progress
  + w_support  * r_support
  + w_goal     * r_waypoint
  + w_prior    * r_diffusion_prior
  + w_ready    * r_stable_stand
  + g_walk     * r_velocity
  - w_limit    * c_joint_limit
  - w_tau      * c_torque
  - w_power    * c_power
  - w_rate     * c_action_rate
  - w_impact   * c_contact_impact
  - w_slip     * c_support_slip
  - w_self     * c_self_collision
```

建议含义：

- `r_progress`：pelvis 归一化高度 × torso uprightness 的改善量和绝对值；
- `r_support`：由当前接触和姿态连续计算的有效支撑，不把手/膝接触一概当
  错误；
- `r_waypoint`：当前结构状态与 Transformer/Diffusion 路标的相似度；
- `r_diffusion_prior`：可选 C6a frozen score motion-prior reward；
- `r_stable_stand`：进入并保持 `G_stand`；
- `r_velocity`：由 `walk_gate` 平滑开启；
- `c_support_slip`：承重手、膝、脚的切向滑移；
- `c_contact_impact`：高冲击接触，不惩罚正常建立支撑；
- `c_torque/c_power`：使用真实执行器约束，防止仿真可行但实机危险。

注意：恢复期不能始终强力奖励默认关节姿态，否则策略会在躺地时直接把腿
拉向站姿，与建立支撑冲突。默认 pose reward 应随 upright/ready 阶段逐步
增强；速度跟踪奖励同样在稳定站立后开启。

### 8.4 PPO 与辅助模型不是一个混合 minibatch loss

为避免“联合训练”被误解成把所有 loss 简单相加，实际有两类优化步骤：

```text
PPO step（严格 on-policy）:
  L_policy = L_PPO + optional lambda_align * L_actor_history_align

Aux step（offline/sim replay）:
  L_aux = lambda_pred * L_local_prediction
        + lambda_diff * L_diffusion
        + optional lambda_reach * L_P_H_or_D_H
        + optional lambda_phase * L_phase_aux
```

PPO rollout 期间冻结发布版 `E_ref/T_ref_context/T_succ`、`D_prior/D_plan` 和
后置 critic；
Actor/Critic 与可选 `E_actor_history` 参数按当前 rollout 更新。rollout 结束后才用
独立 dataloader 更新 auxiliary candidate，再经准入评测原子切换版本。C6a
作为 reward 时也保持冻结，不能让 reward model 与 policy 同时追逐对方。

checkpoint 必须同时记录 policy、encoder、Transformer、Diffusion、normalizer、
数据 hash 和 prior version。这仍是联合系统的交替训练，但不违反 PPO 的
on-policy 数据要求。

## 9. G1 Recovery 环境改造

建议新建独立 task，例如：

```text
Mjlab-Velocity-Recovery-Flat-Unitree-G1
Mjlab-Velocity-Recovery-Rough-Unitree-G1   # 后续阶段
```

不要直接改变普通 G1 velocity 的训练行为。

### 9.1 必须修改

1. 移除 recovery task 的 `fell_over` 终止；
2. 增加 `recovery_succeeded` 和 `recovery_timed_out`；
3. 将 episode 长度设置为足够覆盖起身和站立保持，首版建议 6–8 s；
4. 增加手、前臂、膝/小腿、脚与地面的 contact sensor；
5. 为真实承重支撑几何配置 `condim=3` 和合理摩擦；
6. 增加持续穿透、危险碰撞、关节硬限位或数值异常终止；
7. 恢复期关闭/降低普通 velocity 的 foot clearance、air time、严格默认 pose
   和速度跟踪项，进入 ready 后平滑恢复；
8. 增加 fallen reset、RSI reset、外力和历史失败状态 reset；
9. 记录 posture/support phase、起身时间、力矩、接触冲击、滑移和成功保持。

### 9.2 接触建模

不能盲目把全部非足部几何都设成高摩擦三维接触，这可能造成衣物式粘地和
数值负担。应通过轨迹 FK 和 MuJoCo 可视化确定实际承重几何，只为手掌、
必要的前臂区域、膝/小腿和足部启用合理的切向摩擦，并分别监控：

- 法向力；
- 切向速度/滑移；
- 接触持续时间；
- 接触冲击；
- 自碰撞。

完成该审计前，重定向轨迹的“运动学精确”不能等同于“G1 动力学可执行”。

### 9.3 Reset 分布

三类 reset 混合使用：

- **RSI**：从 62 条安全轨迹的有效帧重置，重点利用支撑阶段；
- **物理 fallen reset**：随机仰卧、俯卧、左右侧卧、关节扰动后 settling；
- **failure replay**：从当前策略高失败状态重置。

由于 43/63 条数据开始时已在支撑阶段，物理 fallen reset 是弥补“首次建立
支撑”数据缺口的必要组成，而不是可选数据增广。

### 9.4 训练辅助力

可复用 RL_BOY `RecoveryAssist` 的课程思想，在最初阶段用小幅方向性辅助力
帮助 PPO 学会离地和建立支撑，然后根据成功率退火到零。但 G1 的质量、
执行器、尺寸和支撑几何不同，不能照搬 RL_BOY 的 50 N 力范围、阈值和
关节集合。最终所有正式评测必须为零辅助力。

## 10. 可执行的分阶段训练路线

每个阶段都有输入、冻结/更新对象、产物和准入门槛。未通过门槛时不叠加后续
组件，否则无法判断问题来自物理、奖励、Transformer 还是 Diffusion。

### S0：数据、特征、接触与站立目标

输入：63 条当前裁剪后的 G1 CSV、`index.json`、G1 MJCF 和默认 keyframe。

实施：

1. 核实源 FPS、坐标系、单位、joint order 和 quaternion `xyzw`；重采样到
   50 Hz，产物显式写入这些 schema 信息和输入哈希；
2. 按 source recording 分 train/validation/test，生成 `H=20` 历史帧、
   `K=10` future 帧的起步配置，pair 永不跨 clip；
3. 实现统一 64D deployable feature，并做同一状态“离线提取→注入 MuJoCo
   →在线提取”的 parity test；另存 FK clearance/contact proxy 作为辅助标签，
   不混入 64D；
4. 在 MuJoCo 中回放/PD 跟踪全部轨迹，标注穿透、限位、危险接触和可用于
   RSI 的帧；
5. 新建 recovery 专用接触配置、传感器和不含 `fell_over` 的 task；
6. 实现解析 `G_stand` sampler：默认关节姿态 + 小 root/joint 状态噪声 +
   physics settle/filter；实现严格 0.3–0.5 s 成功保持判据。

产物：版本化 dataset、normalizer、feature schema、审计报告、固定 reset
评测集和 recovery environment smoke test。

准入：不存在跨 clip future、partial 末端伪补帧或非有限数据；离线/在线
feature 在约定容差内一致；单脚、腾空、高速经过目标高度都不能误报成功；
默认姿态小扰动在零/小动作下大部分能保持，异常状态可被 settle/filter 拒绝。

### S1：先离线训练 Transformer v0

输入：S0 的全部安全有效局部窗口。此阶段不需要 PPO rollout。

更新：`E_ref + T_ref_context + T_succ`；不训练 action policy。

监督：masked reconstruction、末端状态/速度重建、重叠窗一致性、同 clip
K-step future/delta 预测、可选 support/progress 辅助头和 TDR。partial 轨迹
只训练实际存在的局部未来。人工阶段标签不进入主输入。

产物：`transformer_v0.pt`、冻结 `E_ref_v0`、normalizer/schema/split/hash、
OOD calibration 和 persistence/last-state baseline 报告。

准入：held-out recording 的 K-step 指标明确优于 persistence baseline；causal
mask 测试证明未来输入不能影响当前输出；padding/reset history 测试通过；
可视化确认上升/下降没有被 latent 混为一类，接触 proxy 辅助头能区分主要
支撑趋势。这里的通过只
代表“学会运动学局部未来”，不代表已经能物理起身。

可并行训练 `D_prior-v0`，但此时不接入 PPO。若采用更重但物理性更强的路线，
可参考 GPC：先在仿真中训练可执行 skill tokenizer/decoder，再训练 next-token
Transformer；当前小数据版先把它列为对照升级，而不是 S1 的前置依赖。

### S2：纯任务奖励 Recovery PPO 基线

输入：near-stand reset、轨迹安全帧 RSI、随机仰/俯/侧卧 fallen reset 和固定
评测集。Transformer/Diffusion 均不接入。

更新：普通 PPO Actor/Critic。训练初期提高 near-stand/RSI 比例，成功后逐步
提高 fully fallen 和扰动比例；辅助力默认从 0 开始，只有探索确实失败时才
作为可退火课程。

奖励：高度/upright progress、有效支撑、严格 `G_stand` 保持和安全成本；
默认 pose reward 在接近直立后增强，不能在倒地阶段强迫手臂离开地面。

产物：`ppo_task_only` 基线和接触/力矩/滑移/二次跌倒统计。

准入：环境无 NaN/立即终止/成功误判；RSI 与至少一类 fully fallen 初态出现
稳定上升的成功学习曲线；固定评测的成功率和安全指标可重复。该阶段失败就
修物理、reset 和 reward，不加入 motion prior 掩盖问题。

### S3：冻结 Transformer 的 goal-conditioned PPO

输入：S1 的 `E_ref/T_ref_context/T_succ v0` 和 S2 环境。发布版 Transformer
全冻结。

接入数据流：

```text
64D history → frozen E_ref/T_ref_context/T_succ → endpoint z_goal
Actor(history, z_goal, last_action, planner_age) → 29D joint target
Critic(privileged state, z_goal) → value
```

课程分三步：

1. RSI 时使用 demonstration 真实未来窗作为 teacher waypoint，先训练 executor；
2. 在同一 batch/环境比例中逐步把真值 waypoint 替换为冻结 `T_succ` 预测；
3. 扩到随机 fallen reset；OOD 时关闭 waypoint 奖励并退回 task-only PPO。

`z_goal` 必须同时进入 Actor、Critic 和 waypoint reward，并保持 K 步后重规划；
不能只放在 reward 中。记录目标窗、实际到达窗、tracking error 和安全结果。

准入：与 S2 在完全相同固定 reset 上 A/B；冻结权重变化必须为零；至少满足
成功率不下降，并在成功率、起身时间或动作安全中有预先规定的一项显著收益。
可把“成功率提高约 5 个百分点或起身时间明显下降”作为首轮工程 promotion
门槛，但应连同置信区间报告，而不是当作论文定理。

### S4：用仿真数据交替更新 Transformer

前置：S3 已证明 frozen Transformer 确实能引导，而不是仅离线 loss 好看。

一个宏周期严格执行：

1. 冻结 prior vN，PPO 仅用当前 on-policy rollout 更新 Actor/Critic；
2. Recorder 保存非终止真实 transition，并单独保存 terminal state，禁止把
   auto-reset 后 observation 当作 terminal next state；
3. 把数据分入 `B_sim_success` 与 `B_sim_attempt`，和永久
   `B_offline_anchor` 混合；
4. rollout 之间先只微调 `T_succ` head；必要时才以小学习率解冻
   `T_ref_context` 最后 1–2 层；`E_ref v0` 始终冻结；
5. 在 held-out 离线集、固定仿真集和安全集评测 candidate；通过后原子发布
   vN+1，失败则继续使用 vN；
6. 切换版本后再收集下一批 on-policy rollout。

只有分层筛选的 `elite_success` transition 默认训练无条件 desired successor；
所有 transition 可训练 OOD/risk；失败 transition 不能被当成期望动作。旧
replay 永不进入 PPO surrogate。初始 auxiliary batch 可至少一半来自 offline
anchor，随后由
防遗忘评测调节，而不是固定为不可更改的比例。

准入：离线 held-out 指标退化在预设容差内（首轮可用 5%）；固定仿真成功率
不下降；waypoint 跟踪和 OOD 校准改善；安全成本不过阈值。只有通过 promotion
的版本进入 rollout。

### S5：接入 C6a frozen Diffusion motion prior

前置：至少 S2 已通过；推荐在 S3/S4 最佳 checkpoint 上做正交消融。

1. 用所有安全局部窗口训练并冻结 `D_prior`；
2. 以已经执行完成的因果窗口计算 `r_diffusion_prior`，不访问未来状态；
3. 从很小 `w_prior` 开始，与 task-only 或 Transformer PPO 同时 A/B；
4. rollout 中只更新 PPO，`D_prior` 参数严格冻结；
5. 调节顺序优先是 reward weight，再是 SDS scale 和 diffusion steps。

准入：`PPO+prior` 的起身成功率不低于对应无 prior 基线，同时动作冲击、
抖动或 motion-window 相容度至少一项改善。若小数据导致 prior 过拟合并压制
必要的新支撑动作，则降低权重或删除 C6a。

### S6：可选 C6b successor Diffusion planner

前置：S3 确定性路标有效、S4 已产生足够的成功物理 transition，并观察到
明确多模态瓶颈；否则停在 C6a。

1. 冻结 `E_ref/T_ref_context`，离线真实 pair 训练 `D_plan-v0`；
2. 加入 `elite_success` 训练 `D_plan-v1`；普通成功需显式 quality condition，
   失败数据只训练 risk/OOD；
3. 先以单候选替换确定性 head，Actor/Critic 接口不变；
4. 单候选通过后训练 goal-conditioned candidate critic，再尝试 2–4 候选；
5. rollout 固定一个 D 版本，模型更新仍只发生在宏周期之间；
6. 与相同延迟预算的 `T_succ` 做闭环对照。

准入：单候选不弱于确定性基线；多候选在固定初态、不同倒地类型上提高覆盖
或成功率且不增加风险；50 Hz Actor 与低频 planner 满足延迟预算。否则不部署
在线 Diffusion。

### S7：可选 GAS 式 stitching/reachability

只有 S3–S6 仍表现出局部循环/断链时实施：先训练 TDR 或 `D_H/P_H`，只用
真实时间边建基础图；跨 clip 近邻只进 candidate pool；由当前 PPO executor
多次执行确认或否决，使用 ensemble/不确定度和最大边长防止 SoRB 所说的
“wormhole”假捷径。先让图搜索 + 确定性 subgoal 跑通，再让 Diffusion 提议
局部 subplan。启用后 executor 大幅更新必须重新校准 critic/图。

### S8：Velocity 接管、部署精炼与实机

从零速度命令和稳定保持开始，通过 `walk_gate` 逐步加入小速度和完整 velocity
分布；flat 通过后再加入 rough terrain、摩擦/质量/延迟随机化和推扰。参考
HumanUP，早期可放宽动作平滑/速度约束以发现起身，成功后逐步收紧到硬件
允许范围。最终对在线 planner 和蒸馏 Actor 做相同安全测试，实机按吊绳/软垫、
易姿态到难姿态推进。

## 11. 推荐的初始超参数范围

以下只是首轮实验范围，最终需由基线和消融确定：

| 模块 | 初始建议 |
|---|---|
| policy frequency | 50 Hz |
| window length | 固定 10/20 帧起步，之后随机 8–20 帧 |
| Transformer context | 2–4 个窗口，总历史约 20–40 帧 |
| Transformer dim/layers | 128–256 / 3–4 层 |
| `E_ref` latent | 64 或 128 |
| Diffusion horizon | 10–20 policy steps |
| replanning interval | 5–10 policy steps |
| online denoising | 4–8 DDIM steps |
| candidates | 1 起步，稳定后增加到 2–4 |
| C7 cross-clip neighbors | 后置启用时每节点 5–20 个候选，再由 `P_H/D_H` 过滤 |
| PPO rollout | 48–96 steps/env |
| training episode length | 6–8 s（按起身课程调整） |
| handoff evaluation length | 至少 9 s：5 s recovery + 0.5 s hold + 3 s velocity |
| success hold | 0.5 s |
| action | 现有 G1 position action scale |
| training terrain | flat 起步，rough 后置 |

原 G1 PPO 的 `num_steps_per_env=24` 只覆盖约 0.48 s，对包含恢复阶段的优势
估计可能过短，建议从 48 或 96 开始对照，同时重新调节 batch、学习率和显存。

## 12. 评测与消融

### 12.1 固定评测集合

评测初态必须与训练采样器分离并固定种子，至少包含：

- 仰卧；
- 俯卧；
- 左侧卧；
- 右侧卧；
- 四肢交叉/不利关节姿态；
- 策略历史失败状态；
- 不同摩擦、质量、COM、执行器强度和传感噪声；
- held-out source recording 的 RSI 状态。

### 12.2 指标

- recovery-only：5 s 内进入 `G_stand` 并完成 0.5 s 保持的成功率；
- 成功时间分布；
- 站立保持和二次跌倒率；
- support_start、support_complete、ready 的到达率和时间；
- held-out 窗口 successor latent/state 误差；
- OOD/risk 对仿真成功、失败和危险边的校准误差；
- C6c 启用后，`Q_goal/C_risk` 的 Brier/ECE、top-1 候选相对随机候选的成功
  uplift 和 fallback 率；
- C7 启用后，`P_H/D_H` 的有限时域可达校准、跨 clip 物理确认率、错误接受率
  和图搜索循环率；
- 峰值/均方关节力矩、速度和功率；
- action rate 和高频能量；
- 手、膝、脚接触冲击与滑移；
- 自碰撞和地面穿透；
- handoff：成功保持后额外 3 s 的 velocity tracking RMSE；整段评测至少 9 s，
  不与 6–8 s 训练 episode 混用；
- Diffusion 采样延迟和整体控制实时率；
- 按姿态、数据来源和 domain randomization 难度分层的成功率。

### 12.3 必做消融

1. 普通 recovery PPO；
2. PPO + RSI；
3. PPO + RSI + frozen deterministic `T_succ`；
4. Transformer PPO：真值路标课程 vs 直接预测路标；
5. Transformer PPO + 仿真交替更新，对比始终冻结 Transformer；
6. PPO/Transformer PPO + C6a frozen Diffusion prior；
7. Transformer PPO + C6b 单候选 Diffusion；
8. Transformer PPO + C6b 多候选 + goal-conditioned selection；
9. 只用同 clip 重规划，对比后置 C7 GAS 式确认边；
10. 去掉人工阶段辅助损失，验证核心方法不依赖精确阶段标签；
11. 去掉 partial/support-only 数据，量化局部片段价值；
12. 仅关节噪声站立 reset，对比结构化 root/velocity 噪声；
13. 去掉 `B_offline_anchor`，检查 Transformer 灾难性遗忘；
14. 去掉成功/失败 replay 分流，检查是否学到失败 successor；
15. 共享漂移 encoder 对比 frozen `E_ref` + policy-side adapter；
16. AMP-style motion prior 基线，对比 C6a SMP-style frozen prior。

至少使用多个训练种子，报告均值、标准差和置信区间。不能只展示一条看起来
成功的视频。

## 13. 相对原《Transformer PPO 端到端联合训练》方案的修改

| 原思路或假设 | 本项目中的问题 | 修订方案 |
|---|---|---|
| 只有人体/通用恢复数据 | 当前已有高精度 G1 重定向数据 | 主先验改用 `lafan-g1-getup` 的 G1 原生状态 |
| 必须逐条使用完整轨迹并精确切分阶段 | 数据中大量仅支撑片段，阶段边界也不应成为部署依赖 | 全部轨迹统一采样短窗；阶段标注只作辅助监督和评测 |
| 整段轨迹压成单 latent | 容易丢失末端状态、运动方向和接触趋势 | 短窗编码 `z_state + z_motion`，显式输入时长和 horizon |
| 不同片段 latent 接近即可拼接 | 相似姿态可能具有相反速度或不兼容接触 | 首版不做跨 clip 标签；需要时由 `P_H/D_H` 和 PPO/MuJoCo 确认 |
| 局部下一段预测自然形成完整起身 | 局部模型可能循环或停留在支撑动作 | 先靠 `z_stand`、任务 critic 和 receding horizon；实证断链后再加 GAS 式图 |
| 每条训练序列都有完整未来 | 许多轨迹没有完整站立尾段 | 右删失 + valid mask；仅有效局部窗口参与未来监督 |
| partial 末端也可视为恢复终点 | 会把截断误当失败或成功 | 单独保留 `outcome/terminal_mode/terminal_safe` |
| “Diffusion”只有一种接法 | motion-prior reward 与 successor planner 的监督和时机不同 | C6a frozen score reward 与 C6b latent planner 分开实现、分开消融 |
| Diffusion 补全后可直接执行 | 运动学补全不保证接触和力矩可行 | C6b 仅生成短期 latent 路标，PPO 闭环执行 |
| 必须先训练稳定站立模型/buffer | 增加无必要依赖，默认姿态本身已知 | 解析 `G_stand` + 结构化噪声 + settle/filter；PPO 成功尾段后续回灌 |
| 所有 partial endpoint 配对默认站姿 | 会产生虚假跳跃监督 | 只以默认站姿作长期条件；连接由完整数据和 PPO replay 学习 |
| `z_next` 只进入 reward | 策略面对随机隐藏目标，形成部分可观测问题 | `z_goal` 必须输入 Actor 和 Critic |
| 当前时刻直接奖励未来 chunk | 当前 RewardManager 无法因果访问未来 | 持有 endpoint K 步，以 latent 距离改善做 shaping，K 步后验收到达 |
| 同一个在线 Transformer 定义策略、Diffusion 和奖励 | PPO 更新导致 latent 坐标和奖励漂移 | 冻结 `E_ref/T_ref_context/T_succ`，PPO 只训 head/可选 adapter |
| PPO、Transformer、Diffusion 每个 minibatch 同步端到端更新 | PPO 是 on-policy，旧 CSV 又没有 action/log-prob | rollout 内固定 prior；rollout 间分别更新辅助模型并做版本准入 |
| 离线 Transformer 可直接成为动作策略 | CSV 只有状态，没有 action/reward/return | 离线只学窗口与 successor，MuJoCo PPO 学 29D action executor |
| 直接沿用普通 G1 velocity 环境 | 倒地即终止，奖励会阻碍支撑动作 | 新建 G1 recovery task，移除 `fell_over` 并阶段门控 |
| 足部接触足以训练起身 | 手、膝、前臂是关键支撑 | 增加支撑传感器和 task-specific `condim=3` 摩擦 |
| 复制 RL_BOY recovery 参数 | 质量、几何、关节和执行器均不同 | 只复用算法结构，重新标定 G1 力、阈值和关节集合 |
| 用帧数直接定义窗口 | 数据与环境频率不同 | 全部用秒定义，编译时映射到 50 Hz |
| 24-step PPO rollout 足够 | 只覆盖约 0.48 s | 对照 48–96 steps/env |
| 随机切窗口划分数据 | 同源 recording 严重泄漏 | 按 source recording 分组 split |
| 绝对高度/精确接触均可给 Actor | 实机未必可获得 | 不可部署信息只给 privileged Critic |
| 一开始训练完整 4–8 s Diffusion | 当前有效数据量和完整轨迹不足 | 先训练 0.2–0.4 s 局部 successor Diffusion |

## 14. 建议的代码落点

最终实现时建议保持模块边界清晰：

```text
src/mjlab/tasks/velocity/
  config/g1/
    recovery_env_cfgs.py       # G1 recovery 环境配置
    recovery_rl_cfg.py         # Transformer PPO 配置
  mdp/
    g1_recovery.py             # G1 reset、成功、指标、支撑逻辑
    recovery_planner.py        # CommandTerm、z_goal 缓存与 K 步重规划
  recovery_data/               # 扩展已有目录，不另建重复 package
    g1_csv.py                  # 36D CSV → 50 Hz、manifest 与审计
    g1_windows.py              # 64D feature、clip/window/group split
  recovery_prior/              # 扩展已有 VAE 训练基础设施
    features.py                # 离线/在线共享 feature schema
    window_encoder.py          # E_ref / E_actor_history、z_state / z_motion
    transformer.py             # latent context 与确定性 successor baseline
    diffusion_prior.py         # C6a frozen SMP-style prior
    diffusion_planner.py       # C6b conditional latent denoiser
    candidate_critic.py        # C6c Q_goal/C_risk 与校准
    replay.py                  # offline anchor/sim success/sim attempt
    reachability.py            # 可选 P_H/D_H、risk 与 GAS graph
  rl/
    recovery_model.py          # Transformer goal-conditioned Actor/Critic
    recovery_runner.py         # 宏周期、版本准入与原子切换
  scripts/
    build_g1_recovery_dataset.py
    train_g1_recovery_transformer.py
    train_g1_recovery_diffusion_prior.py
    train_g1_recovery_diffusion.py
    build_g1_recovery_graph.py  # 仅 S7
    evaluate_g1_recovery.py
```

已有 `recovery_data/` 主要编译 BVH，已有 `recovery_prior/vae.py` 是 85D 整段
重建 VAE；可复用 manifest、mask、normalizer 和 checkpoint 风格，不能复用
其 feature 维度或权重作为在线 Transformer。`csv_to_npz.py` 已有 SLERP、速度
和 FK 逻辑，应抽成无 `/tmp`/W&B 副作用的批量库函数。

ObservationManager 可提供历史，但应建立一个完整 64D frame term 再 reshape
为 `[B,H,D]`；多个 term 各自 flatten 后通常是 term-major，不应误当 time-major。
`CommandTerm` 适合持有版本化 `z_goal` 并让 observation/reward 共用。RSL-RL
当前标准配置主要面向 MLP/CNN/RNN，因此需要自定义 model 和 runner，不能只
修改 `hidden_dims`。

Recorder 的 terminal transition 需要专门测试：自动 reset 后的新 observation
不能被写成终止前 `next_state`。新 G1 recovery task 注册后，还要同步修改
`tests/test_velocity_task.py` 中对任务集合/命名的假设。

### 14.1 建议按可独立验收的实现增量提交

| 增量 | 实现内容 | 对应阶段 | 必须先通过的测试 |
|---|---|---|---|
| I0 | G1 CSV compiler、64D feature、recording split、window sampler | S0 | schema、FPS、quat、no-cross-clip、feature parity |
| I1 | recovery env、接触、三类 reset、解析站立目标/成功保持 | S0–S2 | reset/接触/成功反例、CPU smoke test |
| I2 | `E_ref/T_ref_context/T_succ` 与离线 trainer | S1 | causal mask、padding、held-out、checkpoint hash |
| I3 | waypoint `CommandTerm` 与 goal-conditioned Transformer PPO model | S3 | K 步缓存、reset history、冻结参数、Actor/Critic 都收到 goal |
| I4 | auxiliary replay、terminal recorder、版本化 runner/promotion | S4 | on-policy 隔离、terminal next-state、rollback/reproducibility |
| I5 | C6a diffusion prior 与冻结 reward | S5 | 无未来泄漏、reward 数值/梯度、prior 参数不变 |
| I6 | C6b latent planner 与 C6c 候选 critic | S6 | 同 clip target、并行候选标签、校准、延迟、fallback、确定性 A/B |
| I7 | 可选 TDR、`P_H/D_H`、graph | S7 | 未标注对不作负样本、方向性、假捷径/物理确认 |
| I8 | velocity handoff、导出、部署延迟与安全 | S8 | 固定初态回归、ONNX/JIT、实时率和 clamp |

不要在 I0–I4 未跑通前一次性提交 I5–I7。每个增量都保留不启用新组件的配置，
以便做严格消融和回退。

## 15. Checkpoint、可复现性与部署

每个训练 checkpoint 至少记录：

- Actor、Critic、可选 `E_actor_history`、固定 `E_ref/T_ref_context`、`T_succ`；
- 启用时的 `D_prior/D_plan`、sampler、`Q_goal/C_risk` 校准、`P_H/D_H` 和
  candidate selector；
- C7 启用时的图节点/边版本，以及真实边、候选边和确认边来源；
- optimizer/scheduler 和 observation normalizer；
- 数据 manifest、split、schema、CSV 输入和解析 `G_stand` sampler 配置哈希；
- G1 XML/碰撞配置哈希；
- reward、curriculum、domain randomization 配置；
- git commit、随机种子、全局 step 和 prior 版本；
- replay 数据来源与过滤配置。

部署提供两种模式：

1. **确定性 planner**：`E_ref/T_ref_context/T_succ + Actor`，作为默认部署基线；
2. **在线 Diffusion planner**：`E_ref/T_ref_context + D_plan + Actor`，仅在闭环增益和
   延迟准入均通过后使用；
3. **蒸馏 actor**：把 planner 输出蒸馏到 Actor，延迟最低，适合作为首版
   实机候选。

实机前必须进行 action/velocity/torque safety clamp、站立状态机保护、急停、
软垫/吊绳测试和由易到难的姿态课程。仿真 90% 成功率不等于可直接无保护
上实机。

## 16. 主要风险与对策

| 风险 | 对策 |
|---|---|
| 完整起身示范少 | 局部窗口先验、物理 fallen reset、成功 replay，避免伪补全 |
| 短窗 latent 只学到姿态 | 分离/约束 `z_state` 与 `z_motion`，重建速度和接触趋势 |
| 跨片段近邻是假连接 | 首版不用；C7 用 `P_H/D_H`、不确定度和 PPO/MuJoCo 确认 |
| 局部预测循环而不到站立 | `G_stand`/task critic/replanning；实证需要后再加 C7 |
| 首次建立支撑覆盖不足 | 随机 fallen reset + failure replay + 逐步退火辅助力 |
| 重定向轨迹动力学不可行 | MuJoCo PD 回放审计，只把安全状态用于 RSI/控制监督 |
| Diffusion 生成 OOD 路标 | 短 horizon、候选筛选、PPO 拒绝/重规划、uncertainty gate |
| latent reward 漂移 | 冻结版本化 `E_ref` |
| Transformer 被失败 rollout 带偏 | success/attempt 分流、offline anchor、防遗忘准入与回退 |
| replay 污染 PPO | PPO surrogate 只读当轮 on-policy storage，辅助 replay 用独立 dataloader |
| 普通 velocity 奖励阻碍起身 | recovery phase 门控，ready 后再恢复速度和默认 pose |
| 手/膝摩擦错误 | 专用支撑几何和传感器，按真实承重部位配置 |
| 训练成功但实机观测缺失 | 从一开始限制 Actor 为可部署观测，Critic 才使用真值 |
| 在线 Diffusion 延迟过高 | 低步数采样、低频重规划、候选数课程、最终蒸馏 |
| 同源数据泄漏 | recording-level split 和固定 held-out 评测 |

## 17. 最终推荐实施顺序

按收益、风险和可诊断性排序：

1. **S0/I0–I1：数据、64D feature parity、G1 recovery 物理环境与解析
   `G_stand`**；
2. **S1/I2：全部有效短窗离线训练 `E_ref/T_succ v0`**；
3. **S2：无 Transformer/Diffusion 的 Recovery PPO 基线**；
4. **S3/I3：冻结 Transformer，真值路标→预测路标课程训练 PPO executor**；
5. **S4/I4：offline anchor + sim replay 的宏周期 Transformer 更新与准入**；
6. **S5/I5：SMP 式 frozen Diffusion prior reward 消融**；
7. **S6/I6：有明确多模态收益时才做 SSD 式 latent Diffusion planner**；
8. **S7/I7：仍有循环/断链时才做 GAS 式 `P_H/D_H` 与恢复图**；
9. **S8/I8：velocity 接管、rough、强随机化、蒸馏与安全部署**。

这一路线能够利用所有有价值的支撑片段，又不会假设它们包含不存在的站立
尾段，也不要求人工精确划分每一个支撑子阶段。默认屈膝站姿为系统提供统一
的长期目标，Transformer 学习局部片段的时间上下文，PPO 的任务回报和
MuJoCo 闭环把局部知识连续组织成完整起身。Diffusion 先可作为冻结运动先验，
再在有证据时承担多模态短期规划；恢复图只解决实证存在的断链，不再作为
首版假设。这样仅支撑片段和完整轨迹才能在不制造虚假监督的前提下共同形成
完整恢复能力。

## 18. 论文、官方实现与采用边界

下表优先列论文原文、作者项目页或作者/实验室官方仓库。这里的“采用”是
复用其已验证的组件职责；把这些组件组合到 mjlab/G1 仍需本项目消融验证。

| 组件/决策 | 论文与官方实现 | 本方案采用 | 不能照搬的部分 |
|---|---|---|---|
| 离线 Transformer→在线适配 | [Online Decision Transformer](https://proceedings.mlr.press/v162/zheng22c.html)、[官方代码](https://github.com/facebookresearch/online-dt) | 先离线预训练、再用在线数据周期适配并保留离线 anchor | ODT 需要 state-action-return 且不是 PPO；本项目旧数据不能进入 PPO loss |
| 轨迹序列建模 | [Trajectory Transformer](https://trajectory-transformer.github.io/)、[官方代码](https://github.com/jannerm/trajectory-transformer) | causal 历史、future-window 建模和规划对照 | 原方法使用状态—动作 token 和搜索；本项目先只建模状态窗口 |
| Transformer stitching 边界 | [Q-learning Decision Transformer](https://proceedings.mlr.press/v202/yamagata23a.html) | 支持“局部 next prediction 不等于自动全局拼接”，必要时加入 value/物理反馈 | 其 Q-learning 数据与算法不能直接换成当前 PPO |
| 离线技能先验→下游 RL | [SPiRL](https://proceedings.mlr.press/v155/pertsch21a.html)、[官方代码](https://github.com/clvrai/spirl) | latent goal/skill 与低层 closed-loop policy 分工 | 原示范含 action；本项目的低层 executor 必须由 MuJoCo PPO 学 |
| 无标注局部片段→PPO 风格约束 | [AMP](https://arxiv.org/abs/2104.02180)、[项目页](https://xbpeng.github.io/projects/AMP/)、[ASE 官方代码](https://github.com/nv-tlabs/ASE) | 未分段 motion clips 与 task reward 共同驱动 PPO；作为强基线 | 判别器训练和机器人/仿真配置不能直接复制 |
| 冻结 Diffusion motion reward | [SMP](https://arxiv.org/abs/2512.03028)、[MimicKit 官方实现](https://github.com/xbpeng/MimicKit/blob/main/docs/README_SMP.md) | C6a：离线训练、冻结、作为滚动窗口 score/SDS reward | 它不是 next-window planner，也不保证到达站立；当前 3129 帧需防过拟合 |
| 物理可执行 token Transformer | [GPC](https://arxiv.org/abs/2606.29148)、[ProtoMotions 官方代码](https://github.com/NVLabs/ProtoMotions) | 作为升级路线：先物理跟踪/量化技能，再 next-token，再任务适配 | 原工作是大规模数据/GPU；直接 CSV latent 没有其物理可执行保证 |
| 短片段 Diffusion stitching | [SSD](https://ojs.aaai.org/index.php/AAAI/article/view/29215)、[作者代码](https://github.com/rlatjddbs/SSD) | C6b：短 subplan、goal critic、receding horizon 的结构 | 原数据含 state-action 且规模大；本方案 diffusion 只给 latent goal，PPO 出 action |
| 轨迹 Diffusion | [Diffuser](https://proceedings.mlr.press/v162/janner22a.html)、[官方代码](https://github.com/jannerm/diffuser)、[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)、[官方代码](https://github.com/real-stanford/diffusion_policy) | goal conditioning、多模态短 horizon 和 action-chunk/replanning 思想 | 不把当前无 action 的 CSV 直接训练成 G1 action diffusion |
| 无人工硬分段的 future-goal relabel | [Relay Policy Learning](https://proceedings.mlr.press/v100/gupta20a.html)、[项目页](https://relay-policy-learning.github.io/) | 滑窗未来目标和后续 policy-gradient fine-tuning 思路 | 示范含 state-action；本项目只借鉴高层 future-window supervision |
| 恢复图/stitching | [GAS](https://proceedings.mlr.press/v267/baek25a.html)、[官方代码](https://github.com/qortmdgh4141/GAS) | S7：TDR→构图→goal policy→图搜索的主线 | 只有 PPO executor 可执行后才能校准 G1 接触可达性 |
| Replay 图与可达边 | [SoRB](https://papers.nips.cc/paper/9660-search-on-the-replay-buffer-bridging-planning-and-reinforcement-learning)、[QRL](https://proceedings.mlr.press/v202/wang23al.html)、[QRL 官方代码](https://github.com/quasimetric-learning/quasimetric-rl) | OOD 假捷径防护、有限边长、方向性距离和不确定度 | 它们没有直接验证 G1 地面起身；`P_H/D_H + PPO 确认` 是本项目适配 |
| G1 起身环境/课程 | [HoST](https://arxiv.org/abs/2502.08378)、[官方代码](https://github.com/InternRobotics/HoST)；[HumanUP](https://arxiv.org/abs/2502.12152)、[官方代码](https://github.com/RunpeiDong/humanup) | 接触丰富起身、多 critic/辅助课程、先发现再收紧硬件约束 | Isaac Gym/机器人参数不能原样移植到 mjlab |
| RSI | [DeepMimic](https://xbpeng.github.io/projects/DeepMimic/DeepMimic_2018.pdf)、[官方代码](https://github.com/xbpeng/DeepMimic) | 从示范安全帧初始化以改善长时恢复探索 | RSI 不能代替随机 fully fallen reset |

关于此前“窗口可达性模型是不是自己想出来的”：答案应精确表述为——可达
critic、replay 图和图搜索都有 GAS/SoRB/QRL 等明确来源；但此前写的独立
`R_link(z_a,z_b)`、人工跨 clip 负样本和 PPO 物理确认的具体组合并不是某篇
论文的原样算法。本文已将它改成后置的 policy-conditional `P_H/D_H`，删除
“随机跨 clip 就是负样本”的不可靠假设，并明确标注 G1 工程适配边界。

最终建议优先复现三个有公开实现、能够独立回答问题的基线：HoST-style
task-only PPO、AMP/SMP-style motion-prior PPO、frozen deterministic
Transformer waypoint PPO。只有第三个基线证明离线局部未来确实有增益，才
继续做 Transformer 仿真更新和 C6b；只有出现真实断链，才进入 C7。
