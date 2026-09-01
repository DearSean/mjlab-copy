# 使用 Conda 训练 mjlab

本仓库可以在 Conda 环境中训练。Conda 只负责隔离 Python 和基础工具，
PyTorch、MuJoCo Warp 及 mjlab 的 Python 依赖仍由 `pip` 安装。训练期间不再
使用 `uv run`，因此不会误用仓库中的 `.venv`。

下面的配置面向 Linux、NVIDIA GPU 和 CUDA 12.8 PyTorch wheel，已经在本机
RTX 4060 Ti 上验证。NVIDIA 驱动显示的 CUDA 13.2 是驱动支持的最高 CUDA
版本，不要求 PyTorch 也必须是 cu132；当前驱动可以运行 cu128 wheel。

## 1. 检查驱动

```bash
nvidia-smi
```

只要该命令能看到 NVIDIA GPU，并且驱动足以支持 CUDA 12.8，就不需要在
Conda 中额外安装 `cudatoolkit` 或 `cuda-toolkit`。PyTorch wheel 自带所需的
CUDA 用户态运行库。

## 2. 创建独立环境

```bash
cd ~/GithubItems/mjlab-copy

conda create -n mjlab-conda python=3.13 pip git -y
conda activate mjlab-conda
unset VIRTUAL_ENV

which python
python --version
python -m pip --version
```

这里的 `which python` 应当指向类似
`.../envs/mjlab-conda/bin/python` 的路径，而不是仓库的 `.venv/bin/python`。
如果 shell 提示符同时显示多个环境，以上三个命令比提示符更可靠。

## 3. 安装依赖和项目

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-conda.txt
python -m pip install --no-deps -e .
```

`requirements-conda.txt` 固定了当前验证过的 PyTorch、Warp、MuJoCo、
MuJoCo Warp 和 mjviser 提交。最后一条命令使用 `--no-deps`，是为了让 mjlab
保持可编辑安装，同时避免 pip 再次从 PyPI 替换这些锁定来源。
Weights & Biases 不属于必装依赖，训练默认只写 TensorBoard 日志。

安装 Git 依赖时必须能够访问 GitHub。如果系统缺少本地编译工具，可安装：

```bash
conda install -n mjlab-conda -c conda-forge cmake ninja -y
```

## 4. 验证 GPU 和核心模块

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

python -c "import mujoco, mujoco_warp, warp; print(mujoco.__version__); print(warp.__version__)"

python -c "import importlib.metadata as m; print(m.version('mjlab'))"
```

预期结果包括：

- PyTorch 版本为 `2.9.0+cu128`；
- `torch.cuda.is_available()` 为 `True`；
- GPU 名称为当前机器的 NVIDIA GPU；
- MuJoCo、MuJoCo Warp、Warp 和 mjlab 均可导入。

还可以确认恢复任务已注册：

```bash
python -m mjlab.scripts.list_envs | grep Mjlab-Velocity-Flat-Unitree-G1-Recovery
```

## 5. 检查恢复训练数据

当前仓库已经生成训练所需的动作、SMP 和物理初始化数据。开始训练前检查：

```bash
test -f artifacts/g1_recovery/manifest.json
test -f artifacts/g1_recovery/smp/best.pt
test -f artifacts/g1_recovery/physical_init.npz
```

三条命令都没有输出并返回成功，就可以直接训练。

只有在 `physical_init.npz` 缺失，或者重新生成了
`artifacts/g1_recovery/clips/*.npz` 后，才需要重建物理初始化库：

```bash
python -m mjlab.tasks.velocity.scripts.build_g1_physical_init
```

## 6. 开始 G1 恢复训练

```bash
python -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-Unitree-G1-Recovery \
  --agent.run-name conda_physical_init_full
```

当前 recovery 配置默认使用 1024 个并行环境，适合本机 8 GB RTX 4060 Ti。
如果出现显存不足，可先降到 512 个环境：

```bash
python -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-Unitree-G1-Recovery \
  --env.scene.num-envs 512 \
  --agent.run-name conda_physical_init_512
```

训练日志默认写入：

```text
logs/rsl_rl/g1_recovery_s2/<时间戳>_<run-name>/
```

查看 TensorBoard：

```bash
tensorboard --logdir logs/rsl_rl --port 6006
```

然后在浏览器访问 `http://localhost:6006`。只有显式使用
`--agent.logger wandb`、W&B checkpoint 或 W&B motion registry 时，才需要
额外安装：

```bash
python -m pip install "wandb>=0.22.3"
```

由于物理初始化分布和全身接触模型已经变化，应当新建训练，不建议续接修改前
的 checkpoint。

## 7. 常见问题

### `torch.cuda.is_available()` 为 `False`

先确认当前 Python 确实属于 Conda 环境：

```bash
which python
python -m pip show torch
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

如果版本没有 `+cu128`，重新安装 CUDA wheel：

```bash
python -m pip uninstall -y torch
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.0+cu128
```

### 出现 `.venv` 或 `VIRTUAL_ENV` 警告

这通常表示先前的 uv/venv 环境变量仍留在 shell 中：

```bash
conda activate mjlab-conda
unset VIRTUAL_ENV
hash -r
which python
```

Conda 训练命令不要加 `uv run`，直接使用当前环境的 `python -m ...`。

### Git 依赖下载失败

确认 `git --version` 正常并且可以访问 GitHub，然后重新执行：

```bash
python -m pip install -r requirements-conda.txt
```

### 提示找不到旧的 MuJoCo nightly 或 NumPy 被 yanked

当前 requirements 使用稳定的 `mujoco==3.8.1` 和未撤回的
`numpy==2.3.3`，不依赖 `py.mujoco.org` 的历史 nightly。如果之前的安装
已经失败，直接更新仓库中的 requirements 后重新运行即可；不需要删除 Conda
环境：

```bash
python -m pip install -r requirements-conda.txt
python -m pip install --no-deps -e .
```

### 更换环境后是否要重新生成动作数据

不需要。`artifacts/g1_recovery` 中的 NPZ、SMP checkpoint 和物理初始化库与
Python 虚拟环境无关。只要仓库路径和文件内容没有改变，新 Conda 环境可以
直接读取它们。
