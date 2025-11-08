# 手工搭建 Transformer

本项目为“大模型基础与应用”课程的期中作业。其核心目标是仅依赖 PyTorch 基础库，从零开始完整地搭建、训练和分析一个标准的 Encoder-Decoder Transformer 模型。

该模型在 **IWSLT2017 (EN↔DE) ** 数据集上进行了训练，并完成了一系列严格的消融实验，以验证其核心组件的必要性。

## 主要特性

- **完整架构实现**: 实现了包含 Encoder 和 Decoder 的标准 Transformer 模型。
- **核心组件**:
  - 缩放点积注意力 (Scaled Dot-Product Attention)
  - 多头注意力机制 (Multi-Head Self-Attention)
  - 位置前馈网络 (Position-wise Feed-Forward Network)
  - 残差连接与层归一化 (Residual Connections & Layer Normalization)
  - 位置编码 (Positional Encoding)
- **训练框架**: 包含完整的数据预处理、模型训练、验证和结果可视化流程。
- **消融实验**: 通过代码和配置分离，轻松复现对**位置编码**、**注意力头数**和**残差连接**的消融研究。

## 项目结构

```
transformer_project/
|-- src/                # 核心源代码
|   |-- model.py        # Transformer 模型定义
|   `-- train.py        # 训练、验证与评估脚本
|-- scripts/            # 运行脚本
|   `-- run.sh          # 自动化安装依赖并启动训练
|-- configs/            # 配置文件
|   `-- base.yaml       # 存储所有超参数和路径设置
|-- results/            # 实验结果（损失曲线图）
|-- tokenizers/         # 训练好的分词器模型
|-- dataset/            # 下载的数据集缓存
|-- requirements.txt    # 项目依赖
`-- README.md           # 项目说明文档
```

## 环境搭建与安装

本项目推荐使用 Conda 进行环境管理，以确保依赖的稳定性和一致性。

### 1. 克隆仓库

```bash
git clone https://github.com/Fresh233/Transformer.git
cd Transformer
```

### 2. 创建并激活 Conda 环境

我们使用 Python 3.10 版本。

```bash
conda create -n transformer python=3.10
conda activate transformer
```

### 3. 安装核心依赖 (PyTorch with CUDA)

为了使用 GPU 加速，我们通过 Conda 安装与 CUDA 版本兼容的 PyTorch。

```bash
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
```

### 4. 安装其余依赖

使用 `pip` 和 `requirements.txt` 文件安装剩余的包。`requirements.txt` 中已锁定 NumPy 版本以确保兼容性。

```bash
pip install -r requirements.txt
```

## 如何运行

### 复现基线实验 (4头注意力)

所有实验的超参数，包括**随机种子**，都定义在 `configs/base.yaml` 文件中，以确保结果的完全可复现性。

```yaml
# configs/base.yaml
# ...
seed: 42
# ...
```

在项目根目录下，首先为运行脚本赋予可执行权限，然后直接运行它。

```bash
chmod +x scripts/run.sh
./scripts/run.sh
```

**预期行为**:
- 脚本会自动安装 `requirements.txt` 中的依赖（如果尚未安装）。
- 程序将开始运行，自动下载 IWSLT2017 数据集（至 `./dataset`），并训练分词器（至 `./tokenizers`）。
- 训练过程的日志将打印在终端上，包括每个 epoch 的训练和验证损失。
- 训练结束后，性能最佳的模型权重将保存为 `best_model.pth`，损失曲线图将保存至 `results/loss_curve.png`。

---

## 复现消融实验

要复现报告中的消融实验，请在运行 `./scripts/run.sh` **之前**进行以下修改。**注意**：每次实验前，建议修改 `src/train.py` 文件底部的输出图片名称，以避免覆盖之前的结果。

### 1. 移除位置编码

- **文件**: `src/model.py`
- **修改**: 在 `Transformer` 类的 `forward` 方法中，注释掉 `self.pos_encoder` 的调用。

```diff
-        # 使用位置编码
-        src_emb = self.pos_encoder(self.src_tok_emb(src))
-        tgt_emb = self.pos_encoder(self.tgt_tok_emb(tgt))
+        # 不使用位置编码
+        src_emb = self.src_tok_emb(src) 
+        tgt_emb = self.tgt_tok_emb(tgt)
```

- **修改**: 在 `Transformer` 类的 `encode` 方法中，注释掉 `self.pos_encoder`

```diff
-        # 使用位置编码
-        return self.encoder(self.pos_encoder(self.src_tok_emb(src)), src_mask)
+        # 不使用位置编码
+        return self.encoder(self.src_tok_emb(src), src_mask)
```

- **修改**: 在 `Transformer` 类的 `decode` 方法中，注释掉 `self.pos_encoder`

```diff
-        # 使用位置编码
-        return self.decoder(self.pos_encoder(self.tgt_tok_emb(tgt)), enc_out, src_mask, tgt_mask)
+        # 不使用位置编码
+        return self.decoder(self.tgt_tok_emb(tgt), enc_out, src_mask, tgt_mask)
```

### 2. 更改注意力头数 (例如，单头或8头)

- **文件**: `configs/base.yaml`
- **修改**: 更改 `n_heads` 的值。

```diff
# ...
n_layer: 4
- n_heads: 4
+ n_heads: 1  # 或者 8
dropout: 0.1
# ...
```

### 3. 移除残差连接

- **文件**: `src/model.py`
- **修改**: 在 `EncoderLayer` 和 `DecoderLayer` 的 `forward` 方法中，移除残差连接的加法操作。

**EncoderLayer:**
```diff
-        # 使用残差连接
-        h = x + self.attention.forward(...)
-        out = h + self.feed_forward.forward(...)
+        # 不使用残差连接
+        h = self.attention.forward(...)
+        out = self.feed_forward.forward(...)
```

**DecoderLayer:**
```diff
-        # 使用残差连接
-        x = x + self.mask_attention.forward(...)
-        h = x + self.attention.forward(...)
-        out = h + self.feed_forward.forward(...)
+        # 不使用残差连接
+        x_after_mask_attn = self.mask_attention.forward(...)
+        h = self.attention.forward(self.attention_norm_2(x_after_mask_attn), ...)
+        out = self.feed_forward.forward(...)
```

## 硬件要求

- **GPU**: 必须使用支持 CUDA 的 NVIDIA GPU。
- **实验环境**: 所有实验均在单张 **NVIDIA Tesla T4 (16GB VRAM)** 上完成。

- **最低要求**: 推荐使用至少有 **6GB VRAM** 的 NVIDIA GPU 以确保训练顺利进行。
