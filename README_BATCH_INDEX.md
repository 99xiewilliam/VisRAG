# VisRAG 批量索引指南

批量处理 `dataset` 目录中的所有 PDF，分别生成**文本 embedding**和**vision tokens**。

---

## 📁 输出结构

```
output/
├── chroma_db/                    # ChromaDB 向量数据库
│   ├── text_pages/              # 文本 embedding 集合
│   └── vision_pages/            # 视觉 embedding 集合
├── vision_tokens/               # 原始 vision token 文件
│   ├── DeepSeek-OCR-_Contexts_Optical_Compression/
│   │   ├── page_1.pt           # 每页的 vision token
│   │   ├── page_2.pt
│   │   └── ...
│   └── ...
└── index_report.json            # 处理报告
```

---

## 🚀 快速开始

### 1. 处理所有 PDF（使用默认配置）

```bash
cd VisRAG
python batch_index.py
```

这会使用 `config.yaml` 中的配置，处理 `dataset/` 中的所有 PDF。

### 2. 使用本地 Embedding 模型（推荐用于批量处理）

```bash
python batch_index.py --config config_batch_example.yaml
```

**为什么推荐本地模型？**
- 无 API 调用延迟
- 无网络依赖
- 无 API 限流问题
- 成本更低

### 3. 只处理前 N 个 PDF（测试用）

```bash
# 只处理前 2 个 PDF 进行测试
python batch_index.py --max-pdfs 2
```

### 4. 自定义路径

```bash
python batch_index.py \
    --dataset-dir /path/to/pdfs \
    --output-dir /path/to/output \
    --report /path/to/report.json
```

---

## ⚙️ 配置说明

### 配置 1：使用 Hash Embedding（最快，质量较低）

```yaml
# config.yaml
embedding:
  backend: "hash"
  dim: 256
```

**适用场景**：快速测试、对比实验

### 配置 2：使用本地 Embedding（推荐）

```yaml
# config_batch_example.yaml
embedding:
  backend: "local"
  dim: 1024
  local:
    model_path: "/data/xwh/models/Qwen3-Embedding-0.6B"
    batch_size: 64      # 增大批处理提高速度
    use_fp16: true      # 开启 fp16 加速
```

**适用场景**：生产环境、批量处理

### 配置 3：使用 OpenAI Embedding（高质量，需联网）

```yaml
embedding:
  backend: "openai"
  dim: 1536
  openai:
    model: "text-embedding-3-small"
    # dimensions: 256  # 可选降维
```

**适用场景**：追求最高质量、PDF 数量少

---

## 📊 处理报告

处理完成后会生成 JSON 报告：

```json
{
  "total": 8,
  "success": 8,
  "failed": 0,
  "details": [
    {
      "doc_id": "DeepSeek-OCR-_Contexts_Optical_Compression",
      "pdf_path": "/data/xwh/VisRAG/dataset/DeepSeek-OCR- Contexts Optical Compression.pdf",
      "text": {"pages": 10, "success": true},
      "vision": {"pages": 10, "tokens_dir": "...", "success": true},
      "error": null
    }
  ]
}
```

---

## 🔍 索引后的使用

### 文本检索

```python
from src.pipeline import VisRAGPipeline

pipe = VisRAGPipeline("./output/chroma_db")
results = pipe.query_text("Transformer 架构", top_k=5)
```

### 视觉检索（以图搜图）

```python
results = pipe.query_vision_by_image("./query_image.png", top_k=5)
```

### 直接查询 ChromaDB

```python
from src.store import ChromaStore

store = ChromaStore("./output/chroma_db")

# 查看集合信息
collection = store.get_collection("text_pages", dim=256)
print(collection.count())  # 文档数量
```

---

## ⚠️ 注意事项

1. **显存占用**：
   - 本地 Embedding 模型需要 GPU 显存
   - Vision Encoder 需要较多显存
   - 如果显存不足，可以分批处理

2. **存储空间**：
   - Vision tokens 会占用较多磁盘空间
   - 每个 PDF 页约 1-5 MB
   - 确保 `output/vision_tokens/` 有足够空间

3. **处理时间估算**（取决于 GPU）：
   - Hash embedding: ~1 秒/10 页
   - Local embedding: ~5 秒/10 页
   - Vision tokens: ~10 秒/页

---

## 🧪 测试流程

```bash
# 1. 先测试 1 个 PDF
python batch_index.py --max-pdfs 1

# 2. 检查结果
ls output/vision_tokens/
ls output/chroma_db/
cat output/index_report.json

# 3. 测试查询
python main.py --persist output/chroma_db query_text --text "测试查询"

# 4. 没问题后处理全部
python batch_index.py
```
