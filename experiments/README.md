# VisRAG 对比实验指南

## 实验目的

对比三种检索策略对问答质量的影响：
1. **纯文本检索 (Text Only)**: 只用文本 embedding 检索
2. **视觉检索 (Vision)**: 用页面描述进行视觉语义检索
3. **双路融合 (Fusion)**: 文本 + 视觉检索结果融合

---

## 实验流程

### 第一步：准备增强版索引

使用 `batch_index_enhanced.py` 生成所有需要的数据：

```bash
cd VisRAG

# 完整索引（包含视觉描述）
python batch_index_enhanced.py \
    --config config_batch_example.yaml \
    --dataset-dir ./dataset \
    --output-dir ./output \
    --max-pdfs 3  # 先测试3个PDF
```

这会生成：
- `output/chroma_db/text_pages` - 文本索引
- `output/chroma_db/vision_pages` - 视觉向量索引
- `output/chroma_db/vision_descriptions` - 页面描述索引（关键）
- `output/images/{doc_id}/page_{n}.png` - 页面图像
- `output/vision_tokens/{doc_id}/page_{n}.pt` - vision tokens

### 第二步：运行对比实验

```bash
cd VisRAG/experiments

# 基础对比（不使用 Decoder）
python compare_retrieval_methods.py \
    --config ../config.yaml \
    --persist-dir ../output/chroma_db \
    --query "Transformer架构的优势是什么？" \
    --top-k 3 \
    --output ./result_basic.json

# 增强对比（视觉检索使用 Decoder 解码图像）
python compare_retrieval_methods.py \
    --config ../config.yaml \
    --persist-dir ../output/chroma_db \
    --pdf-images-dir ../output/images \
    --use-decoder \
    --query "Figure 2展示了什么内容？" \
    --top-k 3 \
    --output ./result_with_decoder.json
```

### 第三步：批量测试多个问题

创建一个测试问题列表 `test_queries.txt`：

```
Transformer架构的核心优势是什么？
Figure 3中展示了哪些实验结果？
这篇论文的方法和传统方法有什么区别？
Table 2中的性能对比数据是什么？
```

然后批量运行：

```bash
while read query; do
    python compare_retrieval_methods.py \
        --persist-dir ../output/chroma_db \
        --query "$query" \
        --output "./results/$(echo $query | md5sum | cut -d' ' -f1).json"
done < test_queries.txt
```

---

## 预期对比维度

| 维度 | 纯文本 | 视觉检索 | 融合 |
|------|--------|----------|------|
| **图表问题** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **纯文本问题** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **召回率** | 中 | 中 | 高 |
| **精确率** | 高(文本) | 高(图表) | 高 |
| **速度** | 快 | 中 | 慢 |

---

## 关键观察指标

1. **检索相关性**
   - 文本检索召回的页面是否包含答案？
   - 视觉检索能否召回包含图表的页面？
   - 融合是否能综合两者优势？

2. **生成质量**
   - 三种方案的答案完整性
   - 对图表/公式的描述准确性
   - 幻觉（hallucination）程度

3. **失败案例分析**
   - 文本检索失败但视觉检索成功的案例
   - 视觉检索引入噪声的案例

---

## 快速开始命令

```bash
# 1. 进入目录
cd VisRAG

# 2. 创建增强索引（测试2个PDF）
python batch_index_enhanced.py --max-pdfs 2

# 3. 运行对比实验
cd experiments
python compare_retrieval_methods.py \
    --query "这篇论文的主要贡献是什么？" \
    --top-k 3
```

---

## 注意事项

1. **显存**: 使用 Decoder 需要较多显存，如果显存不足可以 `--skip-decoder`
2. **时间**: 生成描述会增加索引时间，但可以后续增量生成
3. **公平性**: 对比时确保使用相同的 generator 和相同的 top-k
