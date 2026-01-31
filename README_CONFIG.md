# VisRAG 配置系统使用指南

VisRAG 现在支持通过 YAML 配置文件灵活选择：
- **Generator**（文本生成）：本地模型或 OpenAI API
- **Embedding**（文本向量）：Hash / 本地模型 / OpenAI API

这种设计让你可以灵活组合不同的组件进行对比实验，例如：
- 评估不同 embedding 对检索质量的影响
- 对比本地模型和 OpenAI API 的生成质量
- 快速切换配置进行 A/B 测试

## 快速开始

### 1. 使用本地模型（默认）

无需额外配置，默认使用 `config.yaml`：

```bash
# 使用默认本地模型
python main.py qa_build --dataset-dir ./dataset

# 或明确指定配置
python main.py --config config.yaml qa_build --dataset-dir ./dataset
```

### 2. 使用 OpenAI API

#### 方法一：环境变量（推荐）

```bash
# 设置 API Key
export OPENAI_API_KEY="sk-your-api-key"

# 可选：设置代理
export OPENAI_BASE_URL="https://api.openai.com/v1"

# 使用 OpenAI 配置
python main.py --config config_openai_example.yaml qa_build --dataset-dir ./dataset
```

#### 方法二：直接修改配置文件

编辑 `config_openai_example.yaml`：

```yaml
generator:
  backend: "openai"
  openai:
    api_key: "sk-your-api-key"  # ⚠️ 注意：不要提交到代码仓库！
    model: "gpt-4o-mini"
```

### 3. 测试 Generator

```bash
# 测试本地模型
python main.py test_generator --user "你好，请介绍一下自己"

# 测试 OpenAI
export OPENAI_API_KEY="sk-xxx"
python main.py --config config_openai_example.yaml test_generator --user "你好"
```

## 配置项说明

### generator.backend

- `"local"` - 使用本地 HuggingFace 模型
- `"openai"` - 使用 OpenAI API

### generator.local（本地模型配置）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `model_path` | 模型路径或 HF 模型名 | `/data/xwh/models/Qwen3-1.7B` |
| `max_new_tokens` | 最大生成 token 数 | `768` |
| `load_in_4bit` | 是否 4-bit 量化（省显存） | `false` |
| `temperature` | 采样温度，0 表示贪婪 | `0.0` |
| `device` | 设备选择：`auto`/`cuda`/`cpu` | `auto` |

### generator.openai（OpenAI 配置）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `api_key` | API Key（也可环境变量） | `null` |
| `base_url` | API 代理地址 | `null` |
| `model` | 模型名称 | `gpt-4o-mini` |
| `max_tokens` | 最大生成 token 数 | `768` |
| `temperature` | 采样温度 | `0.0` |
| `timeout` | 请求超时（秒） | `60` |

## Embedding 配置

VisRAG 支持三种文本 embedding 后端：

| 后端 | 说明 | 适用场景 |
|------|------|----------|
| `hash` | MD5 hash（轻量级） | 快速原型、对比实验 |
| `local` | 本地 HuggingFace 模型 | 生产环境、数据隐私 |
| `openai` | OpenAI Embedding API | 高质量语义、云环境 |

### 配置示例

```yaml
embedding:
  backend: "local"
  dim: 1024
  
  local:
    model_path: "/data/xwh/models/Qwen3-Embedding-0.6B"
    device: "auto"
    batch_size: 32
    max_length: 512
    use_fp16: true
  
  openai:
    model: "text-embedding-3-small"
    # dimensions: 256  # 可选：降维
```

### 后端选择建议

| 场景 | 推荐后端 | 原因 |
|------|----------|------|
| 快速验证 | `hash` | 无需下载模型，秒开 |
| 本地部署 | `local` | 隐私安全，无网络依赖 |
| 高质量检索 | `openai` | text-embedding-3 系列效果最佳 |
| 对比实验 | `hash` vs `local` | 控制变量，证明 embedding 质量的影响 |

### 支持的模型

**本地模型：**
- `/data/xwh/models/Qwen3-Embedding-0.6B`
- `BAAI/bge-large-zh-v1.5`
- `sentence-transformers/all-MiniLM-L6-v2`

**OpenAI 模型：**
- `text-embedding-3-small` (推荐，1536 维)
- `text-embedding-3-large` (质量最高，3072 维)
- `text-embedding-ada-002` (旧版)

### 代码中使用

```python
from src.embedder import create_embedder, get_embedder

# 创建 embedder
embedder = create_embedder()

# 批量编码
texts = ["文本1", "文本2", "文本3"]
vectors = embedder.embed(texts)

# 单条编码
vector = embedder.embed_single("单个文本")

# 获取维度
dim = embedder.dim
```

## 日志配置

在 `config.yaml` 中可以配置日志行为：

```yaml
logging:
  # 日志级别: DEBUG, INFO, WARNING, ERROR
  level: "INFO"
  
  # 是否输出到文件
  file_output: true
  
  # 日志文件目录
  log_dir: "./logs"
  
  # 日志文件名（null 表示按日期自动生成: visrag_YYYYMMDD.log）
  log_filename: null
  
  # 是否显示函数名和行号（调试用）
  show_location: false
```

### 日志使用方式

```python
from src.utils import get_logger

# 获取 logger
logger = get_logger(__name__)

# 不同级别的日志
logger.debug("调试信息")
logger.info("一般信息")
logger.warning("警告")
logger.error("错误")
```

### 代码中使用 Logger

```python
from src.utils import LoggerMixin

class MyClass(LoggerMixin):
    def __init__(self):
        super().__init__()
        self.logger.info("MyClass 初始化完成")
        
    def do_something(self):
        self.logger.debug("执行操作...")
```

## 支持的 OpenAI 兼容模型

通过设置 `base_url` 和 `model`，可以接入其他大模型：

```yaml
# Azure OpenAI
generator:
  backend: "openai"
  openai:
    base_url: "https://your-resource.openai.azure.com/openai/deployments/your-deployment"
    api_key: "your-azure-key"
    model: "gpt-4"

# 第三方代理（如 api2d、closeai）
generator:
  backend: "openai"
  openai:
    base_url: "https://openai.api2d.net/v1"
    api_key: "fk-xxx"
    model: "gpt-4o-mini"

# 本地 vLLM / llama.cpp 服务
generator:
  backend: "openai"
  openai:
    base_url: "http://localhost:8000/v1"
    api_key: "not-needed"
    model: "qwen2-7b"
```

## 代码中使用

```python
from src.generator import create_generator, get_generator
from src.config import get_config, reload_config

# 使用当前配置创建 Generator
gen = create_generator()
response = gen.generate("你是助手", "你好")

# 重新加载配置
reload_config("./another_config.yaml")
gen2 = create_generator()  # 使用新配置
```

## 注意事项

1. **API Key 安全**：不要将包含真实 API Key 的配置文件提交到 Git
2. **环境变量优先**：环境变量 `OPENAI_API_KEY` 会覆盖配置文件中的值
3. **依赖安装**：
   - 使用 OpenAI 模式需要 `pip install openai`
   - 使用本地 embedding 模型需要 transformers + torch
4. **模型选择建议**：
   - QA 生成：`gpt-4o-mini`（性价比高）
   - 裁判评分：`gpt-4o`（质量高）
   - 文本检索：`text-embedding-3-small` 或本地 `Qwen3-Embedding-0.6B`

## 快速对比实验示例

```bash
# 实验 1: Hash embedding + 本地 Generator
python main.py --config config.yaml qa_build --dataset-dir ./dataset

# 实验 2: 本地 Embedding + 本地 Generator
# (修改 config.yaml: embedding.backend = "local")
python main.py --config config.yaml qa_build --dataset-dir ./dataset

# 实验 3: OpenAI Embedding + OpenAI Generator
python main.py --config config_openai_example.yaml qa_build --dataset-dir ./dataset
```
