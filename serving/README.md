# Qwen3.5 OpenAI-Compatible API Server

OpenVINO GenAI Modeling API 驱动的 Qwen3.5 serving，提供 OpenAI 兼容接口。

## 特性

- **OpenAI 兼容 API**: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`
- **Streaming SSE**: 逐 token 流式输出
- **Tool Calling**: Qwen3.5 XML → OpenAI tool_calls JSON 自动转换
- **INT4 量化**: 默认 int4_asym (group_size=128)，推理速度提升 ~2.7x
- **VL (Vision-Language)**: 通过 subprocess 支持图片理解
- **`<think>` 过滤**: 自动去除模型内部思考 token

## 快速启动

```powershell
# 1. 首次运行（从 safetensors 构建，~11s 启动）
.\run.bat --model C:\data\models\Huggingface\Qwen3.5-4B --device GPU

# 2. 生成 IR 缓存（可选，加速后续启动）
#    先用 exe 跑一次:
modeling_qwen3_5.exe --model C:\data\models\Huggingface\Qwen3.5-4B --mode text --prompt "hi" --output-tokens 2 --cache-model
#    这会在模型目录下生成 qwen3_5_text_q4a_b4a_g128.xml/.bin
#    然后创建 symlink:
cd C:\data\models\Huggingface\Qwen3.5-4B
mklink openvino_model.xml qwen3_5_text_q4a_b4a_g128.xml
mklink openvino_model.bin qwen3_5_text_q4a_b4a_g128.bin

# 3. 使用 IR 部署（~4.7s 启动，无需 Modeling API）
.\run.bat --model C:\data\models\Huggingface\Qwen3.5-4B --device GPU --quant none
```

## API 使用示例

```bash
# Chat
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "Hello!"}],
  "max_tokens": 100
}'

# Streaming
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "Tell me a joke"}],
  "max_tokens": 200,
  "stream": true
}'

# Tool calling
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "What is the weather in Beijing?"}],
  "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
  "max_tokens": 200
}'

# VL (图片理解)
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "Describe this image"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
  ]}],
  "max_tokens": 200
}'

# Health
curl http://localhost:8000/health
```

## CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | (必需) | HF 模型目录路径 |
| `--device` | GPU | OpenVINO 设备 |
| `--host` | 0.0.0.0 | 监听地址 |
| `--port` | 8000 | 端口 |
| `--workers` | 1 | LLMPipeline 实例数 |
| `--max-tokens` | 2048 | 默认最大生成 token |
| `--quant` | int4_asym | 量化模式 (int4_asym/int8_sym/none) |
| `--quant-group-size` | 128 | 量化 group size |
| `--quant-backup` | int4_asym | 备用量化模式 |
| `--model-name` | (自动) | 模型显示名称 |

## 架构

```
Client → FastAPI → Worker Pool (N × LLMPipeline B=1) → GPU
                 ↘ VL Backend (subprocess exe) → GPU
```

- **Worker Pool**: 每个 worker 独立 `LLMPipeline` + `InferRequest`，Variable state 隔离
- **VL Backend**: `modeling_qwen3_5.exe --mode vl` subprocess（VLMPipeline 不支持 Qwen3.5）
- **量化**: 在线 INT4 量化（从 safetensors 构建时），或直接加载预量化 IR

## IR 文件说明

| 文件 | 大小 | 说明 |
|------|------|------|
| `qwen3_5_text_q4a_b4a_g128.xml` | 3.1 MB | 模型结构 (int4_asym, gs128) |
| `qwen3_5_text_q4a_b4a_g128.bin` | 2.1 GB | 量化权重 |
| `openvino_model.xml/bin` | symlink | LLMPipeline 标准入口 |

## 测试

```powershell
# 启动服务器后
cd serving
python -m pytest test_server.py -v --timeout=120    # 27 tests
python test_smoke.py                                  # 12 quick tests
```
