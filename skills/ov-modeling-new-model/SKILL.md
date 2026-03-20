---
name: ov-modeling-new-model
description: |
  Add new model support to the OpenVINO GenAI Modeling framework. This skill guides the process of implementing explicit model expressions in C++ for LLM, VLM, Diffusion, and other GenAI models. Use this skill when: (1) User provides a HuggingFace transformers/diffusers PyTorch model and asks to implement it in the modeling framework, (2) User wants to add support for a new model architecture like Qwen, LLaMA, Mistral, etc., (3) User asks to create explicit model representation using the modeling infrastructure in D:\data\code\Openvino_new_arch_poc\openvino-new-arch\openvino.genai\src\cpp\src\modeling.
---

# OpenVINO Modeling - Add New Model Support

This skill provides a battle-tested guide for implementing new model architectures in the OpenVINO GenAI Modeling framework. It is based on the successful implementation of multiple models including Qwen3, SmolLM3, Youtu, and LFM2.

## CRITICAL PRINCIPLES

Before starting implementation, understand these hard-won lessons:

1. **Weight names MUST match exactly** - The OpenVINO Module hierarchy generates parameter paths that must exactly match HuggingFace safetensors weight names (except `layers.N` becomes `layers[N]`). A single wrong name causes "Unknown parameter" errors.

2. **Always read the PyTorch source first** - Never guess weight names, layer names, or forward pass logic from documentation alone. Different models use different conventions even for the same concepts.

3. **Use the direct factory function pattern** - Models are registered via public `create_xxx_model()` functions called from `safetensors_modeling.cpp`, NOT via ModelBuilder self-registration. This is a critical distinction.

4. **Follow the `ops::` function signatures exactly** - Check the actual header files for argument order. For example, `ops::slice(data, start, stop, step, axis)` takes 5 args, not 4.

5. **Build and test incrementally** - Compile after each major component. Don't write the entire model before compiling.

## Prerequisites

User should provide:
1. **HuggingFace PyTorch implementation** - The transformers/diffusers source code for reference
2. **SafeTensors model folder** - Downloaded model weights with `config.json`

## Workflow Overview

```
1. Analyze PyTorch Implementation & Weight Names
         |
         v
2. Design Module Hierarchy (match weight paths!)
         |
         v
3. Create Config Struct + Header (.hpp)
         |
         v
4. Implement Module Classes (.cpp)
         |
         v
5. Create Public Factory Function
         |
         v
6. Register in safetensors_modeling.cpp (3 touch points)
         |
         v
7. Add HFConfig parsing for new fields
         |
         v
8. Write Dummy Unit Tests
         |
         v
9. Build, Fix Compilation Errors, Run Tests
         |
         v
10. E2E Validation with Real Model
```

---

## Step 1: Analyze PyTorch Implementation

This is THE MOST IMPORTANT step. Thoroughly read the HuggingFace PyTorch code. Getting weight names wrong is the #1 source of bugs.

### What to Extract

**1. Weight Parameter Names** - The EXACT names PyTorch uses:
```python
# Example: Different models use DIFFERENT naming conventions!
# Qwen3:  self_attn.o_proj.weight, mlp.gate_proj.weight, input_layernorm.weight
# LFM2:   self_attn.out_proj.weight, feed_forward.w1.weight, operator_norm.weight
# Youtu:  self_attn.q_a_proj.weight (MLA-specific)
```

**2. Layer Hierarchy** - How modules are nested:
```python
# Extract the __init__ method to see module names
class ModelDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        self.self_attn = ModelAttention(...)     # -> "self_attn" prefix
        self.mlp = ModelMLP(...)                 # -> "mlp" prefix
        self.input_layernorm = RMSNorm(...)      # -> "input_layernorm" prefix
```

**3. Forward Pass Logic** - Residual connection patterns:
```python
# Pattern A (Qwen3): Fused norm+residual
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)
hidden_states = self.self_attn(hidden_states) + residual

# Pattern B (LFM2): Simple add
hidden_states = self.self_attn(self.operator_norm(hidden_states)) + hidden_states
hidden_states = self.feed_forward(self.ffn_norm(hidden_states)) + hidden_states
```

**4. Special Architecture Features**:
- Hybrid layers (attention + conv alternating)
- Per-head normalization (Q/K norms per head_dim, not hidden_size)
- Non-standard layer names
- Conditional construction based on layer type

### Key Files to Read

```
# In HuggingFace transformers:
transformers/models/<model>/modeling_<model>.py    # Main implementation
transformers/models/<model>/modular_<model>.py     # Modular version (if exists)
transformers/models/<model>/configuration_<model>.py  # Config class

# The actual model:
<model_dir>/config.json                            # Runtime configuration
```

### Weight Name Mapping Rule

HuggingFace safetensors use dot notation for layers: `model.layers.0.self_attn.q_proj.weight`

OpenVINO Module hierarchy uses bracket notation: `model.layers[0].self_attn.q_proj.weight`

**The weight loader automatically handles `layers.N` -> `layers[N]` conversion.** All other name segments MUST match exactly.

### Common Naming Differences Between Models

| Concept | Qwen3/LLaMA | LFM2 | Youtu |
|---------|------------|------|-------|
| Output projection | `o_proj` | `out_proj` | `o_proj` |
| Gate projection | `mlp.gate_proj` | `feed_forward.w1` | `mlp.gate_proj` |
| Up projection | `mlp.up_proj` | `feed_forward.w3` | `mlp.up_proj` |
| Down projection | `mlp.down_proj` | `feed_forward.w2` | `mlp.down_proj` |
| Pre-attn norm | `input_layernorm` | `operator_norm` | `input_layernorm` |
| Post-attn norm | `post_attention_layernorm` | `ffn_norm` | `post_attention_layernorm` |
| Final norm | `norm` | `embedding_norm` | `norm` |
| Q/K norms | N/A | `q_layernorm`, `k_layernorm` | N/A |

---

## Step 2: Design Module Hierarchy

Map PyTorch modules to OpenVINO Modeling framework classes. **The module names you choose in constructors directly determine weight paths.**

### Standard LLM Pattern

```
PyTorch                           Modeling Framework (C++)
---------------------------------------------------------------
ModelForCausalLM                  ModelForCausalLM : Module("")
+-- model                        +-- ModelModel : Module("model")
|   +-- embed_tokens              |   +-- VocabEmbedding("embed_tokens")
|   +-- layers[]                  |   +-- vector<DecoderLayer>("layers[i]")
|   |   +-- self_attn             |   |   +-- Attention("self_attn")
|   |   +-- mlp                   |   |   +-- MLP("mlp")
|   |   +-- input_layernorm       |   |   +-- RMSNorm("input_layernorm")
|   |   +-- post_attention_layernorm    +-- RMSNorm("post_attention_layernorm")
|   +-- norm                      |   +-- RMSNorm("norm")
+-- lm_head                       +-- LMHead("lm_head")
```

### Hybrid Architecture Pattern (e.g., LFM2)

```
ModelForCausalLM : Module("")
+-- ModelModel : Module("model")
|   +-- VocabEmbedding("embed_tokens")
|   +-- vector<DecoderLayer>("layers[i]")
|   |   +-- [CONDITIONAL] Attention("self_attn") OR ShortConv("conv")
|   |   +-- MLP("feed_forward")
|   |   +-- RMSNorm("operator_norm")
|   |   +-- RMSNorm("ffn_norm")
|   +-- RMSNorm("embedding_norm")    # Note: NOT "norm"
+-- LMHead("lm_head")
```

For conditional layer types, use `std::unique_ptr`:
```cpp
class DecoderLayer : public Module {
    std::unique_ptr<Attention> self_attn_;   // Created only for attention layers
    std::unique_ptr<ShortConv> conv_;        // Created only for conv layers
    MLP feed_forward_;                       // Always present
    RMSNorm operator_norm_;                  // Always present
    RMSNorm ffn_norm_;                       // Always present
    bool is_attention_layer_;
};
```

---

## Step 3: Create Config Struct + Header

### File: `modeling/models/<model>/modeling_<model>.hpp`

```cpp
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <openvino/openvino.hpp>

#include "modeling/module.hpp"
#include "modeling/builder_context.hpp"
#include "modeling/layers/rms_norm.hpp"
#include "modeling/layers/vocab_embedding.hpp"
#include "modeling/layers/lm_head.hpp"
#include "modeling/weights/weight_source.hpp"
#include "modeling/weights/weight_finalizer.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

struct MyModelConfig {
    std::string architecture = "mymodel";
    int32_t hidden_size = 0;
    int32_t num_attention_heads = 0;
    int32_t num_key_value_heads = 0;        // For GQA; 0 = same as num_attention_heads
    int32_t head_dim = 0;                    // 0 = hidden_size / num_attention_heads
    int32_t intermediate_size = 0;           // FFN intermediate dimension
    int32_t num_hidden_layers = 0;
    int32_t vocab_size = 0;
    int32_t max_position_embeddings = 0;
    float rms_norm_eps = 1e-6f;              // Normalization epsilon
    float rope_theta = 10000.0f;             // RoPE base frequency
    std::string hidden_act = "silu";         // Activation function
    bool attention_bias = false;             // QKV projection bias
    bool tie_word_embeddings = false;        // Share embed/lm_head weights
    // Add model-specific fields here
};

// Forward declarations for all classes
class MyModelAttention;
class MyModelMLP;
class MyModelDecoderLayer;
class MyModelModel;
class MyModelForCausalLM;

// ... class declarations (see Step 4) ...

// PUBLIC factory function - this is what safetensors_modeling.cpp calls
std::shared_ptr<ov::Model> create_mymodel_model(
    const MyModelConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
```

### Config Field Guidelines

- Include ALL fields needed by the model, not just what seems "standard"
- Use the SAME field names as `config.json` where practical
- Provide sensible defaults matching the target model
- For fields with different names between HF and your struct, document the mapping

---

## Step 4: Implement Module Classes

### File: `modeling/models/<model>/modeling_<model>.cpp`

### 4.1 Attention Module

```cpp
#include "modeling/models/<model>/modeling_<model>.hpp"
#include "modeling/ops/llm.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/ops/kv_cache.hpp"
#include "modeling/ops/shape.hpp"
#include "modeling/weights/weight_loader.hpp"

MyModelAttention::MyModelAttention(BuilderContext& ctx,
                                    const std::string& name,
                                    const MyModelConfig& cfg,
                                    Module* parent)
    : Module(name, ctx, parent),
      num_heads_(cfg.num_attention_heads),
      num_kv_heads_(cfg.num_key_value_heads > 0 ? cfg.num_key_value_heads : cfg.num_attention_heads),
      head_dim_(cfg.head_dim > 0 ? cfg.head_dim : (cfg.hidden_size / cfg.num_attention_heads)),
      hidden_size_(cfg.hidden_size),
      scaling_(1.0f / std::sqrt(static_cast<float>(head_dim_))) {

    // Parameter names MUST match PyTorch weight names exactly
    q_proj_param_ = &register_parameter("q_proj.weight");
    k_proj_param_ = &register_parameter("k_proj.weight");
    v_proj_param_ = &register_parameter("v_proj.weight");
    o_proj_param_ = &register_parameter("o_proj.weight");  // or "out_proj.weight" for LFM2!

    // Optional: attention bias
    if (cfg.attention_bias) {
        q_bias_param_ = &register_parameter("q_proj.bias");
        k_bias_param_ = &register_parameter("k_proj.bias");
        v_bias_param_ = &register_parameter("v_proj.bias");
        o_bias_param_ = &register_parameter("o_proj.bias");
    }

    // Optional: per-head normalization (e.g., LFM2)
    // q_norm_ = RMSNorm(ctx, "q_layernorm", cfg.norm_eps, this);
    // k_norm_ = RMSNorm(ctx, "k_layernorm", cfg.norm_eps, this);
}

Tensor MyModelAttention::forward(const Tensor& hidden_states,
                                  const Tensor& beam_idx,
                                  const Tensor& rope_cos,
                                  const Tensor& rope_sin,
                                  const Tensor& attention_mask) const {
    // 1. Linear projections
    auto q = ops::linear(hidden_states, q_proj_weight());
    auto k = ops::linear(hidden_states, k_proj_weight());
    auto v = ops::linear(hidden_states, v_proj_weight());

    // 2. Reshape: [batch, seq, heads*head_dim] -> [batch, heads, seq, head_dim]
    auto q_heads = q.reshape({0, 0, num_heads_, head_dim_}).permute({0, 2, 1, 3});
    auto k_heads = k.reshape({0, 0, num_kv_heads_, head_dim_}).permute({0, 2, 1, 3});
    auto v_heads = v.reshape({0, 0, num_kv_heads_, head_dim_}).permute({0, 2, 1, 3});

    // 3. Optional: Per-head RMSNorm (e.g., LFM2)
    // q_heads = q_norm_.forward(q_heads);
    // k_heads = k_norm_.forward(k_heads);

    // 4. Apply RoPE
    auto* policy = &ctx().op_policy();
    auto q_rot = ops::llm::apply_rope(q_heads, rope_cos, rope_sin, head_dim_, policy);
    auto k_rot = ops::llm::apply_rope(k_heads, rope_cos, rope_sin, head_dim_, policy);

    // 5. Append to KV cache
    const std::string cache_prefix = full_path().empty() ? name() : full_path();
    auto cached = ops::append_kv_cache(k_rot, v_heads, beam_idx,
                                       num_kv_heads_, head_dim_, cache_prefix, ctx());

    // 6. Repeat KV heads for GQA
    auto k_expanded = ops::llm::repeat_kv(cached.first, num_heads_, num_kv_heads_, head_dim_);
    auto v_expanded = ops::llm::repeat_kv(cached.second, num_heads_, num_kv_heads_, head_dim_);

    // 7. Build causal mask with attention_mask integration
    auto mask = ops::llm::build_kv_causal_mask_with_attention(q_rot, k_expanded, attention_mask);

    // 8. Scaled dot-product attention
    auto context = ops::llm::sdpa(q_rot, k_expanded, v_expanded, scaling_, 3, &mask, false, policy);

    // 9. Merge heads and output projection
    const int64_t attn_out_dim = static_cast<int64_t>(num_heads_) * head_dim_;
    auto merged = context.permute({0, 2, 1, 3}).reshape({0, 0, attn_out_dim});
    return ops::linear(merged, o_proj_weight());
}
```

### 4.2 MLP Module

```cpp
// Standard SwiGLU MLP (Qwen3, LLaMA style)
// Parameter names: "gate_proj.weight", "up_proj.weight", "down_proj.weight"
//
// For LFM2-style naming: "w1.weight", "w3.weight", "w2.weight"
// Where w1=gate, w3=up, w2=down
MyModelMLP::MyModelMLP(BuilderContext& ctx, const std::string& name,
                        const MyModelConfig& cfg, Module* parent)
    : Module(name, ctx, parent) {
    gate_proj_param_ = &register_parameter("gate_proj.weight");  // or "w1.weight"
    up_proj_param_ = &register_parameter("up_proj.weight");      // or "w3.weight"
    down_proj_param_ = &register_parameter("down_proj.weight");  // or "w2.weight"
}

Tensor MyModelMLP::forward(const Tensor& x) const {
    // SwiGLU: down(silu(gate(x)) * up(x))
    auto gate = ops::linear(x, gate_proj_weight());
    auto up = ops::linear(x, up_proj_weight());
    auto gated = ops::silu(gate) * up;
    return ops::linear(gated, down_proj_weight());
}
```

### 4.3 Decoder Layer

```cpp
// Choose the residual connection pattern matching your target model!

// Pattern A: Simple residual (LFM2 style)
Tensor MyModelDecoderLayer::forward(const Tensor& hidden_states,
                                     const Tensor& beam_idx,
                                     const Tensor& rope_cos,
                                     const Tensor& rope_sin,
                                     const Tensor& attention_mask) const {
    auto residual = hidden_states;
    auto normed = input_layernorm_.forward(hidden_states);
    auto attn_out = self_attn_->forward(normed, beam_idx, rope_cos, rope_sin, attention_mask);
    auto after_attn = attn_out + residual;

    auto ffn_normed = post_attention_layernorm_.forward(after_attn);
    auto mlp_out = mlp_.forward(ffn_normed);
    return mlp_out + after_attn;
}

// Pattern B: Fused residual (Qwen3 style, uses RMSNorm's 2-output variant)
std::pair<Tensor, Tensor> MyModelDecoderLayer::forward(
    const Tensor& hidden_states,
    const Tensor& beam_idx,
    const Tensor& rope_cos,
    const Tensor& rope_sin,
    const Tensor& attention_mask,
    const std::optional<Tensor>& residual) const {

    Tensor normed, next_residual;
    if (residual) {
        auto norm_out = input_layernorm_.forward(hidden_states, *residual);
        normed = norm_out.first;
        next_residual = norm_out.second;
    } else {
        normed = input_layernorm_.forward(hidden_states);
        next_residual = hidden_states;
    }

    auto attn_out = self_attn_.forward(normed, beam_idx, rope_cos, rope_sin, attention_mask);
    auto post_norm = post_attention_layernorm_.forward(attn_out, next_residual);
    auto mlp_out = mlp_.forward(post_norm.first);

    return {mlp_out, post_norm.second};
}
```

### 4.4 Model and ForCausalLM

```cpp
MyModelModel::MyModelModel(BuilderContext& ctx, const MyModelConfig& cfg, Module* parent)
    : Module("model", ctx, parent),
      embed_tokens_(ctx, "embed_tokens", this),
      embedding_norm_(ctx, "norm", cfg.rms_norm_eps, this),  // or "embedding_norm" for LFM2
      head_dim_(cfg.head_dim > 0 ? cfg.head_dim
                                 : (cfg.hidden_size / cfg.num_attention_heads)),
      rope_theta_(cfg.rope_theta) {
    layers_.reserve(cfg.num_hidden_layers);
    for (int32_t i = 0; i < cfg.num_hidden_layers; ++i) {
        layers_.emplace_back(ctx, "layers[" + std::to_string(i) + "]", cfg, i, this);
    }
}

Tensor MyModelModel::forward(const Tensor& input_ids,
                              const Tensor& position_ids,
                              const Tensor& beam_idx,
                              const Tensor& attention_mask) {
    auto hidden_states = embed_tokens_.forward(input_ids);
    auto* policy = &ctx().op_policy();
    auto cos_sin = ops::llm::rope_cos_sin(position_ids, head_dim_, rope_theta_, policy);

    for (auto& layer : layers_) {
        hidden_states = layer.forward(hidden_states, beam_idx,
                                      cos_sin.first, cos_sin.second, attention_mask);
    }

    hidden_states = embedding_norm_.forward(hidden_states);
    return hidden_states;
}

// ForCausalLM wraps Model + LMHead
MyModelForCausalLM::MyModelForCausalLM(BuilderContext& ctx, const MyModelConfig& cfg,
                                        Module* parent)
    : Module("", ctx, parent),   // Root module has empty name!
      cfg_(cfg),
      model_(ctx, cfg, this),
      lm_head_(ctx, "lm_head", this) {
    if (cfg_.tie_word_embeddings) {
        lm_head_.tie_to(model_.embed_tokens().weight_param());
    }
}
```

---

## Step 5: Create Public Factory Function

**IMPORTANT:** The factory function must be a public (non-static, non-anonymous-namespace) function declared in the header.

```cpp
// At the end of modeling_<model>.cpp
std::shared_ptr<ov::Model> create_mymodel_model(
    const MyModelConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {

    BuilderContext ctx;
    MyModelForCausalLM model(ctx, cfg);

    // Load weights from source
    ov::genai::modeling::weights::load_model(model, source, finalizer);

    // Create input parameters (standard for all LLMs)
    auto input_ids = ctx.parameter("input_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto attention_mask = ctx.parameter("attention_mask", ov::element::i64, ov::PartialShape{-1, -1});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto beam_idx = ctx.parameter("beam_idx", ov::element::i32, ov::PartialShape{-1});

    // Forward pass
    auto logits = model.forward(input_ids, position_ids, beam_idx, attention_mask);

    // Build OV model
    auto result = std::make_shared<ov::op::v0::Result>(logits.output());
    result->output(0).set_names({"logits"});
    result->set_friendly_name("logits");
    return ctx.build_model({result->output(0)});
}
```

**DO NOT use self-registration pattern like this (it does NOT work):**
```cpp
// WRONG - DO NOT DO THIS
static bool registered = []() {
    ModelBuilder::instance().register_architecture("mymodel", build_mymodel);
    return true;
}();
```

---

## Step 6: Register in safetensors_modeling.cpp

This requires modifications at **3 locations** in `safetensors_utils/safetensors_modeling.cpp`:

### Location 1: Include the model header

```cpp
// Near the top of the file, with other model includes
#include "modeling/models/qwen3/modeling_qwen3.hpp"
#include "modeling/models/qwen3_moe/modeling_qwen3_moe.hpp"
// ... other models ...
#include "modeling/models/mymodel/modeling_mymodel.hpp"   // ADD THIS
```

### Location 2: Force modeling API flag (if needed)

```cpp
// In the load function, find the force_modeling_api line:
const bool force_modeling_api = (model_type == "qwen3_next" || model_type == "lfm2"
                                 || model_type == "mymodel");  // ADD YOUR MODEL

// Also add to the modeling API support list:
if (((model_type == "qwen3" || model_type == "qwen3_moe" || model_type == "qwen3_next"
      || model_type == "smollm3" || model_type == "youtu_llm" || model_type == "lfm2"
      || model_type == "mymodel") &&    // ADD YOUR MODEL
     (use_modeling_api() || force_modeling_api))) {
```

### Location 3: Config mapping else-if block

Add a new `else if` block in the `create_model_with_modeling_api` function:

```cpp
} else if (hf_config.model_type == "mymodel") {
    ov::genai::modeling::models::MyModelConfig cfg;
    cfg.architecture = hf_config.model_type;
    cfg.hidden_size = hf_config.hidden_size;
    cfg.intermediate_size = hf_config.intermediate_size;
    cfg.num_hidden_layers = hf_config.num_hidden_layers;
    cfg.num_attention_heads = hf_config.num_attention_heads;
    cfg.num_key_value_heads = hf_config.kv_heads();
    cfg.head_dim = hf_config.head_size();
    cfg.vocab_size = hf_config.vocab_size;
    cfg.max_position_embeddings = hf_config.max_position_embeddings;
    cfg.rope_theta = hf_config.rope_theta;
    cfg.rms_norm_eps = hf_config.rms_norm_eps;
    cfg.hidden_act = hf_config.hidden_act;
    cfg.tie_word_embeddings = !has_key("lm_head.weight");
    // Add model-specific field mappings here
    ov_model = ov::genai::modeling::models::create_mymodel_model(cfg, source, finalizer);
```

### Important Notes on Config Mapping

- `cfg.tie_word_embeddings = !has_key("lm_head.weight")` - This checks if the actual weight exists in safetensors, which is more reliable than reading the config.json field
- `cfg.attention_bias = has_key("model.layers[0].self_attn.q_proj.bias")` - Detects bias presence from actual weights
- For models with non-standard config field names, use the appropriate HFConfig field:
  - `hf_config.rms_norm_eps` for standard models
  - `hf_config.norm_eps` for models like LFM2 that use `norm_eps` instead

---

## Step 7: Add HFConfig Parsing for New Fields

If your model has config.json fields not already in `HFConfig`, add them.

### File: `safetensors_utils/hf_config.hpp`

```cpp
struct HFConfig {
    // ... existing fields ...

    // MyModel-specific fields
    int my_special_param = 0;
    std::vector<std::string> my_layer_types;
};
```

### File: `safetensors_utils/hf_config.cpp`

In `load_hf_config()`:
```cpp
// MyModel-specific fields
config.my_special_param = extract_int(json, "my_special_param", 0);
config.my_layer_types = extract_string_array(json, "my_layer_types");
```

### Available JSON Extraction Helpers

```cpp
std::string extract_string(json, key);                    // String values
int extract_int(json, key, default_value);                // Integer values
float extract_float(json, key, default_value);            // Float values
bool extract_bool(json, key, default_value);              // Boolean values
std::vector<std::string> extract_string_array(json, key); // String arrays
std::vector<int> extract_int_array(json, key);            // Integer arrays
std::string extract_raw_value(json, key);                 // Raw JSON (for objects)
```

---

## Step 8: Write Dummy Unit Tests

### File: `modeling/tests/mymodel_dummy_test.cpp`

```cpp
#include <gtest/gtest.h>
#include <openvino/openvino.hpp>
#include "modeling/models/mymodel/modeling_mymodel.hpp"
#include "modeling/tests/test_utils.hpp"

using namespace ov::genai::modeling;
namespace test_utils = ov::genai::modeling::tests;

TEST(MyModelDummy, BuildsAndRuns) {
    // Small test dimensions for fast compilation
    const size_t batch = 1, seq_len = 3, vocab = 128;
    const size_t hidden = 64, num_heads = 4, num_kv_heads = 2;
    const size_t head_dim = hidden / num_heads;  // = 16
    const size_t intermediate = 128;
    const size_t num_layers = 2;
    const size_t kv_dim = num_kv_heads * head_dim;  // = 32

    // Create config
    models::MyModelConfig cfg;
    cfg.hidden_size = hidden;
    cfg.num_attention_heads = num_heads;
    cfg.num_key_value_heads = num_kv_heads;
    cfg.head_dim = head_dim;
    cfg.intermediate_size = intermediate;
    cfg.num_hidden_layers = num_layers;
    cfg.vocab_size = vocab;
    cfg.rms_norm_eps = 1e-5f;
    cfg.rope_theta = 10000.0f;
    cfg.tie_word_embeddings = false;

    // Create dummy weights - NAMES MUST MATCH EXACTLY
    test_utils::DummyWeightSource weights;
    test_utils::DummyWeightFinalizer finalizer;

    auto make_w = [](size_t rows, size_t cols, float offset) {
        return test_utils::make_tensor(
            test_utils::make_seq(rows * cols, offset * 0.001f, 0.001f),
            {rows, cols});
    };
    auto make_norm = [](size_t dim, float offset) {
        auto data = std::vector<float>(dim, 1.0f);  // Init norm weights to 1.0
        return test_utils::make_tensor(data, {dim});
    };

    // Global weights
    weights.add("model.embed_tokens.weight", make_w(vocab, hidden, 1));
    weights.add("model.norm.weight", make_norm(hidden, 0));
    weights.add("lm_head.weight", make_w(vocab, hidden, 2));

    // Per-layer weights
    for (size_t i = 0; i < num_layers; ++i) {
        std::string p = "model.layers[" + std::to_string(i) + "].";
        float off = static_cast<float>(i) * 10;

        // Attention
        weights.add(p + "self_attn.q_proj.weight", make_w(hidden, hidden, off + 1));
        weights.add(p + "self_attn.k_proj.weight", make_w(kv_dim, hidden, off + 2));
        weights.add(p + "self_attn.v_proj.weight", make_w(kv_dim, hidden, off + 3));
        weights.add(p + "self_attn.o_proj.weight", make_w(hidden, hidden, off + 4));

        // Norms
        weights.add(p + "input_layernorm.weight", make_norm(hidden, 0));
        weights.add(p + "post_attention_layernorm.weight", make_norm(hidden, 0));

        // MLP
        weights.add(p + "mlp.gate_proj.weight", make_w(intermediate, hidden, off + 5));
        weights.add(p + "mlp.up_proj.weight", make_w(intermediate, hidden, off + 6));
        weights.add(p + "mlp.down_proj.weight", make_w(hidden, intermediate, off + 7));
    }

    // Build model
    BuilderContext ctx;
    models::MyModelForCausalLM model(ctx, cfg);
    weights::load_model(model, weights, finalizer);

    // Create model graph
    auto input_ids = ctx.parameter("input_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto attention_mask = ctx.parameter("attention_mask", ov::element::i64, ov::PartialShape{-1, -1});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto beam_idx = ctx.parameter("beam_idx", ov::element::i32, ov::PartialShape{-1});

    auto logits = model.forward(input_ids, position_ids, beam_idx, attention_mask);
    auto result = std::make_shared<ov::op::v0::Result>(logits.output());
    auto ov_model = ctx.build_model({result->output(0)});

    // Compile and run
    ov::Core core;
    auto compiled = core.compile_model(ov_model, "CPU");
    auto request = compiled.create_infer_request();

    // Set inputs
    ov::Tensor input_ids_t(ov::element::i64, {batch, seq_len});
    auto* ids = input_ids_t.data<int64_t>();
    ids[0] = 1; ids[1] = 5; ids[2] = 10;

    ov::Tensor attn_mask_t(ov::element::i64, {batch, seq_len});
    auto* mask = attn_mask_t.data<int64_t>();
    mask[0] = 1; mask[1] = 1; mask[2] = 1;

    ov::Tensor pos_ids_t(ov::element::i64, {batch, seq_len});
    auto* pos = pos_ids_t.data<int64_t>();
    pos[0] = 0; pos[1] = 1; pos[2] = 2;

    ov::Tensor beam_t(ov::element::i32, {batch});
    beam_t.data<int32_t>()[0] = 0;

    request.set_tensor("input_ids", input_ids_t);
    request.set_tensor("attention_mask", attn_mask_t);
    request.set_tensor("position_ids", pos_ids_t);
    request.set_tensor("beam_idx", beam_t);

    // Infer
    request.infer();

    // Verify output shape: [batch, seq_len, vocab_size]
    auto output = request.get_output_tensor();
    EXPECT_EQ(output.get_shape(), (ov::Shape{batch, seq_len, vocab}));

    // Verify no NaN/Inf
    auto* out_data = output.data<float>();
    for (size_t i = 0; i < output.get_size(); ++i) {
        EXPECT_FALSE(std::isnan(out_data[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(out_data[i])) << "Inf at index " << i;
    }
}

TEST(MyModelDummy, MultiStepInference) {
    // Test prefill + decode to verify cache management
    // ... (similar setup but run multiple infer() calls with
    //      decreasing seq_len and advancing position_ids)
}
```

### Test is automatically included in the build

All `.cpp` files in the `tests/` directory are glob-matched by CMakeLists.txt:
```cmake
file(GLOB modeling_tests_src "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
```

No CMakeLists.txt changes needed for new test files.

---

## Step 9: Build and Debug

### Build Command

```bash
cd openvino.genai
cmake --build build --target test_modeling_api -j8
```

### Common Compilation Errors and Fixes

**Error: "Unknown parameter: model.xxx.weight"**
- Cause: Weight names in your Module hierarchy don't match safetensors weight names
- Fix: Read PyTorch source to find exact parameter names. Common mismatches:
  - `o_proj` vs `out_proj`
  - `mlp.gate_proj` vs `feed_forward.w1`
  - `input_layernorm` vs `operator_norm`
  - `norm` vs `embedding_norm`

**Error: "function does not take N arguments"**
- Cause: Wrong number of arguments to ops functions
- Fix: Check the actual function signature in the header. Key signatures:
  ```cpp
  // ops.hpp
  Tensor slice(const Tensor& data, int64_t start, int64_t stop, int64_t step, int64_t axis);
  Tensor gather(const Tensor& data, const Tensor& indices, int64_t axis);
  Tensor concat(const std::vector<Tensor>& xs, int64_t axis);
  Tensor linear(const Tensor& x, const Tensor& weight);

  // llm.hpp
  std::pair<Tensor, Tensor> rope_cos_sin(const Tensor& positions, int32_t head_dim,
                                          float rope_theta, const OpPolicy* policy);
  Tensor apply_rope(const Tensor& x, const Tensor& cos, const Tensor& sin,
                    int32_t head_dim, const OpPolicy* policy);
  Tensor sdpa(const Tensor& q, const Tensor& k, const Tensor& v, float scale,
              int64_t softmax_axis, const Tensor* mask, bool causal, const OpPolicy* policy);
  Tensor repeat_kv(const Tensor& x, int32_t num_heads, int32_t num_kv_heads, int32_t head_dim);

  // kv_cache.hpp
  std::pair<Tensor, Tensor> append_kv_cache(const Tensor& k, const Tensor& v,
      const Tensor& beam_idx, int32_t num_kv_heads, int32_t head_dim,
      const std::string& cache_prefix, const BuilderContext& ctx);
  ```

**Error: "Unsupported model architecture 'xxx'"**
- Cause: Model type not registered in safetensors_modeling.cpp
- Fix: Ensure all 3 registration locations are updated (include, force flag, else-if block)

**Error: Shape mismatch at runtime**
- Cause: Weight dimensions don't match expected shapes
- Fix: Verify weight shapes match PyTorch: `[out_features, in_features]` for Linear weights

### Run Tests

```bash
# Run all modeling tests
./build/tests/test_modeling_api

# Run specific test
./build/tests/test_modeling_api --gtest_filter="MyModelDummy.*"
```

---

## Step 10: E2E Validation with Real Model

Use the existing `greedy_causal_lm` sample:

```bash
./build/samples/cpp/text_generation/greedy_causal_lm \
    --model_dir /path/to/MyModel \
    --prompt "The quick brown fox" \
    --device CPU \
    --max_new_tokens 50
```

Expected:
- Model loads successfully (no "Unknown parameter" errors)
- Generates text (may not be coherent until numerical validation)
- No crashes, NaN, or Inf values

---

## Available Operations Reference

### Core Ops (`ops::`)
| Function | Signature | Notes |
|----------|-----------|-------|
| `linear` | `(x, weight)` | Matrix multiply: x @ weight^T |
| `matmul` | `(a, b, ta, tb)` | General matmul with transpose flags |
| `slice` | `(data, start, stop, step, axis)` | **5 args!** All int64_t |
| `gather` | `(data, indices, axis)` | Gather along axis |
| `concat` | `(tensors_vec, axis)` | Concatenate tensors |
| `silu` | `(x)` | SiLU activation |
| `rms` | `(x, weight, eps)` | RMS normalization |
| `const_scalar` | `(ctx, value)` | Create scalar constant |
| `const_vec` | `(ctx, values)` | Create vector constant |
| `constant` | `(tensor, ctx)` | Wrap ov::Tensor as constant |

### LLM Ops (`ops::llm::`)
| Function | Signature | Notes |
|----------|-----------|-------|
| `rope_cos_sin` | `(positions, head_dim, theta, policy)` | Compute RoPE embeddings |
| `apply_rope` | `(x, cos, sin, head_dim, policy)` | Apply RoPE to tensor |
| `sdpa` | `(q, k, v, scale, softmax_axis, mask, causal, policy)` | Attention |
| `repeat_kv` | `(x, num_heads, num_kv_heads, head_dim)` | GQA head expansion |
| `build_kv_causal_mask_with_attention` | `(q, k, attention_mask)` | Causal mask |
| `append_kv_cache` | `(k, v, beam_idx, num_kv_heads, head_dim, prefix, ctx)` | KV cache |

### Shape Ops (`shape::`)
| Function | Signature | Notes |
|----------|-----------|-------|
| `dim` | `(tensor, axis)` | Get dimension size |
| `of` | `(tensor)` | Get full shape |
| `make` | `({dim1, dim2, ...})` | Construct shape |
| `broadcast_to` | `(tensor, shape)` | Broadcast tensor |

### NN Ops (`ops::nn::`)
| Function | Signature | Notes |
|----------|-----------|-------|
| `gelu` | `(x, approximate)` | GELU activation |
| `layer_norm` | `(x, weight, bias, eps, axis)` | Layer normalization |
| `group_norm` | `(x, weight, bias, groups, eps)` | Group normalization |

### Tensor Methods
| Method | Notes |
|--------|-------|
| `.reshape({dims})` | Reshape; 0 = keep dim, -1 = infer |
| `.permute({order})` | Transpose dimensions |
| `.unsqueeze(axis)` | Add dimension |
| `.squeeze(axis)` | Remove dimension |
| `.to(dtype)` | Type conversion |
| `.softmax(axis)` | Softmax |
| `.mean(axis, keepdim)` | Mean reduction |
| `+`, `-`, `*`, `/` | Element-wise arithmetic |

---

## Advanced Patterns

### Depthwise Convolution with State Cache (LFM2/Qwen3-Next)

For models with 1D depthwise convolution layers that maintain state across inference steps:

```cpp
Tensor ShortConv::forward(const Tensor& x, const Tensor& beam_idx) const {
    auto* op_ctx = x.context();

    // 1. Project input to get gating components
    auto projected = ops::linear(x, in_proj_weight());
    auto projected_t = projected.permute({0, 2, 1});  // [batch, channels, seq]

    // 2. Split along channel dimension
    auto B = ops::slice(projected_t, int64_t(0), int64_t(hidden), int64_t(1), int64_t(1));
    auto C = ops::slice(projected_t, int64_t(hidden), int64_t(2*hidden), int64_t(1), int64_t(1));
    auto x_gate = ops::slice(projected_t, int64_t(2*hidden), int64_t(3*hidden), int64_t(1), int64_t(1));

    // 3. Pre-conv gating
    auto Bx = B * x_gate;

    // 4. Conv state cache (Variable/ReadValue/Assign pattern)
    auto conv_shape = shape::make({batch, hidden_const, cache_const});
    auto conv_init = shape::broadcast_to(
        Tensor(ops::const_scalar(op_ctx, 0.0f), op_ctx).to(x.dtype()), conv_shape);

    ov::op::util::VariableInfo info{ov::PartialShape{-1, hidden_, cache_len_},
                                    x.dtype(), full_path() + ".conv_state"};
    auto var = std::make_shared<ov::op::util::Variable>(info);
    auto read = std::make_shared<ov::op::v6::ReadValue>(conv_init.output(), var);
    auto cached_state = ops::gather(Tensor(read->output(0), op_ctx), beam_idx, 0);

    // 5. Concatenate state + new input
    auto input_with_state = ops::concat({cached_state, Bx}, 2);

    // 6. GroupConvolution (depthwise)
    auto weight_4d = conv_weight().reshape({hidden_, 1, 1, kernel_}, false);
    auto conv = std::make_shared<ov::op::v1::GroupConvolution>(
        input_with_state.output(), weight_4d.output(),
        ov::Strides{1}, ov::CoordinateDiff{0}, ov::CoordinateDiff{0}, ov::Strides{1});

    // 7. Slice output and update state
    // ... (use ov::op::v8::Slice for dynamic slicing)

    // 8. Register state update as sink
    auto assign = std::make_shared<ov::opset13::Assign>(new_state.output(), var);
    ctx().register_sink(assign);

    // 9. Post-conv gating and output projection
    auto y = C * conv_output;
    return ops::linear(y.permute({0, 2, 1}), out_proj_weight());
}
```

### Config with Full Attention Index Resolution

When models specify attention layers by index instead of explicit type list:

```cpp
// In safetensors_modeling.cpp config mapping:
cfg.layer_types = hf_config.layer_types;
if (cfg.layer_types.empty()) {
    std::set<int> attn_set(hf_config.full_attn_idxs.begin(),
                           hf_config.full_attn_idxs.end());
    if (attn_set.empty()) {
        // Default: all layers are attention
        for (int32_t i = 0; i < cfg.num_hidden_layers; ++i)
            attn_set.insert(i);
    }
    for (int32_t i = 0; i < cfg.num_hidden_layers; ++i) {
        cfg.layer_types.push_back(
            attn_set.count(i) ? "full_attention" : "conv");
    }
}
```

### Weight Tying

```cpp
// In ForCausalLM constructor:
if (cfg.tie_word_embeddings) {
    lm_head_.tie_to(model_.embed_tokens().weight_param());
}

// In safetensors_modeling.cpp:
cfg.tie_word_embeddings = !has_key("lm_head.weight");
// ^^ Detects from actual weights, more reliable than config.json
```

---

## Key Lessons from LFM2 Implementation

These are critical issues encountered during real implementation:

### 1. Weight Names Are Model-Specific - NEVER Assume Standard Names
LFM2 uses completely different naming than Qwen3/LLaMA:
- `out_proj` not `o_proj`
- `feed_forward.w1/w2/w3` not `mlp.gate_proj/up_proj/down_proj`
- `operator_norm`/`ffn_norm` not `input_layernorm`/`post_attention_layernorm`
- `embedding_norm` not `norm`

**Always read the PyTorch source to find the exact weight names.**

### 2. The Registration Pattern is NOT Self-Registration
The correct pattern is a public factory function called directly from `safetensors_modeling.cpp`. The `ModelBuilder::register_architecture()` pattern does NOT work for this integration path.

### 3. Two Config Formats for Layer Types
LFM2-1.2B uses `full_attn_idxs: [2,5,8,10,12,14]` while LFM2.5-1.2B-Instruct uses explicit `layer_types: ["conv", "conv", "full_attention", ...]`. Your code must handle both.

### 4. ops::slice Takes 5 Arguments, Not 4
```cpp
// WRONG: ops::slice(tensor, axis, start, stop)
// RIGHT: ops::slice(tensor, start, stop, step, axis)
auto sliced = ops::slice(tensor, int64_t(0), int64_t(hidden), int64_t(1), int64_t(1));
```

### 5. RMSNorm Dimension Matters
Per-head normalization (like LFM2's q_layernorm/k_layernorm) operates on `head_dim`, not `hidden_size`. The RMSNorm weight shape is `[head_dim]`, and it's applied on the last dimension of `[batch, heads, seq, head_dim]` tensors.

### 6. Residual Connection Patterns Differ
Qwen3 uses a fused norm+residual pattern (RMSNorm returns 2 values). LFM2 uses simple addition. Match the PyTorch implementation exactly.

### 7. Conv Weight Reshaping
PyTorch Conv1d weight shape is `[out_channels, in_channels/groups, kernel_size]` = `[hidden, 1, kernel]`.
OpenVINO GroupConvolution expects `[groups, out_per_group, in_per_group, kernel]` = `[hidden, 1, 1, kernel]`.
Reshape in forward pass: `conv_weight.reshape({hidden, 1, 1, kernel}, false)`.

### 8. Dynamic Slicing Requires OV Ops
For slicing with dynamic dimensions (e.g., taking last N elements of a dynamically-sized tensor), use `ov::op::v8::Slice` directly instead of `ops::slice` (which only handles static int64_t indices).

---

## File Checklist

When adding a new model, these are ALL the files you need to create/modify:

### New Files to Create
| File | Purpose |
|------|---------|
| `modeling/models/<model>/modeling_<model>.hpp` | Config struct + class declarations + factory function declaration |
| `modeling/models/<model>/modeling_<model>.cpp` | All implementations + factory function |
| `modeling/tests/<model>_dummy_test.cpp` | Unit tests with synthetic weights |

### Existing Files to Modify
| File | Change |
|------|--------|
| `safetensors_utils/safetensors_modeling.cpp` | 3 locations: include, force flag, config mapping |
| `safetensors_utils/hf_config.hpp` | Add model-specific config fields (if needed) |
| `safetensors_utils/hf_config.cpp` | Parse new config.json fields (if needed) |

### Files That Do NOT Need Changes
| File | Reason |
|------|--------|
| `modeling/CMakeLists.txt` | Sources are glob-matched automatically |
| `modeling/tests/CMakeLists.txt` | Tests are glob-matched automatically |
| `loaders/model_builder.hpp` | NOT used for this registration pattern |

---

## Reference: Existing Model Implementations

Use these as templates:

| Model | Directory | Pattern | Special Features |
|-------|-----------|---------|------------------|
| Qwen3 Dense | `models/qwen3/` | Standard LLM | Fused norm+residual, attention bias |
| Qwen3 MoE | `models/qwen3_moe/` | MoE | Expert routing, top-k selection |
| Qwen3 Next | `models/qwen3_next/` | Hybrid | Depthwise conv + MoE + attention |
| SmolLM3 | `models/smollm3/` | Standard LLM | MLP bias support |
| Youtu | `models/youtu/` | MLA | Multi-head latent attention, rope_interleave |
| LFM2 | `models/lfm2/` | Hybrid | Attention/conv alternating, per-head norms |
