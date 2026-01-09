# Module 7: LoRA Fine-tuning

## What is LoRA?

**LoRA (Low-Rank Adaptation)** is a technique for efficiently fine-tuning large language models by:
1. Freezing all pretrained weights
2. Injecting small, trainable "adapter" matrices into transformer layers
3. Training only these adapters (0.1-1% of total parameters)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LORA ARCHITECTURE                               │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  TRANSFORMER LAYER                                               │   │
│  │                                                                   │   │
│  │     Input ───┬─────────────────┬─────────────────> Output        │   │
│  │              │                 │                                  │   │
│  │              ▼                 ▼                                  │   │
│  │     ┌────────────────┐   ┌──────────┐                            │   │
│  │     │ Original Weight│   │ LoRA     │                            │   │
│  │     │ W (frozen)     │   │ Adapter  │                            │   │
│  │     │ 1.1B params    │   │ A × B    │                            │   │
│  │     └────────────────┘   │ 2.2M     │                            │   │
│  │              │           │ params   │                            │   │
│  │              │           └──────────┘                            │   │
│  │              │                 │                                  │   │
│  │              └────────+────────┘                                  │   │
│  │                       │                                           │   │
│  │                       ▼                                           │   │
│  │                  W + A×B = Fine-tuned output                     │   │
│  │                                                                   │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Key Insight: Instead of updating W directly, we learn a low-rank      │
│  decomposition A × B that modifies the output.                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Why Use LoRA?

| Full Fine-tuning | LoRA Fine-tuning |
|------------------|------------------|
| Updates all 7B+ parameters | Updates ~2-10M parameters (0.2%) |
| Needs 40GB+ VRAM | Works on 4GB VRAM |
| Hours to train | Minutes to train |
| Saves 14GB+ model file | Saves 10-50MB adapter file |
| Risk of catastrophic forgetting | Preserves base model capabilities |

---

## Actual Results (Reproducible)

Run `python demo.py --full` to reproduce:

### Training Statistics

| Metric | Value |
|--------|-------|
| Base Model | TinyLlama-1.1B-Chat |
| Total Parameters | 1,100,048,384 |
| Trainable (LoRA) | 2,252,800 (0.20%) |
| Training Time | 3.8 minutes |
| GPU Memory Used | ~3.5GB |
| Training Loss | 12.1 → 8.1 |

### Task: Customer Service Tone Transformation

Input: Informal customer complaints
Output: Professional, empathetic responses

**Example:**
```
Input:  "my package is lost and nobody cares"

Base Model:
"I am sorry to hear that your package is lost. We take customer
service very seriously, and we are working to resolve this issue..."

Fine-tuned (3 epochs):
"Thank you for bringing this to our attention. We understand your
frustration..."
```

### Why Similar Results?

TinyLlama-Chat is already fine-tuned for helpful responses. Our small dataset (16 examples) and few epochs (3) show the training pipeline works, but larger improvements require:
- More training data (50-100+ examples)
- More epochs (10-20)
- Larger rank (r=16 or 32)
- A base model less suited to the task

---

## Implementation

### 1. Training Data (dataset.py)

```python
TRAINING_DATA = [
    {
        "input": "yo my order never showed up wtf",
        "output": "I sincerely apologize for the inconvenience with your
                   order delivery. I understand how frustrating this must be.
                   Let me look into this immediately..."
    },
    # ... 20 examples total
]

def format_for_training(example):
    return f"""### Instruction:
Transform this informal customer message into a professional response.

### Input:
{example['input']}

### Response:
{example['output']}"""
```

### 2. LoRA Configuration (trainer.py)

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,                    # Rank of low-rank matrices
    lora_alpha=16,          # Scaling factor (typically 2×r)
    lora_dropout=0.1,       # Dropout for regularization
    target_modules=[        # Which layers to adapt
        "q_proj",           # Query projection
        "v_proj",           # Value projection
        "k_proj",           # Key projection
        "o_proj",           # Output projection
    ],
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, lora_config)
# Trainable: 2,252,800 / 1,102,301,184 (0.20%)
```

### 3. Memory-Efficient Loading

```python
from transformers import BitsAndBytesConfig

# 4-bit quantization for 4GB VRAM
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quantization_config=quantization_config,
    device_map="auto",
)
```

---

## Running the Demo

```bash
cd /home/adam/LLMControl

# Full demo: train + compare
PYTHONPATH=src/python python src/python/07_lora_finetuning/demo.py --full

# Just training
PYTHONPATH=src/python python src/python/07_lora_finetuning/demo.py --train

# Test existing model
PYTHONPATH=src/python python src/python/07_lora_finetuning/demo.py --test
```

**Expected output:**
```
LORA FINE-TUNING DEMO
GPU: NVIDIA GeForce GTX 1650
Memory: 3.6GB free / 3.6GB total

Loading model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
Model loaded. Parameters: 1,100,048,384
Applying LoRA adapters...
Trainable parameters: 2,252,800 / 1,102,301,184 (0.20%)

Starting training...
{'train_runtime': 221.3s, 'train_loss': 11.24, 'epoch': 3.0}
Training complete.
```

---

## Files in This Module

| File | Description |
|------|-------------|
| `dataset.py` | Training/test data, formatting functions |
| `trainer.py` | LoRATrainer class, configuration, training loop |
| `demo.py` | Training and comparison demo |
| `lora_output/` | Saved adapter weights (after training) |
| `README.md` | This documentation |

---

## Key Hyperparameters

### LoRA-Specific

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `r` (rank) | 8 | 4-64 | Higher = more capacity, more VRAM |
| `lora_alpha` | 16 | r to 2r | Scaling factor for adapter output |
| `lora_dropout` | 0.1 | 0-0.3 | Regularization |
| `target_modules` | q,k,v,o_proj | varies | Which layers to adapt |

### Training

| Parameter | Default | Notes |
|-----------|---------|-------|
| `epochs` | 3 | More for harder tasks |
| `batch_size` | 2 | Reduce if OOM |
| `learning_rate` | 2e-4 | Standard for LoRA |
| `gradient_accumulation` | 4 | Effective batch = 8 |

---

## When to Use LoRA

### Good Use Cases

1. **Style transfer** - Change how the model writes
2. **Domain adaptation** - Medical, legal, technical language
3. **Task-specific** - Classification, extraction, formatting
4. **Multi-task** - Swap adapters for different behaviors

### Less Ideal

1. **Adding factual knowledge** - Use RAG instead
2. **General improvement** - Full fine-tuning may be needed
3. **Very different task** - May need different base model

---

## Troubleshooting

### Out of Memory

```python
# Reduce batch size
config = LoRAConfig(batch_size=1)

# Use CPU offloading
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    offload_folder="offload",
)
```

### Poor Results

1. **More data** - 50-100 examples minimum
2. **More epochs** - 10-20 for complex tasks
3. **Higher rank** - r=16 or r=32
4. **Learning rate** - Try 1e-4 to 3e-4
5. **Different base model** - Match model to task

### Training Not Converging

```python
# Add warmup
training_args = TrainingArguments(
    warmup_ratio=0.1,
    weight_decay=0.01,
)

# Check data formatting
for example in train_data[:3]:
    print(format_for_training(example))
```

---

## Production Considerations

### Adapter Management

```python
# Save adapters (small files)
model.save_pretrained("customer_service_adapter")

# Load on top of base model
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "customer_service_adapter")

# Merge into base model (for faster inference)
merged_model = model.merge_and_unload()
```

### Multiple Adapters

```python
# Train different adapters for different tasks
model.save_pretrained("formal_tone")
model.save_pretrained("casual_tone")

# Switch at runtime
model.load_adapter("formal_tone")
response = model.generate(formal_input)

model.load_adapter("casual_tone")
response = model.generate(casual_input)
```

---

## Dependencies

```bash
pip install transformers peft accelerate bitsandbytes datasets torch
```

Requirements:
- NVIDIA GPU with 4GB+ VRAM (for training)
- CUDA toolkit installed
- ~2GB disk space for model downloads

---

## Key Insights

### What We Learned

1. **LoRA enables fine-tuning on consumer hardware** - 4GB GTX 1650 can train 1.1B model
2. **Training is fast** - 3 epochs in under 4 minutes
3. **Adapter files are tiny** - ~10MB vs 4GB+ for full model
4. **Base model matters** - A well-suited base may already do well

### Trade-offs

| More Training | Less Training |
|---------------|---------------|
| Better task fit | Faster iteration |
| Risk of overfitting | May underfit |
| More compute time | Quick experiments |

---

## Next Steps

This completes the Python single-LLM modules. Next:
- **TypeScript Orchestration Patterns**
  - Sequential Chain
  - Parallel Fan-out
  - Router/Dispatcher
  - Iterative Refinement
