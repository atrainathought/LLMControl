"""
LoRA Fine-tuning Info Page (Read-only)
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from styles import apply_styles, page_header, section_divider, setup_sidebar

st.set_page_config(page_title="LoRA Fine-tuning | LLMControl", page_icon="🎛️", layout="wide")

apply_styles()
page_header("LoRA Fine-tuning", "Parameter-efficient fine-tuning for domain adaptation", "🎛️")

section_divider()

st.info("This is an informational page. LoRA training requires GPU resources and is not interactive.")

# What is LoRA
st.markdown("### What is LoRA?")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    **LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning technique that:

    - Freezes the original model weights
    - Adds small trainable matrices (adapters) to each layer
    - Reduces trainable parameters by **99%+**
    - Enables fine-tuning on consumer GPUs

    Instead of updating all model parameters, LoRA decomposes weight updates into low-rank matrices:

    ```
    W' = W + BA
    ```

    Where:
    - `W` = Original frozen weights
    - `B, A` = Low-rank trainable matrices (rank 4-64)
    - `W'` = Effective adapted weights
    """)

with col2:
    st.markdown("**Key Benefits**")
    st.success("99% less parameters")
    st.success("10x less memory")
    st.success("Runs on consumer GPUs")
    st.success("Easy to swap adapters")

section_divider()

# Comparison
st.markdown("### Full Fine-tuning vs LoRA")

import pandas as pd

comparison = pd.DataFrame({
    "Aspect": ["Trainable Params", "GPU Memory", "Training Time", "Storage", "Switching Tasks"],
    "Full Fine-tuning": ["100%", "80+ GB", "Hours-Days", "Full model copy", "Load new model"],
    "LoRA": ["0.1-1%", "8-24 GB", "Minutes-Hours", "Small adapter", "Swap adapter file"]
})

st.dataframe(comparison, use_container_width=True, hide_index=True)

section_divider()

# When to use
st.markdown("### When to Use LoRA")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Good Use Cases**")
    st.markdown("""
    - Domain-specific terminology
    - Consistent output style/format
    - Specialized knowledge areas
    - Multi-task with adapter switching
    - Resource-constrained environments
    """)

with col2:
    st.markdown("**Consider Alternatives When**")
    st.markdown("""
    - Few-shot prompting works well
    - RAG can provide needed knowledge
    - Task is too general
    - No training data available
    - Real-time adaptation needed
    """)

section_divider()

# Training Process
st.markdown("### Training Process")

st.markdown("""
```
1. Prepare Dataset
   └── Format: {"instruction": "...", "input": "...", "output": "..."}

2. Configure LoRA
   └── rank: 8-64 (higher = more capacity)
   └── alpha: 16-128 (scaling factor)
   └── target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

3. Train
   └── Epochs: 1-5
   └── Learning rate: 1e-4 to 3e-4
   └── Batch size: 4-16

4. Evaluate & Merge
   └── Test on held-out data
   └── Optionally merge adapter into base model
```
""")

section_divider()

# Configuration Example
st.markdown("### Configuration Example")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**LoRA Config**")
    st.code("""
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,                    # Rank
    lora_alpha=32,           # Scaling
    target_modules=[
        "q_proj", "k_proj",
        "v_proj", "o_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
    """, language="python")

with col2:
    st.markdown("**Training Config**")
    st.code("""
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./lora_adapter",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,  # Mixed precision
)
    """, language="python")

section_divider()

# Dataset Format
st.markdown("### Dataset Format")

st.markdown("Training data should be in instruction format:")

st.code("""
[
  {
    "instruction": "Classify the sentiment of this review",
    "input": "This product exceeded my expectations!",
    "output": "positive"
  },
  {
    "instruction": "Summarize this technical document",
    "input": "The API endpoint accepts POST requests...",
    "output": "The API uses POST for data submission with JSON payloads."
  }
]
""", language="json")

section_divider()

# Results Example
st.markdown("### Expected Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Training Time", "~30 min", help="On RTX 3090, 1K examples")
with col2:
    st.metric("Adapter Size", "~50 MB", help="vs 14GB for full model")
with col3:
    st.metric("Task Accuracy", "+15-25%", help="Improvement over base model")

st.markdown("**Sample Training Curve**")

# Simulated training curve
import pandas as pd

epochs = list(range(1, 11))
loss = [2.5, 1.8, 1.4, 1.1, 0.9, 0.75, 0.65, 0.58, 0.52, 0.48]
accuracy = [0.65, 0.72, 0.78, 0.82, 0.85, 0.87, 0.89, 0.90, 0.91, 0.92]

chart_data = pd.DataFrame({
    "Epoch": epochs,
    "Loss": loss,
    "Accuracy": [a * 100 for a in accuracy]
})

col1, col2 = st.columns(2)
with col1:
    st.line_chart(chart_data.set_index("Epoch")["Loss"])
    st.caption("Training Loss")
with col2:
    st.line_chart(chart_data.set_index("Epoch")["Accuracy"])
    st.caption("Validation Accuracy (%)")

section_divider()

# Full Example
with st.expander("Full Training Script"):
    st.code("""
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer
)
from peft import get_peft_model, LoraConfig

# 1. Load base model
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 3. Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.062%

# 4. Load dataset
dataset = load_dataset("json", data_files="training_data.json")

# 5. Train
trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],
    args=TrainingArguments(
        output_dir="./lora_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-4,
        fp16=True,
    )
)
trainer.train()

# 6. Save adapter
model.save_pretrained("./my_lora_adapter")

# 7. Load and use
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained(model_name)
model = PeftModel.from_pretrained(base_model, "./my_lora_adapter")
    """, language="python")

# Resources
st.markdown("### Resources")

st.markdown("""
- [PEFT Library](https://github.com/huggingface/peft) - Hugging Face's parameter-efficient fine-tuning
- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Original research paper
- [QLoRA](https://arxiv.org/abs/2305.14314) - 4-bit quantized LoRA for even smaller memory
""")

setup_sidebar()
