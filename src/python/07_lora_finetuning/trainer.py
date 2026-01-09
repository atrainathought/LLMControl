"""
LoRA Fine-tuning Trainer

This module implements LoRA (Low-Rank Adaptation) fine-tuning using:
- PEFT (Parameter-Efficient Fine-Tuning) library
- 4-bit quantization for memory efficiency
- Small models suitable for 4GB VRAM

LoRA works by freezing the pretrained model and injecting trainable
low-rank matrices into transformer layers, dramatically reducing
the number of trainable parameters.
"""

import os
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

from dataset import prepare_hf_dataset, format_for_inference


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""
    # Model settings
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small model for 4GB VRAM

    # LoRA hyperparameters
    lora_r: int = 8           # Rank of the low-rank matrices
    lora_alpha: int = 16      # Scaling factor
    lora_dropout: float = 0.1  # Dropout probability
    target_modules: list = None  # Which modules to apply LoRA to

    # Training settings
    epochs: int = 3
    batch_size: int = 2       # Small batch for limited VRAM
    learning_rate: float = 2e-4
    max_length: int = 256

    # Quantization
    use_4bit: bool = True     # Use 4-bit quantization
    bnb_4bit_compute_dtype: str = "float16"

    # Output
    output_dir: str = "./lora_output"

    def __post_init__(self):
        if self.target_modules is None:
            # Default target modules for LLaMA-style models
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


class LoRATrainer:
    """
    LoRA Fine-tuning Trainer

    Uses Parameter-Efficient Fine-Tuning (PEFT) to train only a small
    number of adapter weights while keeping the base model frozen.
    """

    def __init__(self, config: LoRAConfig = None):
        self.config = config or LoRAConfig()
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup_quantization(self):
        """Configure 4-bit quantization for memory efficiency."""
        if self.config.use_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=True,
            )
        return None

    def load_model(self):
        """Load the base model with quantization."""
        print(f"Loading model: {self.config.model_name}")

        quantization_config = self.setup_quantization()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Prepare model for k-bit training
        if self.config.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        print(f"Model loaded. Parameters: {self.model.num_parameters():,}")
        return self.model, self.tokenizer

    def apply_lora(self):
        """Apply LoRA adapters to the model."""
        print("Applying LoRA adapters...")

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.model, lora_config)

        trainable, total = self.model.get_nb_trainable_parameters()
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

        return self.model

    def train(self, train_dataset, val_dataset):
        """Run the training loop."""
        print("Starting training...")

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=4,  # Effective batch size = 8
            learning_rate=self.config.learning_rate,
            weight_decay=0.01,
            logging_steps=5,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            fp16=True,  # Mixed precision for speed
            report_to="none",  # Disable wandb etc.
            warmup_ratio=0.1,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        # Train
        train_result = self.trainer.train()

        # Save the LoRA adapters
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)

        print(f"Training complete. Model saved to {self.config.output_dir}")
        return train_result

    def generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        """Generate a response using the model."""
        formatted = format_for_inference(prompt)

        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the response part
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()

        return response


def run_training(output_dir: str = None) -> LoRATrainer:
    """
    Run the complete LoRA fine-tuning pipeline.

    Returns the trained LoRATrainer instance.
    """
    # Configuration
    config = LoRAConfig(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        lora_r=8,
        lora_alpha=16,
        epochs=3,
        batch_size=2,
        learning_rate=2e-4,
        output_dir=output_dir or "./lora_output",
    )

    # Initialize trainer
    trainer = LoRATrainer(config)

    # Load and prepare model
    trainer.load_model()
    trainer.apply_lora()

    # Prepare datasets
    print("Preparing datasets...")
    train_dataset, val_dataset = prepare_hf_dataset(
        trainer.tokenizer,
        max_length=config.max_length
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Train
    trainer.train(train_dataset, val_dataset)

    return trainer


def load_trained_model(model_path: str) -> LoRATrainer:
    """Load a previously trained LoRA model."""
    from peft import PeftModel

    config = LoRAConfig(output_dir=model_path)
    trainer = LoRATrainer(config)

    # Load base model
    trainer.load_model()

    # Load LoRA adapters
    trainer.model = PeftModel.from_pretrained(
        trainer.model,
        model_path,
        device_map="auto",
    )

    print(f"Loaded LoRA model from {model_path}")
    return trainer


if __name__ == "__main__":
    # Quick test
    print("LoRA Trainer module loaded successfully")
    print(f"Default config: {LoRAConfig()}")
