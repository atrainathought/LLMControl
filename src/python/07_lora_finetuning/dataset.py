"""
Training Dataset for LoRA Fine-tuning

This module creates a custom dataset for fine-tuning a model on a specific task:
Converting informal customer messages into professional support responses.

Task: Customer Service Tone Transformation
- Input: Casual/informal customer message
- Output: Professional, empathetic support response
"""

from typing import List, Dict
import json
import random

# Training examples for tone transformation
# Format: informal input -> professional output
TRAINING_DATA = [
    {
        "input": "yo my order never showed up wtf",
        "output": "I sincerely apologize for the inconvenience with your order delivery. I understand how frustrating this must be. Let me look into this immediately and find out where your package is. Could you please provide your order number so I can track it for you?"
    },
    {
        "input": "this thing is broken already smh",
        "output": "I'm very sorry to hear that your product isn't working as expected. That's certainly not the experience we want you to have. I'd be happy to help you troubleshoot the issue or arrange a replacement. Could you tell me more about what's happening with the product?"
    },
    {
        "input": "been waiting forever for someone to help me",
        "output": "I apologize for the extended wait time you've experienced. I understand your time is valuable, and I appreciate your patience. I'm here now and ready to give your issue my full attention. How may I assist you today?"
    },
    {
        "input": "your website is so confusing i cant find anything",
        "output": "I apologize for the difficulty you've had navigating our website. Thank you for bringing this to our attention. I'd be happy to help you find what you're looking for. What specific product or information are you trying to locate?"
    },
    {
        "input": "charged me twice fix it now",
        "output": "I sincerely apologize for the duplicate charge on your account. I understand how concerning this must be. Let me investigate this right away and ensure we process a refund for the extra charge. May I have your order number or the email associated with your account?"
    },
    {
        "input": "this is the worst service ever",
        "output": "I'm truly sorry that your experience hasn't met your expectations. Your feedback is important to us, and I want to make this right. Could you share more details about what went wrong so I can address it properly and improve your experience?"
    },
    {
        "input": "nobody knows what theyre doing there",
        "output": "I apologize if you've received inconsistent information from our team. I understand how frustrating that can be. Let me take ownership of your issue and ensure we get it resolved correctly this time. What can I help you with today?"
    },
    {
        "input": "ugh another broken product from you guys",
        "output": "I'm very sorry to hear you've received another defective item. That's absolutely not acceptable, and I understand your frustration. Let me arrange an immediate replacement and look into why this has happened more than once. We value your business and want to make this right."
    },
    {
        "input": "where is my refund its been weeks",
        "output": "I apologize for the delay in processing your refund. I completely understand your concern about the timeline. Let me check on the status right now and provide you with a clear update. Could you please share your order number or transaction ID?"
    },
    {
        "input": "i want to cancel everything this is ridiculous",
        "output": "I'm sorry to hear you're considering cancellation. I understand your frustration, and I'd like the opportunity to address your concerns before you make that decision. Would you be willing to share what's prompted this so I can try to resolve it for you?"
    },
    {
        "input": "hello i need help with my account",
        "output": "Hello! I'd be happy to help you with your account. What specific aspect of your account can I assist you with today?"
    },
    {
        "input": "can u send me a new one or what",
        "output": "Absolutely, I'd be glad to arrange a replacement for you. To process this quickly, could you please provide your order number and confirm your shipping address?"
    },
    {
        "input": "literally the worst experience of my life",
        "output": "I'm deeply sorry that your experience has been so disappointing. That's certainly not what we strive for. I want to understand what happened and see how we can make it right. Would you mind sharing the details of your experience?"
    },
    {
        "input": "do you even care about customers",
        "output": "I absolutely understand why you might feel that way given your experience, and I apologize. Your satisfaction truly matters to us. I'm here to listen and help resolve whatever issue you're facing. Please tell me what's happened."
    },
    {
        "input": "just give me my money back already",
        "output": "I completely understand your desire for a prompt refund. I apologize for any delays you've experienced. Let me process this for you right away. Could you provide your order number so I can initiate the refund immediately?"
    },
    {
        "input": "how do i return this garbage",
        "output": "I'm sorry the product didn't meet your expectations. I'd be happy to help you with the return process. I can email you a prepaid shipping label, and once we receive the item, we'll process your refund within 3-5 business days. Would you like me to send that label now?"
    },
    {
        "input": "your app keeps crashing",
        "output": "I apologize for the technical difficulties you're experiencing with our app. That must be frustrating. Let me help you troubleshoot this. Could you tell me what device you're using and which version of the app you have installed?"
    },
    {
        "input": "been on hold for an hour this is insane",
        "output": "I sincerely apologize for the extensive hold time you experienced. That's unacceptable, and I appreciate your patience in waiting. I'm fully available now to help resolve your issue. What can I assist you with?"
    },
    {
        "input": "wrong size again how hard is it",
        "output": "I apologize for sending the incorrect size again. I understand how frustrating it is to receive the wrong item, especially more than once. Let me arrange an immediate exchange with expedited shipping at no extra cost. What size do you need?"
    },
    {
        "input": "this is a scam company",
        "output": "I'm sorry you feel that way about our company. I assure you we're a legitimate business, and I want to address whatever issue has led to this perception. Could you share what's happened so I can investigate and make things right?"
    },
]

# Test cases to evaluate the fine-tuned model
TEST_DATA = [
    {
        "input": "my package is lost and nobody cares",
        "expected_tone": ["apologize", "understand", "help", "track"],
        "should_not_contain": ["wtf", "smh", "lol", "idk"]
    },
    {
        "input": "this is such a ripoff",
        "expected_tone": ["sorry", "understand", "value", "help"],
        "should_not_contain": ["ripoff", "scam", "worst"]
    },
    {
        "input": "ur customer service sucks",
        "expected_tone": ["apologize", "feedback", "improve", "assist"],
        "should_not_contain": ["sucks", "ur", "u"]
    },
    {
        "input": "need a refund asap",
        "expected_tone": ["happy", "help", "refund", "process"],
        "should_not_contain": ["asap", "now", "immediately"]
    },
    {
        "input": "been waiting 2 weeks for shipping",
        "expected_tone": ["apologize", "delay", "understand", "check"],
        "should_not_contain": ["wtf", "ridiculous", "insane"]
    },
]


def format_for_training(example: Dict, tokenizer=None) -> str:
    """Format a single example for causal LM training."""
    # Format as instruction-response pair
    prompt = f"""### Instruction:
Transform this informal customer message into a professional, empathetic customer service response.

### Input:
{example['input']}

### Response:
{example['output']}"""
    return prompt


def format_for_inference(user_input: str) -> str:
    """Format input for inference (without the response)."""
    prompt = f"""### Instruction:
Transform this informal customer message into a professional, empathetic customer service response.

### Input:
{user_input}

### Response:"""
    return prompt


def create_dataset_splits(train_ratio: float = 0.8):
    """Split data into train and validation sets."""
    data = TRAINING_DATA.copy()
    random.seed(42)  # For reproducibility
    random.shuffle(data)

    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    return train_data, val_data


def prepare_hf_dataset(tokenizer, max_length: int = 256):
    """Prepare dataset in HuggingFace format for training."""
    from datasets import Dataset

    train_data, val_data = create_dataset_splits()

    def tokenize_function(examples):
        # Format each example
        texts = [format_for_training({"input": inp, "output": out})
                 for inp, out in zip(examples["input"], examples["output"])]

        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

        # For causal LM, labels = input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    # Create HuggingFace datasets
    train_dict = {"input": [d["input"] for d in train_data],
                  "output": [d["output"] for d in train_data]}
    val_dict = {"input": [d["input"] for d in val_data],
                "output": [d["output"] for d in val_data]}

    train_dataset = Dataset.from_dict(train_dict).map(
        tokenize_function,
        batched=True,
        remove_columns=["input", "output"]
    )

    val_dataset = Dataset.from_dict(val_dict).map(
        tokenize_function,
        batched=True,
        remove_columns=["input", "output"]
    )

    return train_dataset, val_dataset


def get_dataset_stats():
    """Get statistics about the training data."""
    train_data, val_data = create_dataset_splits()

    avg_input_len = sum(len(d["input"]) for d in TRAINING_DATA) / len(TRAINING_DATA)
    avg_output_len = sum(len(d["output"]) for d in TRAINING_DATA) / len(TRAINING_DATA)

    return {
        "total_examples": len(TRAINING_DATA),
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "test_examples": len(TEST_DATA),
        "avg_input_chars": avg_input_len,
        "avg_output_chars": avg_output_len,
    }
