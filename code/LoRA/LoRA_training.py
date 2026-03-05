import os
os.environ["HF_HOME"] = "*/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "*/hf_datasets"
os.environ["TRANSFORMERS_CACHE"] = "*/transformers_cache"
os.environ["TORCH_HOME"] = "*/torch_cache"

# =====================================================
from peft import LoraConfig, get_peft_model, TaskType
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments, Trainer, TrainerCallback
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, default_data_collator
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import sys

# =====================================================
class Tee:
    def __init__(self, filepath, mode="w"):
        self.file = open(filepath, mode)
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()


# =====================================================
@dataclass
class DataCollatorForCausalLM:
    tokenizer: Any
    max_length: int = 512
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [feature["input_ids"] for feature in features]
        labels = [feature["labels"] for feature in features]

        input_ids = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )["input_ids"]

        labels = self.tokenizer.pad(
            {"input_ids": labels},
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )["input_ids"]

        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": (input_ids != self.tokenizer.pad_token_id).long(),
            "labels": labels
        }


# =====================================================
model_path = "*/llama-3.1-8b"

tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    trust_remote_code=True,
    cache_dir="/scratch/dl5683/transformers_cache"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =====================================================
dataset = load_dataset("json", data_files="train30p.jsonl", split="train")
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_data = split_dataset['train']
val_data = split_dataset['test']

def preprocess_function(examples):
    texts = []
    system_msgs   = examples["System_message"]
    lifestyles    = examples["Input1: Lifestyle"]
    environments  = examples["Input2: Environment"]
    disasters     = examples["Input3: Disaster Information"]
    outputs       = examples["Output"]
    
    for sys_msg, lf, env, dis, out in zip(system_msgs, lifestyles, environments, disasters, outputs):
        user_content = f"{lf}\n\n{env}\n\n{dis}"
        full_text = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n"
            f"{sys_msg}\n"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{user_content}\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
            f"{out}{tokenizer.eos_token}"
        )
        texts.append(full_text)
    
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=1024,
        padding=False,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("Preprocessing training data...")
train_dataset = train_data.map(
    preprocess_function,
    batched=True,
    desc="preprocess train"
)

print("Preprocessing validation data...")
val_dataset = val_data.map(
    preprocess_function,
    batched=True,
    desc="preprocess val"
)

train_dataset = train_dataset.remove_columns(train_data.column_names)
val_dataset = val_dataset.remove_columns(val_data.column_names)

print("Training dataset size:", len(train_dataset))
print("Validation dataset size:", len(val_dataset))
lengths = [len(x["input_ids"]) for x in train_dataset]
print(f"Seq len: min {min(lengths)}, max {max(lengths)}, avg {sum(lengths)/len(lengths):.1f}")

# =====================================================
config = AutoConfig.from_pretrained(
    model_path, 
    trust_remote_code=True,
    cache_dir="*/transformers_cache"
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    config=config,
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    cache_dir="*/transformers_cache"
)

# =====================================================
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
    inference_mode=False,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("LoRA model setup completed!")
print(f"Model device: {next(model.parameters()).device}")

# =====================================================
output_dir = "*/lora_models/llama3_1_LoRA"
os.makedirs(output_dir, exist_ok=True)

log_path = f"{output_dir}/training_log.txt"
sys.stdout = Tee(log_path, "w")

# =====================================================
data_collator = DataCollatorForCausalLM(
    tokenizer=tokenizer,
    max_length=512
)

# =====================================================
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    warmup_steps=10,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=300,
    save_steps=600,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to=[],
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    optim="adamw_torch",
    dataloader_drop_last=False,
)


# =====================================================
from transformers import TrainerCallback

class LossLoggingCallback(TrainerCallback):
    def __init__(self):
        self.loss_history = []
        print("\n" + "="*80)
        print(f"{'Step':<10} {'Training Loss':<20} {'Validation Loss':<20} {'Epoch':<10}")
        print("="*80)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            step = state.global_step
            epoch = state.epoch if state.epoch is not None else 0.0
            
            if 'loss' in logs:
                train_loss = logs['loss']
                self.loss_history.append({
                    'step': step,
                    'train_loss': train_loss,
                    'val_loss': None,
                    'epoch': epoch
                })
                print(f"{step:<10} {train_loss:<20.6f} {'N/A':<20} {epoch:<10.2f}")
            
            if 'eval_loss' in logs:
                val_loss = logs['eval_loss']
                if self.loss_history and self.loss_history[-1]['step'] == step:
                    self.loss_history[-1]['val_loss'] = val_loss
                    print(f"{step:<10} {'N/A':<20} {val_loss:<20.6f} {epoch:<10.2f}")
                else:
                    self.loss_history.append({
                        'step': step,
                        'train_loss': None,
                        'val_loss': val_loss,
                        'epoch': epoch
                    })
                    print(f"{step:<10} {'N/A':<20} {val_loss:<20.6f} {epoch:<10.2f}")
    
    def on_train_end(self, args, state, control, **kwargs):
        print("="*80)
        print("\nTraining Summary")
        print("="*80)
        
        csv_path = f"{args.output_dir}/loss_history.csv"
        with open(csv_path, 'w') as f:
            f.write("step,train_loss,val_loss,epoch\n")
            for entry in self.loss_history:
                train_loss = f"{entry['train_loss']:.6f}" if entry['train_loss'] is not None else ""
                val_loss = f"{entry['val_loss']:.6f}" if entry['val_loss'] is not None else ""
                f.write(f"{entry['step']},{train_loss},{val_loss},{entry['epoch']:.2f}\n")
        print(f"Loss history saved to: {csv_path}")


# =====================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    callbacks=[LossLoggingCallback()],
)

# =====================================================
print("\n=== Initial Evaluation BEFORE Training ===")
initial_eval = trainer.evaluate()
print(f"Initial eval loss: {initial_eval['eval_loss']}")
print("==========================================\n")


# =====================================================
print("Starting 30% scale LoRA training...")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"Expected steps per epoch: {len(train_dataset) // 16 + (1 if len(train_dataset) % 16 > 0 else 0)}")

if torch.cuda.is_available():
    print(f"GPU memory before training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# =====================================================
trainer.train()

# =====================================================
print("\nSaving 30% scale LoRA model...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"30% scale training completed! Model saved to {output_dir}")
print("===============================================")
print(" Training + logs saved successfully! ")
print("===============================================")
