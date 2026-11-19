#!/usr/bin/env python3
"""Llama-3.1-8B-Instruct full-sequence LoRA training for a 5-fold tract holdout.

Expected project layout (paths can be overridden through environment variables):

  data/fold{1..5}_train.csv
  data/fold{1..5}_test.csv
  data/jsonl/fold{1..5}_train.jsonl
  models/llama-3.1-8b-instruct/

The JSONL contains model-facing prompt fields and training-only synthetic
evacuation labels. The CSV and JSONL MUST have exactly the same row order.
Their contents are spot-checked on the first 32 records before geographic
fold assignment; full-record alignment is not checked by default.

Outer test-tract personas and labels are never used for training or checkpoint
selection. Each outer-training set is further split at the tract level into
LoRA fitting and internal checkpoint-validation subsets.

Full-sequence loss is computed on every real token (system + user + assistant),
with padding masked. The targets remain decision-only: explanations are NOT
included in supervision. The experiment tests whether an explanation can still
be generated at inference time using the model's existing capability.

The model tokenizer's official chat template is used.

Examples:
  python train_lora.py --fold 1 --preflight-only
  python train_lora.py --fold 1 --test-mode
  sbatch run_training.slurm
"""

# Set cache defaults before importing transformers or datasets.
# HPC users can override CACHE_ROOT, HF_HOME, HF_DATASETS_CACHE, and TORCH_HOME.
import os

_DEFAULT_CACHE_ROOT = os.environ.get(
    "CACHE_ROOT",
    os.path.join(os.path.expanduser("~"), ".cache", "llm-evacuation"),
)
os.environ.setdefault("HF_HOME", os.path.join(_DEFAULT_CACHE_ROOT, "huggingface"))
os.environ.setdefault(
    "HF_DATASETS_CACHE",
    os.path.join(_DEFAULT_CACHE_ROOT, "huggingface", "datasets"),
)
os.environ.setdefault("TORCH_HOME", os.path.join(_DEFAULT_CACHE_ROOT, "torch"))

import argparse
import gc
import inspect
import json
import math
import sys
from contextlib import redirect_stdout
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.model_selection import GroupShuffleSplit
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

import yaml
CONFIG_FILE = Path(
    os.environ.get("TRAINING_CONFIG", Path(__file__).with_name("training_config.yaml"))
).expanduser().resolve()

with CONFIG_FILE.open("r", encoding="utf-8") as stream:
    _paths = yaml.safe_load(stream)["paths"]


def resolve_config_path(config_key: str, env_key: str) -> Path:
    value = os.environ.get(env_key, _paths[config_key])
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = CONFIG_FILE.parent / path
    return path.resolve()


DATA_DIR = resolve_config_path("data_dir", "DATA_DIR")
JSONL_DIR = resolve_config_path("jsonl_dir", "JSONL_DIR")
MODEL_PATH = resolve_config_path("model_path", "MODEL_PATH")
OUTPUT_ROOT = resolve_config_path("output_root", "OUTPUT_ROOT")

BASE_SEED = 42
INNER_VAL_TRACT_FRACTION = 0.20
MAX_LENGTH = 2048

NUM_EPOCHS = 3
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LEARNING_RATE = 2e-4
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
WARMUP_STEPS = 10
LOGGING_STEPS = 10
EVAL_STEPS = 300
SAVE_STEPS = 300

REQUIRED_JSON_KEYS = {
    "System_message",
    "Input1: Lifestyle",
    "Input2: Environment",
    "Input3: Disaster Information",
    "Output",
}
VALID_RESPONSES = {
    "Final answer: Evacuate",
    "Final answer: Not evacuate",
}

class Tee:
    def __init__(self, path: Path):
        self.original = sys.stdout
        self.file = open(path, "w", encoding="utf-8", buffering=1)

    def write(self, data):
        self.original.write(data)
        self.file.write(data)

    def flush(self):
        self.original.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def norm_tract(series: pd.Series) -> pd.Series:
    result = (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(11)
    )
    if result.isna().any() or result.eq("").any():
        raise ValueError("Missing tract_id detected.")
    return result


def read_jsonl(path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if limit is not None and line_no > limit:
                break
            if not line.strip():
                raise ValueError(f"Blank JSONL line at {path}:{line_no}")
            item = json.loads(line)
            missing = REQUIRED_JSON_KEYS - set(item)
            if missing:
                raise ValueError(f"{path}:{line_no} missing keys: {missing}")
            if item["Output"] not in VALID_RESPONSES:
                raise ValueError(
                    f"{path}:{line_no}: Unexpected training output: "
                    f"{item['Output']!r}"
                )
            rows.append(item)
    if not rows:
        raise ValueError(f"No JSONL records in {path}")
    return rows


def load_and_check_outer_fold(fold: int, *, head_rows=None, validation_rows=32):
    jsonl_path = JSONL_DIR / f"fold{fold}_train.jsonl"
    train_csv_path = DATA_DIR / f"fold{fold}_train.csv"
    test_csv_path = DATA_DIR / f"fold{fold}_test.csv"
    for path in (jsonl_path, train_csv_path, test_csv_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input file: {path}")

    records = read_jsonl(jsonl_path, limit=head_rows)
    train_csv = pd.read_csv(train_csv_path, dtype=str, keep_default_na=False, nrows=head_rows)
    test_csv = pd.read_csv(test_csv_path, dtype=str, keep_default_na=False,
                           nrows=head_rows if head_rows is not None else None)
    if "tract_id" not in train_csv or "tract_id" not in test_csv:
        raise ValueError("Both fold CSVs must contain tract_id.")
    if len(records) != len(train_csv):
        raise ValueError(
            f"Fold {fold}: JSONL has {len(records)} records but train CSV "
            f"has {len(train_csv)} rows. Re-export aligned CSV/JSONL."
        )
    train_csv["tract_id"] = norm_tract(train_csv["tract_id"])
    test_csv["tract_id"] = norm_tract(test_csv["tract_id"])
    outer_train_tracts = set(train_csv["tract_id"])
    outer_test_tracts = set(test_csv["tract_id"])
    intersection = outer_train_tracts & outer_test_tracts
    if intersection:
        raise ValueError(f"Fold {fold}: TRAIN/TEST tract overlap: {intersection}")

    required_alignment_columns = [
        "persona_output",
        "evacuation_label",
        "distance_to_fire_km",
        "direction_from_fire",
    ]
    missing_alignment_columns = [
        col for col in required_alignment_columns if col not in train_csv.columns
    ]
    if missing_alignment_columns:
        raise ValueError(
            f"Fold {fold}: CSV is missing essential alignment columns: "
            f"{missing_alignment_columns}"
        )

    expected_lifestyles = train_csv["persona_output"].astype(str).str.strip()

    print(f"Checking the first {min(validation_rows, len(records))} aligned records only")
    for i, rec in enumerate(records[:validation_rows]):
        expected_lifestyle = expected_lifestyles.iloc[i]
        input_lifestyle = rec["Input1: Lifestyle"].strip()

        lifestyle_prefix = "Below is the individual's lifestyle and mobility pattern:"
        if input_lifestyle.startswith(lifestyle_prefix):
            input_lifestyle = input_lifestyle[len(lifestyle_prefix):].strip()
        if " ".join(input_lifestyle.split()) != " ".join(expected_lifestyle.split()):
            raise ValueError(
                f"Fold {fold}, row {i}: Cleaned JSONL and CSV lifestyle differ. "
                "Check the income-removal audit and row alignment."
            )

        expected_label = train_csv["evacuation_label"].iloc[i].strip()
        if rec["Output"].strip() != expected_label:
            raise ValueError(f"Fold {fold}, row {i}: JSONL/CSV labels differ.")

        environment = rec["Input2: Environment"]

        d_raw = train_csv["distance_to_fire_km"].iloc[i]
        distance_marker = f"{float(d_raw):.2f} km"
        if distance_marker not in environment:
            raise ValueError(
                f"Fold {fold}, row {i}: fire distance mismatches CSV."
            )

        direction = str(train_csv["direction_from_fire"].iloc[i]).strip()
        if f"to the {direction} " not in environment:
            raise ValueError(
                f"Fold {fold}, row {i}: fire direction mismatches CSV."
            )

    print(
        f"Fold {fold} {'HEAD' if head_rows is not None else 'FULL'} outer train: {len(train_csv):,} personas / "
        f"{len(outer_train_tracts)} tracts"
    )
    print(
        f"Fold {fold} held-out test: {len(test_csv):,} personas / "
        f"{len(outer_test_tracts)} tracts (NOT loaded into model)"
    )
    return records, train_csv, outer_train_tracts, outer_test_tracts


def split_inner_train_validation(records, train_csv, fold: int):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=INNER_VAL_TRACT_FRACTION,
        random_state=BASE_SEED + fold,
    )
    train_idx, val_idx = next(
        splitter.split(train_csv, groups=train_csv["tract_id"])
    )
    fit_tracts = set(train_csv.iloc[train_idx]["tract_id"])
    val_tracts = set(train_csv.iloc[val_idx]["tract_id"])
    if not fit_tracts or not val_tracts or fit_tracts & val_tracts:
        raise ValueError(f"Fold {fold}: invalid internal geographic split")

    fit_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    print(
        f"Inner LoRA train: {len(fit_records):,} personas / "
        f"{len(fit_tracts)} tracts"
    )
    print(
        f"Inner checkpoint validation: {len(val_records):,} personas / "
        f"{len(val_tracts)} tracts"
    )
    return fit_records, val_records, fit_tracts, val_tracts


def tokenize_training_record(item: dict, tokenizer, max_length: int) -> dict:
    user_content = (
        f"{item['Input1: Lifestyle']}\n\n"
        f"{item['Input2: Environment']}\n\n"
        f"{item['Input3: Disaster Information']}"
    )
    history = [
        {"role": "system", "content": item["System_message"]},
        {"role": "user", "content": user_content},
    ]
    if not str(item["Output"]).strip():
        raise ValueError("Assistant output is empty.")
    encoded = tokenizer.apply_chat_template(
        history + [{"role": "assistant", "content": item["Output"]}],
        tokenize=True,
        add_generation_prompt=False,
    )
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise ValueError(
                f"Chat template returned keys {list(encoded.keys())}, without input_ids"
            )
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, tuple):
        encoded = list(encoded)
    if isinstance(encoded, list) and encoded and isinstance(encoded[0], (list, tuple)):
        if len(encoded) != 1:
            raise ValueError("Expected one tokenized conversation, not a batch.")
        encoded = list(encoded[0])
    if not isinstance(encoded, list) or not encoded or not all(
        isinstance(token_id, (int, np.integer)) for token_id in encoded
    ):
        raise TypeError(
            "Chat template must yield a flat list of token IDs; "
            f"got {type(encoded).__name__}"
        )
    full_ids = [int(token_id) for token_id in encoded]
    if len(full_ids) > max_length:
        raise ValueError(
            f"One prompt is {len(full_ids)} tokens (> max_length={max_length}). "
            "Increase --max-length; do not truncate away training labels."
        )
    if not full_ids:
        raise ValueError("Chat template produced no tokens.")
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": full_ids.copy(),
    }


@dataclass
class FullSequenceCollator:
    pad_token_id: int
    pad_to_multiple_of: int = 8

    def __call__(self, examples: List[Dict[str, Any]]):
        max_len = max(len(x["input_ids"]) for x in examples)
        k = self.pad_to_multiple_of
        if k:
            max_len = ((max_len + k - 1) // k) * k
        ids, masks, labels = [], [], []
        for example in examples:
            n = len(example["input_ids"])
            gap = max_len - n
            ids.append(example["input_ids"] + [self.pad_token_id] * gap)
            masks.append([1] * n + [0] * gap)
            labels.append(example["labels"] + [-100] * gap)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def to_tokenized_dataset(records, tokenizer, max_length, name):
    dataset = Dataset.from_list(records)
    original_columns = dataset.column_names
    dataset = dataset.map(
        lambda item: tokenize_training_record(item, tokenizer, max_length),
        batched=False,
        remove_columns=original_columns,
        load_from_cache_file=False,
        desc=f"Tokenize {name}",
    )
    lengths = np.fromiter(
        (len(ids) for ids in dataset["input_ids"]),
        dtype=np.int32,
    )
    print(
        f"{name}: n={len(dataset):,}, token lengths "
        f"min={lengths.min()}, mean={lengths.mean():.1f}, "
        f"max={lengths.max()}"
    )
    return dataset


class LossLoggingCallback(TrainerCallback):
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "loss" not in logs and "eval_loss" not in logs:
            return
        line = {
            "step": state.global_step,
            "epoch": state.epoch,
            "train_loss": logs.get("loss"),
            "val_loss": logs.get("eval_loss"),
        }
        self.history.append(line)
        print(
            f"Step {state.global_step} | epoch={state.epoch} | "
            f"training_loss={line['train_loss']} | validation_loss={line['val_loss']}"
        )

    def on_train_end(self, args, state, control, **kwargs):
        pd.DataFrame(self.history).to_csv(
            self.output_dir / "loss_history.csv", index=False
        )


def build_training_args(fold_output_dir: Path, seed: int, test_mode: bool = False, dry_run: bool = False):
    settings = dict(
        output_dir=str(fold_output_dir / "checkpoints"),
        num_train_epochs=1 if test_mode else NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=0 if test_mode else WARMUP_STEPS,
        bf16=True,
        logging_strategy="steps",
        logging_steps=1 if test_mode else LOGGING_STEPS,
        eval_steps=1 if test_mode else EVAL_STEPS,
        save_strategy="steps",
        save_steps=1 if test_mode else SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_torch",
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        report_to="none",
        seed=seed,
        data_seed=seed,
        prediction_loss_only=True,
    )
    parameter_names = inspect.signature(TrainingArguments).parameters
    if "eval_strategy" in parameter_names:
        settings["eval_strategy"] = "steps"
    else:
        settings["evaluation_strategy"] = "steps"
    if "gradient_checkpointing_kwargs" in parameter_names:
        settings["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    unsupported = sorted(set(settings) - set(parameter_names))
    if unsupported:
        raise TypeError(f"Unsupported TrainingArguments parameters: {unsupported}")
    if dry_run:
        return settings
    return TrainingArguments(**settings)


def check_tokenizer_collator(records, tokenizer, max_length):
    samples = [
        tokenize_training_record(item, tokenizer, max_length)
        for item in records[:min(3, len(records))]
    ]
    if not samples:
        raise ValueError("No samples available for tokenizer preflight")
    batch = FullSequenceCollator(tokenizer.pad_token_id)(samples)
    real = batch["attention_mask"].bool()
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["input_ids"].ndim == 2
    assert torch.equal(batch["input_ids"][real], batch["labels"][real])
    assert torch.all(batch["labels"][~real] == -100)
    assert sum(item["Output"].startswith("Final answer:") for item in records[:len(samples)]) == len(samples)
    print(f"PREFLIGHT PASSED: {len(samples)} conversations; "
          f"padded shape={tuple(batch['input_ids'].shape)}; "
          "all prompt/answer tokens supervised; only padding masked")
    return batch


def train_fold(fold: int, args):
    fold_output_dir = args.output_root / f"fold{fold}"
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    tee = Tee(fold_output_dir / "training_log.txt")
    with redirect_stdout(tee):
        try:
            print("=" * 72)
            print(f"Llama-3.1-8B-Instruct FULL-SEQUENCE LoRA / outer Fold {fold}")
            print("=" * 72)
            print(f"Base model: {args.model_path}")
            print(f"CSV directory: {DATA_DIR}")
            print(f"JSONL directory: {JSONL_DIR}")
            print(f"Output: {fold_output_dir}")
            print("Data policy: outer-test outcomes NEVER used for training")

            set_seed(BASE_SEED + fold)
            if args.check_data_only:
                load_and_check_outer_fold(fold, head_rows=args.head_rows,
                                          validation_rows=args.head_rows)
                print(f"HEAD-ONLY data check passed: fold {fold}, first {args.head_rows} rows")
                return
            if args.preflight_only:
                head_records, _, _, _ = load_and_check_outer_fold(
                    fold, head_rows=args.head_rows, validation_rows=args.head_rows
                )
                tok = AutoTokenizer.from_pretrained(str(args.model_path))
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                tok.padding_side = "right"
                if not tok.chat_template:
                    raise ValueError("Tokenizer has no chat template")
                check_tokenizer_collator(head_records, tok, args.max_length)
                build_training_args(fold_output_dir, BASE_SEED + fold,
                                    test_mode=True, dry_run=True)
                print("PREFLIGHT PASSED: TrainingArguments signature supported; no GPU/model used")
                del tok
                return
            limit = args.smoke_head_rows if args.test_mode else None
            records, train_csv, outer_train, outer_test = (
                load_and_check_outer_fold(fold, head_rows=limit,
                                          validation_rows=args.head_rows)
            )
            fit, val, fit_tracts, val_tracts = (
                split_inner_train_validation(records, train_csv, fold)
            )
            assert (fit_tracts | val_tracts) == outer_train

            if args.test_mode:
                fit = fit[:args.test_train_samples]
                val = val[:args.test_val_samples]
                if not fit or not val:
                    raise ValueError("Test subset is empty; increase sample counts.")
                print(f"SMOKE TEST: train={len(fit)}, validation={len(val)}")
            assert (fit_tracts | val_tracts).isdisjoint(outer_test)

            pd.DataFrame(
                [{"tract_id": tract, "role": "lora_train"}
                 for tract in sorted(fit_tracts)]
                + [{"tract_id": tract, "role": "internal_validation"}
                   for tract in sorted(val_tracts)]
                + [{"tract_id": tract, "role": "outer_test"}
                   for tract in sorted(outer_test)]
            ).to_csv(fold_output_dir / "tract_split_manifest.csv", index=False)

            config = {
                "fold": fold,
                "test_mode": args.test_mode,
                "base_model_path": str(args.model_path),
                "jsonl_path": str(JSONL_DIR / f"fold{fold}_train.jsonl"),
                "csv_path": str(DATA_DIR / f"fold{fold}_train.csv"),
                "max_length": args.max_length,
                "seed": BASE_SEED + fold,
                "inner_val_tract_fraction": INNER_VAL_TRACT_FRACTION,
                "outer_train_tracts": len(outer_train),
                "outer_test_tracts": len(outer_test),
                "internal_train_tracts": len(fit_tracts),
                "internal_validation_tracts": len(val_tracts),
                "num_internal_train_examples": len(fit),
                "num_internal_val_examples": len(val),
                "num_epochs": 1 if args.test_mode else NUM_EPOCHS,
                "batch_size": TRAIN_BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "learning_rate": LEARNING_RATE,
                "warmup_steps": 0 if args.test_mode else WARMUP_STEPS,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "lora_dropout": LORA_DROPOUT,
                "lora_target_modules": [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                "eval_steps": 1 if args.test_mode else EVAL_STEPS,
                "save_steps": 1 if args.test_mode else SAVE_STEPS,
                "loss": "full_sequence_cross_entropy",
                "training_target": "decision_only_no_explanation",
                "explanation_inference_probe": "separate_script",
                "prompt_format": "tokenizer.apply_chat_template",
                "persona_source": "uploaded_generator_jsonl_persona_output_unchanged",
                "alert_context": "as_present_in_generated_jsonl",
                "alert_categories": ["none", "warning", "order", "normal"],
            }
            with (fold_output_dir / "run_config.json").open("w") as out:
                json.dump(config, out, indent=2)

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU is required for Llama-3.1-8B training.")
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("GPU does not support BF16: adjust precision before running.")
            gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU: {gpu_name}")
            print(f"Visible CUDA devices: {torch.cuda.device_count()}")
            if "H200" not in gpu_name.upper():
                raise RuntimeError(
                    f"Expected H200, got {gpu_name}. Check Slurm --constraint=h200."
                )

            tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            if not tokenizer.chat_template:
                raise ValueError("Tokenizer has no chat template")
            print("Using tokenizer native chat template.")

            training_args = build_training_args(
                fold_output_dir, BASE_SEED + fold, test_mode=args.test_mode
            )
            check_tokenizer_collator(fit, tokenizer, args.max_length)

            train_data = to_tokenized_dataset(
                fit, tokenizer, args.max_length, "LoRA training"
            )
            val_data = to_tokenized_dataset(
                val, tokenizer, args.max_length, "Checkpoint validation"
            )
            sample0 = train_data[0]
            assert any(y != -100 for y in sample0["labels"])
            assert sample0["labels"] == sample0["input_ids"]
            print("Sample full-sequence labels verified (prompt + answer, padding excluded).")

            print("Loading a NEW base model (never reusing another fold's LoRA).")
            model = AutoModelForCausalLM.from_pretrained(
                str(args.model_path),
                torch_dtype=torch.bfloat16,
                device_map={"": torch.cuda.current_device()},
                low_cpu_mem_usage=True,
            )
            model.config.use_cache = False
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                bias="none",
                inference_mode=False,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_data,
                eval_dataset=val_data,
                data_collator=FullSequenceCollator(tokenizer.pad_token_id),
                callbacks=[LossLoggingCallback(fold_output_dir)],
            )
            print("Starting LoRA training...")
            train_result = trainer.train()
            print("Final internal-validation evaluation...")
            validation_metrics = trainer.evaluate()

            final_adapter_dir = fold_output_dir / "final_adapter"
            trainer.save_model(str(final_adapter_dir))
            tokenizer.save_pretrained(str(final_adapter_dir))

            metrics = {
                "fold": fold,
                "gpu_name": gpu_name,
                "pytorch_version": torch.__version__,
                "cuda_runtime_version": torch.version.cuda,
                "train_metrics": train_result.metrics,
                "internal_val_metrics": validation_metrics,
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "best_eval_loss": trainer.state.best_metric,
                "NOTE": "Validation is TRAIN-INTERNAL. Held-out test GT unused.",
            }
            with (fold_output_dir / "metrics.json").open("w") as out:
                json.dump(metrics, out, indent=2, default=float)
            trainer.save_state()
            print(f"Finished fold {fold}. Adapter saved: {final_adapter_dir}")

            del trainer, model, tokenizer, train_data, val_data
            gc.collect()
            torch.cuda.empty_cache()
        finally:
            tee.flush()
    tee.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold", choices=["1", "2", "3", "4", "5"], default="1",
        help="Exactly one fold per GPU/job (use Slurm array 1-5 for parallelism).",
    )
    parser.add_argument(
        "--max-length", type=int, default=MAX_LENGTH,
        help="Maximum combined prompt+answer token count; errors if exceeded.",
    )
    parser.add_argument(
        "--check-data-only", action="store_true",
        help="Verify CSV/JSONL and geographic splits without loading model/GPU.",
    )
    parser.add_argument("--test-mode", action="store_true",
                        help="Run a short end-to-end smoke test with small train/val subsets.")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Check first --head-rows examples with real tokenizer, collator and TrainingArguments; no GPU or full-data read.")
    parser.add_argument("--head-rows", type=int, default=32,
                        help="Number of rows to verify; also limits --check-data-only file reads.")
    parser.add_argument("--smoke-head-rows", type=int, default=2000,
                        help="Read only this many rows for --test-mode before tract splitting.")
    parser.add_argument("--test-train-samples", type=int, default=32)
    parser.add_argument("--test-val-samples", type=int, default=16)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.head_rows < 1 or args.smoke_head_rows < 2:
        raise ValueError("Head sample counts must be positive (smoke >= 2).")
    if args.test_train_samples < 1 or args.test_val_samples < 1:
        raise ValueError("Test sample counts must be positive.")
    if args.test_mode and args.output_root == OUTPUT_ROOT:
        args.output_root = OUTPUT_ROOT.parent / (OUTPUT_ROOT.name + "_TEST")
    if args.max_length < 64:
        raise ValueError("--max-length is too small.")
    if not args.check_data_only and not args.model_path.is_dir():
        raise FileNotFoundError(f"Base model not found: {args.model_path}")
    folds = [int(args.fold)]
    print(f"Python={sys.version.split()[0]}, NumPy={np.__version__}")
    print(f"PyTorch={torch.__version__}, CUDA={torch.cuda.is_available()}")
    for fold in folds:
        train_fold(fold, args)
    print("Requested fold(s) completed.")


if __name__ == "__main__":
    main()