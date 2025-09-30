#!/usr/bin/env python3

import os
import json
import random
import math
import ast
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
import torch
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


DATA_DIR = "/Users/siddu/Developer/Business-Insights-Recommender"
ENGINEERED_CSV = os.path.join(DATA_DIR, "data/processed/engineered_business_data.csv")
LABELS_JSONL = os.path.join(DATA_DIR, "data/processed/training_labels.jsonl")
OUTPUT_DIR = os.path.join(DATA_DIR, "models/flan_t5_lora")

MODEL_NAME = "google/flan-t5-small"


def detect_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def safe_parse_list(text: str) -> List[str]:
    try:
        return list(ast.literal_eval(text))
    except Exception:
        return []


def create_business_summary(row: pd.Series) -> Dict:
    challenges = (
        ", ".join(safe_parse_list(row["Main_Challenges"]))
        if isinstance(row.get("Main_Challenges"), str)
        else "N/A"
    )

    regulatory = str(row.get("Regulatory_Constraints", ""))
    ig_comments = str(row.get("IG_Comments", ""))
    if len(regulatory) > 150:
        regulatory = regulatory[:150] + "..."
    if len(ig_comments) > 150:
        ig_comments = ig_comments[:150] + "..."

    return {
        "business_id": row["Business_ID"],
        "industry": row["Industry"],
        "revenue": f"${int(row['Revenue_Last_12M']):,}",
        "location": row["Location"],
        "growth_stage": row["Growth_Stage"],
        "online_presence": row["Online_Presence"],
        "digital_payment": row["Digital_Payment_Adoption"],
        "young_customers_pct": f"{float(row['Pct_Young_Customers']):.1f}%",
        "market_spending_growth": f"{row['Consumer_Spending_Growth_YoY']}%",
        "competition_density": int(row["Competition_Density_Score"]),
        "industry_trend": row["Industry_Growth_Trend"],
        "revenue_growth_potential": f"{float(row['Revenue_Growth_Potential']):.1f}",
        "risk_score": f"{float(row['Risk_Score']):.1f}",
        "digital_readiness": f"{float(row['Digital_Readiness']):.1f}",
        "regulatory_constraints": regulatory,
        "customer_feedback": ig_comments,
        "main_challenges": challenges,
    }


def build_prompt(summary: Dict) -> str:
    header = (
        "Given the following business profile, produce personalized recommendations as structured JSON "
        "with keys: business_id, recommendations (3-5 items), confidence_score (0-100)."
    )
    return f"{header}\nBusiness Profile:\n{json.dumps(summary, ensure_ascii=False)}"


def load_dataset() -> List[Dict]:
    df = pd.read_csv(ENGINEERED_CSV)
    df = df.drop_duplicates(subset=["Business_ID"])

    id_to_row = {row.Business_ID: row for _, row in df.iterrows()}

    samples: List[Dict] = []

    def _append_example(ex: Dict):
        biz_id = ex["business_id"]
        row = id_to_row.get(biz_id)
        if row is None:
            return
        summary = create_business_summary(row)
        input_text = build_prompt(summary)
        target_obj = {
            "business_id": biz_id,
            "recommendations": ex.get("recommendations", []),
            "confidence_score": ex.get("confidence_score", 70),
        }
        target_text = json.dumps(target_obj, ensure_ascii=False)
        samples.append({"input_text": input_text, "target_text": target_text})

    if not os.path.exists(LABELS_JSONL):
        raise FileNotFoundError(f"Training labels file not found: {LABELS_JSONL}")
    
    with open(LABELS_JSONL, "r") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            _append_example(ex)

    random.shuffle(samples)
    return samples


@dataclass
class Seq2SeqBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class SimpleSeq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, data: List[Dict], tokenizer: T5TokenizerFast, max_input_len: int = 768, max_target_len: int = 384):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        enc = self.tokenizer(
            item["input_text"],
            truncation=True,
            max_length=self.max_input_len,
            padding=False,
            return_tensors="pt",
        )
        with self.tokenizer.as_target_tokenizer():
            lab = self.tokenizer(
                item["target_text"],
                truncation=True,
                max_length=self.max_target_len,
                padding=False,
                return_tensors="pt",
            )

        return {
            "input_ids": enc.input_ids[0],
            "attention_mask": enc.attention_mask[0],
            "labels": lab.input_ids[0],
        }


def data_collator(features: List[Dict[str, torch.Tensor]]):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [f["input_ids"] for f in features], batch_first=True, padding_value=0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [f["attention_mask"] for f in features], batch_first=True, padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [f["labels"] for f in features], batch_first=True, padding_value=-100
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = detect_device()
    print(f"Using device: {device}")

    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
    base_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q", "v"],
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(base_model, lora_config)

    samples = load_dataset()
    if len(samples) == 0:
        raise RuntimeError("No training samples found. Ensure training_labels.jsonl and engineered_business_data.csv are aligned.")

    max_samples_env = int(os.getenv("MAX_TRAIN_SAMPLES", "0") or 0)
    if max_samples_env > 0:
        samples = samples[:max_samples_env]

    val_size = max(32, int(0.1 * len(samples)))
    train_data = samples[val_size:]
    val_data = samples[:val_size]

    print(f"Train examples: {len(train_data)} | Val examples: {len(val_data)}")

    train_ds = SimpleSeq2SeqDataset(train_data, tokenizer)
    val_ds = SimpleSeq2SeqDataset(val_data, tokenizer)

    num_epochs = int(os.getenv("NUM_EPOCHS", "10"))
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=num_epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
                        
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved LoRA adapter and tokenizer to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


