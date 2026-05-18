import torch
import os
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from typing import List

class ParaphraserModel:
    def __init__(self, model_config: dict, device: str = None):
        self.config = model_config
        self.pretrained = model_config["pretrained_name"]
        self.max_length = model_config.get("max_length", 128)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.pretrained).to(self.device)

    def train(self, train_loader, val_loader, trainer_config: dict = None, metrics_config: dict = None):
        train_dataset = Dataset.from_pandas(train_loader.dataset)
        val_dataset = Dataset.from_pandas(val_loader.dataset)

        def tokenize_fn(examples):
            model_inputs = self.tokenizer(
                examples["original"], max_length=self.max_length, truncation=True, padding=True
            )
            labels = self.tokenizer(
                examples["paraphrase"], max_length=self.max_length, truncation=True, padding=True
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        train_dataset = train_dataset.map(tokenize_fn, batched=True)
        val_dataset = val_dataset.map(tokenize_fn, batched=True)

        lr = float(self.config.get("learning_rate", 3e-4))
        output_dir = trainer_config.get("output_dir", "./saves")
        args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=trainer_config.get("batch_size", self.config.get("batch_size", 8)),
            per_device_eval_batch_size=trainer_config.get("batch_size", self.config.get("batch_size", 8)),
            learning_rate=lr,
            num_train_epochs=int(self.config.get("num_epochs", 3)),
            warmup_steps=int(self.config.get("warmup_steps", 0)),
            weight_decay=float(self.config.get("weight_decay", 0.01)),
            gradient_accumulation_steps=int(self.config.get("gradient_accumulation_steps", 1)),
            logging_strategy=trainer_config.get("logging_strategy", "epoch"),
            eval_strategy=trainer_config.get("eval_strategy", "epoch"),
            save_strategy=trainer_config.get("save_strategy", "epoch"),
            save_total_limit=int(trainer_config.get("save_total_limit", 2)),
            predict_with_generate=True,
            fp16=bool(trainer_config.get("fp16", False)),
            report_to="none",
            load_best_model_at_end=trainer_config.get("load_best_model_at_end", True),
            metric_for_best_model=trainer_config.get("metric_for_best_model", "eval_loss"),
            greater_is_better=trainer_config.get("greater_is_better", False),
        )

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        if metrics_config and "metrics" in metrics_config:
            from ..metrics import compute_metrics_for_trainer
            compute_metrics_fn = compute_metrics_for_trainer(
                self.tokenizer, metrics_config["metrics"], metrics_config
            )
        else:
            compute_metrics_fn = None

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn,
        )

        checkpoint = None
        if trainer_config.get("resume_from_checkpoint", True):
            if os.path.isdir(output_dir) and any(
                f.startswith("checkpoint-") for f in os.listdir(output_dir)
            ):
                checkpoint = True
                print("Найден чекпоинт, продолжаем обучение...")

        trainer.train(resume_from_checkpoint=checkpoint)
        self.trainer = trainer
        return trainer.state.log_history

    def evaluate(self, test_loader, metrics_config=None):
        test_dataset = Dataset.from_pandas(test_loader.dataset)
        def tokenize_fn(examples):
            model_inputs = self.tokenizer(
                examples["original"], max_length=self.max_length, truncation=True, padding=True
            )
            labels = self.tokenizer(
                examples["paraphrase"], max_length=self.max_length, truncation=True, padding=True
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        test_dataset = test_dataset.map(tokenize_fn, batched=True)
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        eval_batch_size = 8
        if metrics_config:
            eval_batch_size = int(metrics_config.get("batch_size", 8))

        args = Seq2SeqTrainingArguments(
            output_dir="./tmp_eval",
            per_device_eval_batch_size=eval_batch_size,
            predict_with_generate=True,
            report_to="none",
        )
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=args,
            eval_dataset=test_dataset,
            data_collator=data_collator,
        )
        metrics = trainer.evaluate()
        return metrics

    def generate(self, texts: List[str], num_return_sequences=1) -> List[str]:
        inputs = self.tokenizer(texts, max_length=self.max_length, truncation=True, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.generate(
            **inputs,
            max_length=self.max_length,
            min_length=6,
            do_sample=True,
            temperature=0.85,
            top_p=0.9,
            repetition_penalty=1.2,
            num_return_sequences=num_return_sequences,
            early_stopping=True
        )
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if num_return_sequences > 1:
            return [decoded[i:i+num_return_sequences] for i in range(0, len(decoded), num_return_sequences)]
        return decoded

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)