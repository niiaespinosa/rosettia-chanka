"""Fine-tune NLLB-200 for Spanish→Chanka Quechua (spa_Latn → quy_Latn).

NLLB-200 natively supports `quy_Latn` (Ayacucho Quechua) — unlike Qwen, it has
real Quechua pretraining exposure, which is the main lever for breaking past the
~40 ChrF plateau we hit fine-tuning a Quechua-naive decoder-only model.

Trains on the normalized v34 corpus (jsonl: {spanish, chanka}). LoRA by default
(fits the 3.3B on one 80GB card with headroom); --full for full fine-tune.
"""
import argparse, json, os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    # BSC AmericasNLP-2024 winner recipe defaults (NLLB-1.3B + LoRA r=256/a=512).
    ap.add_argument("--model-id", default="facebook/nllb-200-1.3B",
                    help="facebook/nllb-200-{distilled-600M,1.3B,3.3B}")
    ap.add_argument("--train-jsonl", default=None,
                    help="jsonl with src/tgt fields; OR use --train-parquet")
    ap.add_argument("--train-parquet", default=None,
                    help="parquet with reviewed_spanish/reviewed_chanka_quechua (or --src/--tgt-field)")
    ap.add_argument("--src-field", default="spanish")
    ap.add_argument("--tgt-field", default="chanka")
    ap.add_argument("--src-lang", default="spa_Latn")
    ap.add_argument("--tgt-lang", default="quy_Latn")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--epochs", type=float, default=10.0)
    ap.add_argument("--lr", type=float, default=2e-4)        # BSC: 2e-4 inverse-sqrt
    ap.add_argument("--scheduler", default="inverse_sqrt",
                    choices=["inverse_sqrt", "cosine", "linear"])
    ap.add_argument("--warmup-steps", type=int, default=15000)  # BSC: 15k warmup
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--eval-steps", type=int, default=2000)
    ap.add_argument("--save-total-limit", type=int, default=0)
    ap.add_argument("--lora-r", type=int, default=256)      # BSC: r=256
    ap.add_argument("--lora-alpha", type=int, default=512)  # BSC: a=512
    ap.add_argument("--full", action="store_true", help="full fine-tune instead of LoRA")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    import torch, random

    if args.train_parquet:
        import polars as pl
        df = pl.read_parquet(args.train_parquet)
        # map known parquet column names to src/tgt fields
        cols = df.columns
        sc = args.src_field if args.src_field in cols else ("reviewed_spanish" if "reviewed_spanish" in cols else cols[0])
        tc = args.tgt_field if args.tgt_field in cols else ("reviewed_chanka_quechua" if "reviewed_chanka_quechua" in cols else cols[1])
        rows = [{args.src_field: str(r[sc]).strip(), args.tgt_field: str(r[tc]).strip()}
                for r in df.select([sc, tc]).iter_rows(named=True)
                if str(r[sc]).strip() and str(r[tc]).strip()]
        print(f"loaded {len(rows)} rows from parquet {args.train_parquet} ({sc}->{tc})")
    else:
        assert args.train_jsonl, "need --train-jsonl or --train-parquet"
        rows = [json.loads(l) for l in open(args.train_jsonl) if l.strip()]

    from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              DataCollatorForSeq2Seq)
    from datasets import Dataset

    random.Random(args.seed).shuffle(rows)
    n_val = max(64, int(len(rows) * args.val_fraction))
    val, train = rows[:n_val], rows[n_val:]
    print(f"train={len(train)} val={len(val)}")

    tok = AutoTokenizer.from_pretrained(args.model_id, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)

    if not args.full:
        from peft import LoraConfig, get_peft_model
        lc = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
                        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
                        task_type="SEQ_2_SEQ_LM")
        model = get_peft_model(model, lc)
        model.print_trainable_parameters()

    def preprocess(batch):
        model_in = tok(batch[args.src_field], max_length=args.max_len, truncation=True)
        labels = tok(text_target=batch[args.tgt_field], max_length=args.max_len, truncation=True)
        model_in["labels"] = labels["input_ids"]
        return model_in

    train_ds = Dataset.from_list(train).map(preprocess, batched=True, remove_columns=list(train[0].keys()))
    val_ds = Dataset.from_list(val).map(preprocess, batched=True, remove_columns=list(val[0].keys()))
    collator = DataCollatorForSeq2Seq(tok, model=model)

    targs = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.scheduler,
        eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.eval_steps,
        save_total_limit=(None if args.save_total_limit == 0 else args.save_total_limit),
        logging_steps=50, bf16=True, seed=args.seed,
        predict_with_generate=False, report_to=[],
    )
    # transformers 5.x renamed Trainer(tokenizer=) -> processing_class; support both.
    import inspect
    trainer_kwargs = dict(model=model, args=targs, train_dataset=train_ds,
                          eval_dataset=val_ds, data_collator=collator)
    sig = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "processing_class" in sig:
        trainer_kwargs["processing_class"] = tok
    else:
        trainer_kwargs["tokenizer"] = tok
    trainer = Seq2SeqTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))
    print("done ->", args.output_dir)


if __name__ == "__main__":
    main()
