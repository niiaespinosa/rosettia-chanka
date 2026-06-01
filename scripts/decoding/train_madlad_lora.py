"""Fine-tune MADLAD-400 (T5) with LoRA for Spanish -> Chanka Quechua, to turn the
weak zero-shot member (19.78 ChrF) into a strong, architecturally-diverse ensemble
member. MADLAD controls target language via an input prefix '<2quy>'.

Runs in the ISOLATED transformers-4.46 env (/root/madlad_venv) — transformers 5.x
mishandles MADLAD's untied embeddings. T5 is bf16-fragile, so default fp32.
"""
import argparse, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="google/madlad400-3b-mt")
    ap.add_argument("--train-parquet", required=True)
    ap.add_argument("--src-field", default="reviewed_spanish")
    ap.add_argument("--tgt-field", default="reviewed_chanka_quechua")
    ap.add_argument("--tgt-tag", default="<2quy>")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--eval-steps", type=int, default=1000)
    ap.add_argument("--lora-r", type=int, default=128)
    ap.add_argument("--lora-alpha", type=int, default=256)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    import torch, random, inspect, polars as pl
    from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq)
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    df = pl.read_parquet(args.train_parquet)
    rows = [{"src": f'{args.tgt_tag} {str(r[args.src_field]).strip()}',
             "tgt": str(r[args.tgt_field]).strip()}
            for r in df.select([args.src_field, args.tgt_field]).iter_rows(named=True)
            if str(r[args.src_field]).strip() and str(r[args.tgt_field]).strip()]
    random.Random(args.seed).shuffle(rows)
    n_val = max(64, int(len(rows) * args.val_fraction))
    val, train = rows[:n_val], rows[n_val:]
    print(f"train={len(train)} val={len(val)} dtype={args.dtype}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id, torch_dtype=dtype)
    lc = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
                    target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
                    task_type="SEQ_2_SEQ_LM")
    model = get_peft_model(model, lc)
    model.print_trainable_parameters()

    def prep(batch):
        mi = tok(batch["src"], max_length=args.max_len, truncation=True)
        lab = tok(text_target=batch["tgt"], max_length=args.max_len, truncation=True)
        mi["labels"] = lab["input_ids"]
        return mi
    train_ds = Dataset.from_list(train).map(prep, batched=True, remove_columns=["src", "tgt"])
    val_ds = Dataset.from_list(val).map(prep, batched=True, remove_columns=["src", "tgt"])
    collator = DataCollatorForSeq2Seq(tok, model=model)

    targs = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        num_train_epochs=args.epochs, warmup_ratio=0.03, lr_scheduler_type="cosine",
        eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.eval_steps, save_total_limit=2,
        logging_steps=25, bf16=(args.dtype == "bf16"), fp16=False, seed=args.seed,
        predict_with_generate=False, report_to=[])
    kw = dict(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds, data_collator=collator)
    if "processing_class" in inspect.signature(Seq2SeqTrainer.__init__).parameters:
        kw["processing_class"] = tok
    else:
        kw["tokenizer"] = tok
    trainer = Seq2SeqTrainer(**kw)
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))
    tok.save_pretrained(os.path.join(args.output_dir, "final"))
    print("done ->", args.output_dir)


if __name__ == "__main__":
    main()
