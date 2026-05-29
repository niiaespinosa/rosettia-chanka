"""Continued pretraining (CPT / language-adaptive pretraining) of Qwen3.5-9B on
raw monolingual Quechua text, to inject quy language knowledge the base model
lacks (the diagnosed ~40-ChrF ceiling cause).

Unsloth CPT recipe: train embed_tokens + lm_head (with a smaller
embedding_learning_rate) IN ADDITION to LoRA on the linear layers, so the token
representations adapt to Quechua's heavy subword fragmentation. Plain causal-LM
next-token objective on packed raw text (no instruction format).

Output: a LoRA adapter to use as the base for the downstream chanka translation
SFT chain (and as a stronger, quy-aware ensemble member).
"""
import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="unsloth/Qwen3.5-9B")
    ap.add_argument("--adapter-path", default=None,
                    help="optional starting adapter (e.g. v24f broad) to merge first")
    ap.add_argument("--text-file", required=True, help="raw quy text, one segment/line")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-seq-length", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=128)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--embedding-lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments, is_bfloat16_supported
    from datasets import load_dataset

    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.model_id, max_seq_length=args.max_seq_length,
        dtype=None, load_in_4bit=False)
    if args.adapter_path:
        # load + merge an existing adapter into the base, then add fresh CPT LoRA
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()

    # CPT: include embed_tokens + lm_head in the trainable modules (Unsloth recipe)
    model = FastLanguageModel.get_peft_model(
        model, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                        "embed_tokens", "lm_head"],
        use_gradient_checkpointing="unsloth", random_state=args.seed)

    ds = load_dataset("text", data_files={"train": args.text_file}, split="train")
    EOS = tok.eos_token

    def add_eos(ex):
        return {"text": ex["text"] + EOS}
    ds = ds.map(add_eos)

    trainer = UnslothTrainer(
        model=model, tokenizer=tok, train_dataset=ds,
        dataset_text_field="text", max_seq_length=args.max_seq_length,
        dataset_num_proc=4, packing=True,
        args=UnslothTrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            embedding_learning_rate=args.embedding_lr,   # << key CPT knob
            bf16=is_bfloat16_supported(), fp16=not is_bfloat16_supported(),
            logging_steps=20, optim="adamw_8bit", weight_decay=0.01,
            lr_scheduler_type="cosine", seed=args.seed,
            save_strategy="steps", save_steps=args.save_steps, save_total_limit=3,
            output_dir=args.output_dir, report_to=[]),
    )
    trainer.train()
    model.save_pretrained(os.path.join(args.output_dir, "final"))
    tok.save_pretrained(os.path.join(args.output_dir, "final"))
    print("CPT done ->", args.output_dir)


if __name__ == "__main__":
    main()
