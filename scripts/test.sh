litgpt pretrain Qwen3-30B-A3B \
   --initial_checkpoint_dir checkpoints/Qwen3-30B-A3B-Base \
   --tokenizer_dir checkpoints/Qwen3-30B-A3B-Base \
   --out_dir checkpoints/Qwen3-30B-A3B-Base-CPT \
   --data JSONLPretrain \
   --data.train_data_path data/bio-retain-corpus.jsonl \
   --data.val_split_fraction 0.01 \
   --train.max_tokens 1_000_000
