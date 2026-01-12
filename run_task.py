import sys

# [CRITICAL] This function runs on EVERY TPU CORE.
# The patch MUST be applied here to work.
def _mp_fn(index):
    import os
    import torch
    import torch_xla

    # [FIX] Apply Monkey-Patch inside the worker process!
    # This tricks PyTorch into finding 'torch.xla' for checkpointing.
    if not hasattr(torch, "xla"):
        torch.xla = torch_xla

    import torch_xla.core.xla_model as xm
    from datasets import load_dataset
    from transformers import TrainingArguments, Trainer, TrainerCallback
    from filelock import FileLock
    import vlm_lib

    device = xm.xla_device()
    if index == 0: print(f"[CORE {index}] TPU Initialized: {device}", flush=True)

    torch.manual_seed(99)

    conf = vlm_lib.VLMConfig()
    tok = vlm_lib.Qwen2TokenizerFast.from_pretrained(conf.llm_id)

    if tok.pad_token is None:
        tok.pad_token_id = 151643

    img_proc = vlm_lib.CLIPImageProcessor.from_pretrained(conf.vision_id)

    if index == 0: print(">>> Loading Dataset...", flush=True)
    with FileLock("/tmp/data_load.lock"):
        raw_ds = load_dataset("HuggingFaceH4/llava-instruct-mix-vsft", split="train")

    proc = vlm_lib.Processor(tok, img_proc, conf.image_token_count, max_len=2048)

    class CleanDataset(torch.utils.data.Dataset):
        def __init__(self, ds):
            self.ds = ds
        def __len__(self): return len(self.ds)
        def __getitem__(self, idx):
            res = proc(self.ds[idx])
            if res is not None: return res
            return proc(self.ds[0])

    train_ds = CleanDataset(raw_ds)
    if index == 0:
        print("\n>>> DATASET SANITY CHECK (Sample 0) <<<", flush=True)
        try:
            sample = train_ds[0]
            ids = sample["input_ids"]
            # Filter out padding (-100 or pad_token) for clean reading
            valid_ids = ids[ids != tok.pad_token_id]
            decoded_text = tok.decode(valid_ids)

            print(f"Token Length: {len(ids)}")
            print(f"Decoded Text:\n{decoded_text}\n", flush=True)

            if len(valid_ids) < 10:
                print(">>> WARNING: Sample looks suspiciously empty!", flush=True)
        except Exception as e:
            print(f">>> CRITICAL DATASET ERROR: {e}", flush=True)
    # ------------------------------
    model = vlm_lib.EnterpriseVLM(conf).to(device)
    ckpt_path = "/kaggle/input/clipqwenstep10000/projector_v1.pt"

    if os.path.exists(ckpt_path):
        if index == 0: print(f">>> RESUMING FROM: {ckpt_path}", flush=True)
        # Load weights onto CPU first, then model moves them to TPU automatically
        state = torch.load(ckpt_path, map_location="cpu")
        model.projector.load_state_dict(state)
    else:
        if index == 0: print(f">>> WARNING: {ckpt_path} NOT FOUND. Starting fresh.", flush=True)
    class PrinterCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_local_process_zero:
                output = {k: v for k, v in logs.items() if k in ['loss', 'epoch', 'learning_rate']}
                if output:
                    print(f"[Step {state.global_step}] Stats: {output}", flush=True)




    args = TrainingArguments(
        output_dir="/kaggle/working/checkpoints",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=20000,
        learning_rate=5e-5,
        max_grad_norm=0.5,
        num_train_epochs=1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        save_strategy="no",
        save_total_limit=1,
        logging_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adafactor",
        disable_tqdm=True ,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        local_rank=index,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, callbacks=[PrinterCallback])

    if index == 0: print(">>> Starting Training Loop...", flush=True)
    trainer.train()

    if index == 0:
        print(">>> Saving Adapter...", flush=True)
        state = {k: v.cpu() for k, v in model.projector.state_dict().items()}
        torch.save(state, "projector_v2.pt")

if __name__ == "__main__":
    os.environ["PJRT_DEVICE"] = "TPU"
    os.environ["XLA_USE_BF16"] = "1"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print(">>> [MASTER] Ensuring model files are cached...", flush=True)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download("Qwen/Qwen2.5-3B-Instruct")
        snapshot_download("openai/clip-vit-large-patch14")
    except Exception as e:
        print(f"Pre-download warning: {e}")

    import torch_xla.distributed.xla_multiprocessing as xmp
    print(">>> [LAUNCHER] Spawning XLA...", flush=True)
    xmp.spawn(_mp_fn, args=(), start_method='spawn')
