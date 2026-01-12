import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, Qwen2TokenizerFast, CLIPImageProcessor

# --- CONFIGURATION ---
class VLMConfig(PretrainedConfig):
    model_type = "enterprise_vlm"
    def __init__(self, llm_id="Qwen/Qwen2.5-3B-Instruct", vision_id="openai/clip-vit-large-patch14", image_token_count=256, **kwargs):
        super().__init__(**kwargs)
        self.llm_id = llm_id
        self.vision_id = vision_id
        self.image_token_count = image_token_count

# --- MODEL ARCHITECTURE ---
class Projector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))
    def forward(s
    _supports_gradient_checkpointing = True

    def __init__(self, config):
        import transformers
        super().__init__(config)
        self.vision_model = transformers.CLIPVisionModel.from_pretrained(config.vision_id)

        llm_config = transformers.AutoConfig.from_pretrained(config.llm_id)

        self.llm = transformers.Qwen2ForCausalLM.from_pretrained(
            config.llm_id, elf, x): return self.net(x)

class EnterpriseVLM(PreTrainedModel):
    config_class = VLMConfig
            config=llm_config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        self.projector = Projector(self.vision_model.config.hidden_size, self.llm.config.hidden_size)
        self._freeze()

    def _freeze(self):
        self.vision_model.requires_grad_(False)
        self.llm.requires_grad_(False)
        self.projector.requires_grad_(True)
        self.projector.to(dtype=torch.bfloat16)

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    # [CRITICAL FIX] Manually forward the enable command to the inner model
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def forward(self, input_ids, labels, pixel_values, **kwargs):
        use_cache = False if self.training else True

        with torch.no_grad():
            # CHANGE: Enable output_hidden_states to true in the forward call
            outputs = self.vision_model(pixel_values, output_hidden_states=True)
            # CHANGE: Access penultimate layer (-2).
            # Note: outputs.hidden_states is a tuple. -1 is final layer, -2 is penultimate.
            # We skip the CLS token ([:, 1:]) just like before.
            vis = outputs.hidden_states[-2][:, 1:]

        vis = self.projector(vis.to(dtype=torch.bfloat16))



        txt_emb = self.llm.get_input_embeddings()(input_ids)
        combined_emb = torch.cat((vis, txt_emb[:, self.config.image_token_count:]), dim=1)

        return self.llm(inputs_embeds=combined_emb, labels=labels, use_cache=use_cache)

# --- ROBUST DATA PROCESSING ---
class Processor:
    def __init__(self, tokenizer, img_proc, token_count, max_len=2048):
        self.tok = tokenizer
        self.img = img_proc
        self.tokens = token_count
        self.max_len = max_len
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, item):
        if "messages" not in item: return None
        try:
            if "images" in item and item["images"] and len(item["images"]) > 0:
                image = item["images"][0].convert("RGB")
                px = self.img(image, return_tensors="pt").pixel_values[0]
            else:
                return None

            full_text = ""
            for msg in item["messages"]:
                role = msg["role"]
                content = msg["content"]

                if role == "user": full_text += "<|im_start|>user\n"
                elif role == "assistant": full_text += "<|im_start|>assistant\n"

                if isinstance(content, str):
                    full_text += content
                elif isinstance(content, list):
                    for part in content:
                        if part["type"] == "text":
                            full_text += part["text"]

                full_text += "<|im_end|>\n"

            full_text += "<|endoftext|>"

            text_cap = self.max_len - self.tokens
            ids = self.tok(
                full_text,
                max_length=text_cap,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
                add_special_tokens=False
            ).input_ids[0]

            full_ids = torch.cat([torch.tensor([self.pad_id]*self.tokens, dtype=torch.long), ids])

            labels = full_ids.clone()
            labels[:self.tokens] = -100
            labels[labels == self.pad_id] = -100

            return {"input_ids": full_ids, "labels": labels, "pixel_values": px}
        except Exception as e:
            print(f"Processor Error: {e}", flush=True)
            return None
