import os
from transformers import AutoTokenizer, CLIPTextModelWithProjection
from transformers.utils import logging
from huggingface_hub.utils._errors import LocalEntryNotFoundError

logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "true" # needed to suppress warning about potential deadlock
tokenizer = "openai/clip-vit-large-patch14" #"openai/clip-vit-base-patch32"
local_cache_dir = os.path.expanduser(os.path.join(os.environ.get("HF_HOME", "./.cache"), "clip"))

def safe_from_pretrained(model_class, pretrained_name, **kwargs):
    try:
        # online + cache
        return model_class.from_pretrained(pretrained_name, **kwargs)
    except (OSError, LocalEntryNotFoundError):
        # fallback to offline cache only
        print(f"[INFO] Online download failed. Trying offline cache at {local_cache_dir}")
        return model_class.from_pretrained(local_cache_dir, local_files_only=True, **kwargs)

lang_emb_model = safe_from_pretrained(
    CLIPTextModelWithProjection,
    tokenizer,
    cache_dir=local_cache_dir
).eval()

try:
    tz = AutoTokenizer.from_pretrained(tokenizer, TOKENIZERS_PARALLELISM=True)
except (OSError, LocalEntryNotFoundError):
    print(f"[INFO] Tokenizer online load failed. Using offline cache at {local_cache_dir}")
    tz = AutoTokenizer.from_pretrained(local_cache_dir, TOKENIZERS_PARALLELISM=True, local_files_only=True)

LANG_EMB_OBS_KEY = "lang_emb"

def get_lang_emb(lang):
    if lang is None:
        return None
    
    tokens = tz(
        text=lang,                   # the sentence to be encoded
        add_special_tokens=True,             # Add [CLS] and [SEP]
        max_length=25,  # maximum length of a sentence
        padding="max_length",
        return_attention_mask=True,        # Generate the attention mask
        return_tensors="pt",               # ask the function to return PyTorch tensors
    )
    lang_emb = lang_emb_model(**tokens)['text_embeds'].detach()[0]

    return lang_emb

def get_lang_emb_shape():
    return list(get_lang_emb('dummy').shape)