import os
from transformers import AutoTokenizer, CLIPTextModelWithProjection
from transformers.utils import logging
from huggingface_hub.utils._errors import LocalEntryNotFoundError

logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "true" # needed to suppress warning about potential deadlock
tokenizer = "openai/clip-vit-large-patch14" #"openai/clip-vit-base-patch32"
cache_dir = os.path.expanduser(os.path.join(os.environ.get("HF_HOME", "./.cache"), "clip"))

snapshots_dir = os.path.join(cache_dir, "models--openai--clip-vit-large-patch14", "snapshots")
snapshots = sorted(os.listdir(snapshots_dir))
if len(snapshots) == 0:
    raise RuntimeError(f"No snapshot found in {cache_dir}. Please download the model first.")
snapshot_dir = os.path.join(snapshots_dir, snapshots[0])

def safe_from_pretrained(model_class, pretrained_name, snapshot_dir=None):
    try:
        # First try online download
        return model_class.from_pretrained(pretrained_name, cache_dir=cache_dir).eval()
    except (OSError, LocalEntryNotFoundError) as e:
        try:
            print(f"[INFO] Online download failed. Trying offline cache at {snapshot_dir}")
            return model_class.from_pretrained(snapshot_dir, local_files_only=True).eval()
        except Exception as e2:
            raise RuntimeError(f"Failed to load model online and offline: {e2}")

lang_emb_model = safe_from_pretrained(CLIPTextModelWithProjection, tokenizer, snapshot_dir=snapshot_dir).eval()

try:
    tz = AutoTokenizer.from_pretrained(tokenizer, TOKENIZERS_PARALLELISM=True)
except (OSError, LocalEntryNotFoundError):
    print(f"[INFO] Tokenizer online load failed. Using offline cache at {snapshot_dir}")
    tz = AutoTokenizer.from_pretrained(snapshot_dir, TOKENIZERS_PARALLELISM=True, local_files_only=True)

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