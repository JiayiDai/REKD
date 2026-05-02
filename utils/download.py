from datasets import load_dataset
from transformers import ViTModel
from transformers import AutoModel, AutoTokenizer
import os

cache_path = "hf_cache"

def download_bert():
    models_to_download = [
        "prajjwal1/bert-tiny",   # L=2, H=128
        "prajjwal1/bert-mini",   # L=4, H=256
        "prajjwal1/bert-small",  # L=4, H=512
        "bert-base-uncased"
    ]
    for model_name in models_to_download:
        AutoModel.from_pretrained(model_name, cache_dir=cache_path)
        AutoTokenizer.from_pretrained(model_name, cache_dir=cache_path)
    print("All downloads complete!")

def download_vit():
    models_to_download = [
        "google/vit-base-patch16-224",
        "WinKawaks/vit-small-patch16-224",
        "WinKawaks/vit-tiny-patch16-224"
    ]
    for model_name in models_to_download:
        model = ViTModel.from_pretrained(model_name, cache_dir=cache_path)
    print("All downloads complete!")

load_dataset('imdb', split=['train', 'test'], cache_dir=cache_path)
load_dataset("cifar10", split=['train', 'test'], cache_dir=cache_path)
load_dataset("cifar100", split=['train', 'test'], cache_dir=cache_path)
download_bert()
download_vit()