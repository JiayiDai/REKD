import os
from datasets import load_dataset

def load_mednli_dataset(data_dir="data/med_nli", cache_dir="hf_cache"):
    """
    Loads and preprocesses the MedNLI dataset from local JSONL files.
    Returns a dataset dictionary with 'train', 'validation', and 'test' splits.
    """
    data_files = {
        "train": f"{data_dir}/mli_train_v1.jsonl",
        "validation": f"{data_dir}/mli_dev_v1.jsonl",
        "test": f"{data_dir}/mli_test_v1.jsonl"
    }
    
    # Load local JSONL files
    raw_dataset = load_dataset("json", data_files=data_files, cache_dir=cache_dir)
    
    label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    
    # Filter out rows with missing/invalid gold labels and map text/labels
    raw_dataset = raw_dataset.filter(lambda x: x.get('gold_label') in label_map)
    mapped_dataset = raw_dataset.map(lambda x: {
        'label': label_map[x['gold_label']],
        # Creating a single text field as a fallback, though bert_loader handles sentence1/2
        'text': str(x.get('sentence1', '')) + " [SEP] " + str(x.get('sentence2', '')) 
    })
    return mapped_dataset