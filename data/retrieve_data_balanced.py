import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset, ClassLabel
import numpy as np
from torchvision import transforms
import random
from data.mednli_utils import load_mednli_dataset

cache_dir="hf_cache"

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataloaders(args, train_split=0.8):
    is_pre_split = False
    if args.dataset == 'imdb':
        args.num_class = 2
        dataset = load_dataset('imdb', split=['train', 'test'], cache_dir=cache_dir)
    elif args.dataset == '20news':
        args.num_class = 20
        dataset = load_dataset("SetFit/20_newsgroups", split=["train", "test"], cache_dir=cache_dir)
    elif args.dataset == 'fever':
        args.num_class = 2
    elif args.dataset == 'mednli':
        args.num_class = 3 # entailment (0), contradiction (1), neutral (2)
        raw_dataset = load_mednli_dataset(data_dir="data/med_nli", cache_dir=cache_dir)
        original_train = raw_dataset['train']
        test = raw_dataset['test']
        
        # Map existing validation set to the 'test' key in stratified_split dict 
        # so it plays nicely with the rest of the script's architecture
        stratified_split = {
            'train': raw_dataset['train'],
            'test': raw_dataset['validation']
        }
        is_pre_split = True

    elif args.dataset == 'cifar10':
        args.num_class = 10
        dataset = load_dataset("cifar10", split=['train', 'test'], cache_dir=cache_dir)
    elif args.dataset == 'cifar100':
        args.num_class = 20
        dataset = load_dataset("cifar100", split=['train', 'test'], cache_dir=cache_dir)
        dataset[0] = dataset[0].rename_column("coarse_label", "label")
        dataset[1] = dataset[1].rename_column("coarse_label", "label")
    elif args.dataset == 'cifar100-100':
        args.num_class = 100
        dataset = load_dataset("cifar100", split=['train', 'test'], cache_dir=cache_dir)
        dataset[0] = dataset[0].rename_column("fine_label", "label")
        dataset[1] = dataset[1].rename_column("fine_label", "label")
    else:
        raise Exception("Unknown dataset!")
    
    if not is_pre_split:
        # Load and verify original dataset balance
        original_train = dataset[0]
        if args.dataset == '20news':
            original_train = original_train.cast_column("label", ClassLabel(num_classes=20))
        test = dataset[1]

        original_labels = np.array(original_train['label'])

        print(f"Original class distribution: {np.unique(original_labels, return_counts=True)}")

        # Perform stratified split with balance verification
        stratified_split = original_train.train_test_split(
            test_size=1 - train_split,
            stratify_by_column='label',
            seed=args.rand_seed
        )
    else:
        original_labels = np.array(original_train['label'])
        print(f"Original class distribution: {np.unique(original_labels, return_counts=True)}")

    # Verify split balance
    def check_balance(dataset, name):
        labels = np.array(dataset['label'])
        unique, counts = np.unique(labels, return_counts=True)
        print(f"{name} class distribution: {dict(zip(unique, counts))}")

    check_balance(stratified_split['train'], "Training")
    check_balance(stratified_split['test'], "Validation")
    check_balance(test, "Testing")
    #print(stratified_split['train'][0]['text'])
    if args.dataset == 'imdb':
        return bert_loader(args, stratified_split, test, MAX_LENGTH=256)
    if args.dataset == '20news':
        return bert_loader(args, stratified_split, test, MAX_LENGTH=256)
    elif args.dataset == 'mednli':
        # Using 150 as max length (standard based on Clinical BERT paper)
        return bert_loader(args, stratified_split, test, MAX_LENGTH=128)
    elif 'cifar' in args.dataset:
        return res_loader(args, stratified_split, test)
    else:
        raise Exception("Unknown dataset!")    

def bert_loader(args, stratified_split, test, MAX_LENGTH=256):
        # Tokenization after splitting
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir=cache_dir, local_files_only=True)
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=MAX_LENGTH
            )
        train_data = stratified_split['train'].map(
            tokenize_function,
            batched=True,
            num_proc=1,
            load_from_cache_file=False
        )
        val_data = stratified_split['test'].map(
            tokenize_function,
            batched=True,
            num_proc=1
        )
        test_data = test.map(
            tokenize_function,
            batched=True,
            num_proc=1
        )

        # Collate function remains the same
        def collate_fn(batch):
            input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
            attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
            labels = torch.tensor([item['label'] for item in batch])
            tokenized_texts = [tokenizer.convert_ids_to_tokens(item['input_ids']) for item in batch]
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': labels,
                'text': tokenized_texts
            }
        g = torch.Generator()
        g.manual_seed(args.rand_seed)
        # Create dataloaders with balance verification
        train_dataloader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            generator=g,
            num_workers=0,
            collate_fn=collate_fn
        )
        val_dataloader = DataLoader(
            val_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
        test_dataloader = DataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
        return train_dataloader, val_dataloader, test_dataloader

def res_loader(args, stratified_split, test):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    def transform_cifar_train(examples):
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        pixel_values = [transform(image.convert('RGB')) for image in examples['img']]
        return {'pixel_values': pixel_values, 'label': examples['label']}
    
    def transform_cifar_test(examples):
        transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        pixel_values = [transform(image.convert('RGB')) for image in examples['img']]
        return {'pixel_values': pixel_values, 'label': examples['label']}

    # Apply transforms to datasets
    train_dataset = stratified_split['train']
    train_dataset.set_transform(transform_cifar_train)
    val_dataset = stratified_split['test']
    val_dataset.set_transform(transform_cifar_test)
    test_dataset = test
    test_dataset.set_transform(transform_cifar_test)
    g = torch.Generator()
    g.manual_seed(args.rand_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=g,
        worker_init_fn=seed_worker,
        num_workers=8,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    return train_loader, val_loader, test_loader