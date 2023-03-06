from config import PATH_TRANSFORMER_CACHE, NUM_LABELS

def load_tokenizer(model):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=PATH_TRANSFORMER_CACHE)
    return tokenizer

def load_model(args):
    from transformers import AutoConfig
    from models import FrameMultiClassification, FrameBinaryClassification
    tokenizer = load_tokenizer(args.model)
    config = AutoConfig.from_pretrained(args.model, cache_dir=PATH_TRANSFORMER_CACHE, num_labels=NUM_LABELS, output_attentions = False, output_hidden_states = False, task_specific_params={})
    if args.mode == "multi":
        model = FrameMultiClassification(config)
    elif args.mode == "ovr":
        model = FrameBinaryClassification.from_pretrained(args.model, num_labels = 1, cache_dir=PATH_TRANSFORMER_CACHE, output_attentions = False, output_hidden_states = False, task_specific_params={})
    return model, tokenizer

def get_optimizer(model, total_steps, lr):
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    optim = AdamW(model.parameters(), lr = lr)
    num_warmup = int(total_steps*0.05)
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps = num_warmup, num_training_steps = total_steps)
    return optim, scheduler

import datetime
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def convert2onehot(labels):
    labels = [int(l) for l in labels]
    onehot = [0]* NUM_LABELS
    for l in labels:
        onehot[l-1] = 1
    return {"label_onehot": onehot}

import numpy as np
def get_fscore(fmetric, precision, recall):
    if fmetric == "1":
        score = (2 * precision * recall) / (precision + recall) # F1
    elif fmetric == "0.5":
        score = 1.25 * (precision * recall) / (0.25*precision+recall) # F0.5
    elif fmetric == "2":
        score = 5 * (precision * recall) / (4*precision+recall) # F2
    score = np.nan_to_num(score, nan=-1)
    return score

def convert_logits_to_preds(preds, thre_pos, thre_neg):
    if thre_neg > thre_pos:
        print(f"neg thre larger than pos thre :( (pos:{thre_pos:.3f} < neg:{thre_neg:.3f})")
        converted = [2 if p > thre_neg else 0 for p in preds]
    else:
        converted = []
        for p in preds:
            if p >=  thre_pos:
                converted.append(2)
            elif p <= thre_neg:
                converted.append(0)
            else:
                converted.append(1)
    return converted

def binarize_preds(preds, thre):
    preds= (np.array(preds)>thre)*1
    return preds

def load_data(args, tokenizer):
    from data_loaders import load_mfc_data
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
    data, indomains = load_mfc_data(args.data, args.foldidx, bool_fewshot=args.fewshot)
    if args.bool_final_training:
        from datasets import concatenate_datasets
        assert 'test' in data
        print(len(data['train']))
        data['train'] = concatenate_datasets([data['train'], data['test']])
        print(len(data['train']))
        data['test'] = None

    dataloader = {}
    for split in data:
        if data[split] is None:
            dataloader[split] = None
            continue
        if args.bool_test and len(data[split])>args.num_test:
            data[split] = data[split].select(range(args.num_test))
        data[split] = data[split].map(lambda e: tokenizer(e['sentence'], add_special_tokens=True, truncation=True, padding='max_length', max_length=args.max_len, return_attention_mask=True), batched=True)
        data[split] = data[split].map(lambda e: convert2onehot(e['label']))
        data[split].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label_onehot'])
        # dataloader[split] = data[split]
        dataloader[split] = DataLoader(data[split], batch_size=args.batch_size, sampler = RandomSampler(data[split]) if split == "train" else SequentialSampler(data[split]))

    dataloader_indomain = None
    if indomains is not None:
        dataloader_indomain = {}
        for split in indomains:
            if indomains[split] is None:
                dataloader_indomain[split] = None
                continue
            if args.bool_test:
                indomains[split] = indomains[split].select(range(args.num_test))
            indomains[split] = indomains[split].map(lambda e: tokenizer(e['sentence'], add_special_tokens=True, truncation=True, padding='max_length', max_length=args.max_len, return_attention_mask=True), batched=True)
            indomains[split] = indomains[split].map(lambda e: convert2onehot(e['label']))
            indomains[split].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label_onehot'])
            dataloader_indomain[split] = DataLoader(indomains[split], batch_size=args.batch_size, sampler = RandomSampler(indomains[split]) if split == "train" else SequentialSampler(indomains[split]))
    return dataloader, dataloader_indomain