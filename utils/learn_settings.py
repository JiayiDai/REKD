import torch
import torch.nn as nn
import torch.nn.functional as F

def get_optimizer(models, args):
    params = []
    for model in models:
        params.extend([param for param in model.parameters() if param.requires_grad])
    if "resnet" in args.model_form.lower():
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    #optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer

CELoss = nn.CrossEntropyLoss()
def get_loss(logit,y):
    return CELoss(logit, y)

KLDLoss = nn.KLDivLoss(reduction="batchmean", log_target=True)
def get_kd_r_loss(args, prob_s, prob_t, log_prob_s, log_prob_t):
    if args.hard_label:
        return get_kd_r_loss_CE(prob_s, prob_t, hard=args.hard_label)
    return KLDLoss(log_prob_s, log_prob_t)

def get_kd_y_loss(args, logit_s, logit_t, T):
    if args.hard_label:
        return get_kd_y_loss_CE(logit_s, logit_t, T)
    student_log_probs = F.log_softmax(logit_s / T, dim=-1)
    teacher_log_probs = F.log_softmax(logit_t / T, dim=-1)
    return KLDLoss(student_log_probs, teacher_log_probs)

def CE_loss(prob_pred, prob_true, eps=1e-8):
    #probability inputs
    #return shape batch_size*length
    prob_pred = torch.clamp(prob_pred, eps, 1. - eps)
    loss = -torch.sum(prob_true * torch.log(prob_pred), dim=-1)
    return loss

def get_kd_r_loss_CE(prob_pred, prob_true, hard=False):
    #cross_entropy_loss for matching feature selection (i.e., sum over features)
    if hard:
        loss = CE_loss(prob_pred, prob_to_onehot(prob_true))
    else:
        loss = CE_loss(prob_pred, prob_true)    
    loss = torch.sum(loss, dim=1)
    return loss.mean()

def get_kd_y_loss_CE(logit_s, logit_t, T):
    student_probs = F.softmax(logit_s / T, dim=-1)
    teacher_probs = F.softmax(logit_t / T, dim=-1)
    loss = CE_loss(student_probs, teacher_probs)
    return loss.mean()

def prob_to_onehot(probs):
    """
    Convert probability distribution to one-hot encoding
    probs: tensor of shape (..., num_classes=2)
    returns: one-hot tensor of same shape as probs
    """
    indices = torch.argmax(probs, dim=-1)
    one_hot = F.one_hot(indices, num_classes=2)#bernoulli
    return one_hot.float()

def get_lr_with_warmup(current_step, warmup_steps, max_lr):
    initial_lr = max_lr/10
    if current_step < warmup_steps:
        return initial_lr + (max_lr - initial_lr) * (current_step / warmup_steps)
    return max_lr

def adjust_learning_rate(optimizer, current_step, warmup_steps, max_lr):
    lr = get_lr_with_warmup(current_step, warmup_steps, max_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_rationales(mask, text):
    if mask is None:
        return text
    masked_text = []
    for i, t in enumerate(text):
        sample_mask = list(mask.data[i])
        original_words = t#.split()
        words = [ w if m  > .5 else "_" for w,m in zip(original_words, sample_mask) ]
        masked_sample = " ".join(words)
        masked_text.append(masked_sample)
    return masked_text
