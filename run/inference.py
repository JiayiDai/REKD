import utils.learn_settings as learn_settings
import tqdm
import utils.metrics as metrics
import torch
import numpy as np
import os
import wandb
import nns.generator as generator
import nns.encoder as encoder

def test(data_test, gen, enc, args, gen_t=None, enc_t=None):
    args.gumbel_t = args.init_t
    if args.cuda:
        gen = gen.cuda()
        enc = enc.cuda()
    if gen_t:
        gen_t, enc_t = gen_t.cuda(), enc_t.cuda()
        gen_t.eval()
        enc_t.eval()
    epoch_stat, _, rationales, texts, preds, y = run_epoch(data_test, gen, enc, args, 
                                                               gen_t=gen_t, enc_t=enc_t)
    metrics.save_rationales(texts, rationales, preds, y, args.dataset, args.model_form, run_id=args.result_path)
    print("test", epoch_stat)
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)
    metrics.performance_log(args, epoch_stat, args.result_path)
    return(epoch_stat)

def run_epoch(data_loader, gen, enc, args, is_train=False, optimizer=None, step=None, gen_t=None, enc_t=None):
    data_iter = data_loader.__iter__()#len=batch number
    losses = []
    pred_losses = []
    select_losses = []
    selects = []
    kd_r_losses = []
    kd_y_losses = []
    preds = []
    golds = []
    texts = []
    rationales = []
    rationales_t = []
    if is_train:
        gen.train()
        enc.train()
    else:
        gen.eval()
        enc.eval()
    
    target_final_t = 0.1
    total_steps = args.epochs*len(data_iter)
    gumbel_decay = np.log(args.init_t / target_final_t) / total_steps
    for batch in data_iter:#tqdm.tqdm
        if is_train:
            step += 1
            if  step % 100 == 0:
                args.gumbel_t = max(args.init_t * np.exp(-gumbel_decay * step), target_final_t)
                if args.warmup:
                    learn_settings.adjust_learning_rate(optimizer, step, 10*len(data_loader), args.init_lr)

        #batch keys: ['input_ids', 'attention_mask', 'label', 'text'] for bert models
        #['label', 'pixel_values'] for images
        if "cifar" in args.dataset:
            x = batch["pixel_values"]
            y = batch["label"]
        elif "bert" in args.model_form:
            x = batch["input_ids"]
            text = batch["text"]
            y = batch["label"]
            att_mask = batch["attention_mask"]
        else:
            raise NotImplementedError("Model form {} not yet supported!".format(args.model_form))
        if args.cuda:
            x, y = x.cuda(), y.cuda()
            if "bert" in args.model_form:
                att_mask = att_mask.cuda()
        if args.get_rationales:
            if "bert" in args.model_form:
                mask, prob, log_prob = gen(x, att_mask=att_mask)
            else:
                mask, prob, log_prob = gen(x)
            select, select_loss = gen.loss(mask)
            selects.append(select.item())
            select_losses.append(select_loss.item())
            if gen_t:
                mask_t, prob_t, log_prob_t = gen_t(x, att_mask=att_mask)
                kd_r_loss = learn_settings.get_kd_r_loss(args, prob, prob_t, log_prob, log_prob_t)
                kd_r_losses.append(kd_r_loss.item())
            #used for example demonstration
            if not is_train:
                rationales.extend(mask)
                if "cifar" in args.dataset:
                    texts.extend(x)
                elif "bert" in args.model_form:
                    texts.extend(text)
        else:
            mask = None
        if "bert" in args.model_form:
            logit = enc(x, att_mask=att_mask, mask=mask)
        else:
            logit = enc(x, mask=mask)
        pred_loss = learn_settings.get_loss(logit, y)
        if gen_t:
            logit_t = enc_t(x, att_mask=att_mask, mask=mask)
            kd_y_loss = learn_settings.get_kd_y_loss(args, logit, logit_t, args.gumbel_t)
            kd_y_losses.append(kd_y_loss.item())
        if args.get_rationales:
            loss = pred_loss + args.select_lambda*select_loss
            if gen_t:
                re_lambda = (args.init_t - args.gumbel_t) / (args.init_t - target_final_t)
                loss = re_lambda * loss + (1-re_lambda) * (args.kd_r_lambda*kd_r_loss + kd_y_loss*args.gumbel_t**2)
        else:
            loss = pred_loss
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        losses.append(loss.item())
        pred_losses.append(pred_loss.item())
        preds.extend(torch.argmax(logit, dim=-1).cpu().numpy())
        
        golds.extend(y.cpu().numpy())
    epoch_metrics = metrics.get_metrics(preds, golds, args.num_class)
    if args.get_rationales:
        epoch_stat = {'loss' : float(np.round(np.mean(losses),3)), 'pred_loss': float(np.round(np.mean(pred_losses),3)), \
        'select_loss' : float(np.round(np.mean(select_losses),3)), 'select' : float(np.round(np.mean(selects),3))}
        if gen_t:
            epoch_stat["kd_r_loss"] = float(np.round(np.mean(kd_r_losses),3))
            epoch_stat["kd_y_loss"] = float(np.round(np.mean(kd_y_losses),3))
        else:
            epoch_stat["kd_r_loss"] = float(0)
            epoch_stat["kd_y_loss"] = float(0)
    else:
        epoch_stat = {'loss': float(np.mean(pred_losses))}
    epoch_stat.update(epoch_metrics)
    return(epoch_stat, step, rationales, texts, preds, golds)
