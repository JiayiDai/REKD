import utils.learn_settings as learn_settings
import tqdm
import utils.metrics as metrics
import torch
import numpy as np
import os
import wandb
import nns.generator as generator
import nns.encoder as encoder

def train(data_train, data_dev, gen, enc, args, gen_t=None, enc_t=None):
    wandb.init(project=args.id, name=args.id+"_"+str(args.rand_seed))
    if args.cuda:
        gen, enc  = gen.cuda(), enc.cuda()
    if gen_t:
        gen_t, enc_t = gen_t.cuda(), enc_t.cuda()
        gen_t.eval()
        enc_t.eval()
    optimizer = learn_settings.get_optimizer([gen, enc], args)
    step = 0
    epoch_no_improvement = 0
    dev_min_loss = float('inf')
    dev_epoch_stats = []
    dev_best_epoch = 1
    dev_pred_losses = []
    dev_select_losses = []
    dev_re_losses = []
    dev_losses = []
    kd_r_losses = []
    kd_y_losses = []
    epoch_stat = None
    if gen_t:
        critiria = 're_loss'
    else:
        critiria = 'loss'
    for epoch in range(1, args.epochs+1):
        print("-------------\nEpoch {}:\n".format(epoch))
        for mode, data_loader in [("Train", data_train), ("Dev", data_dev)]:
            epoch_stat, step, _, _, _ = run_epoch(data_loader, gen, enc, args, mode=="Train", optimizer=optimizer, step=step, gen_t=gen_t, enc_t=enc_t)
            print(mode, epoch_stat)
            if mode == "Dev":
                metrics.append_losses(args.id, epoch_stat, 
                                      dev_losses=dev_losses, dev_re_losses=dev_re_losses, 
                                      dev_pred_losses=dev_pred_losses, dev_select_losses=dev_select_losses, 
                                      kd_r_losses=kd_r_losses, kd_y_losses=kd_y_losses)
                metrics.wandb_log(args.id, epoch_stat, epoch)
        dev_epoch_stats.append(epoch_stat)
        if epoch_stat[critiria] <= dev_min_loss:
            epoch_no_improvement = 0
            dev_min_loss = epoch_stat[critiria]
            dev_best_epoch = epoch
            if not os.path.isdir(args.save_dir):
                os.makedirs(args.save_dir)
            torch.save(gen.state_dict(), args.g_path+str(args.rand_seed))
            torch.save(enc.state_dict(), args.f_path+str(args.rand_seed))
        else:
            epoch_no_improvement += 1
        print('---- Best Dev {} is {:.4f} at epoch {}'.format(
            critiria,
            dev_epoch_stats[dev_best_epoch-1][critiria],
            dev_best_epoch))
        
        if epoch_no_improvement >= args.patience:
            args.lr = args.lr*0.5
            print("Reducing learning rate to {}".format(args.lr))
            epoch_no_improvement = 0
            gen, enc = generator.Generator(args), encoder.Encoder(args)
            gen.load_state_dict(torch.load(args.g_path+str(args.rand_seed), weights_only=True))
            enc.load_state_dict(torch.load(args.f_path+str(args.rand_seed), weights_only=True))
            if args.cuda:
                gen = gen.cuda()
                enc = enc.cuda()
            optimizer = learn_settings.get_optimizer([gen, enc], args)
    wandb.finish()
    metrics.loss_log(dev_losses, dev_pred_losses, dev_select_losses, kd_r_losses, kd_y_losses, args.gumbel_t, args.result_path, dev_min_loss)
    if os.path.exists(args.save_dir):
        gen, enc = generator.Generator(args), encoder.Encoder(args)
        gen.load_state_dict(torch.load(args.g_path+str(args.rand_seed), weights_only=True))
        enc.load_state_dict(torch.load(args.f_path+str(args.rand_seed), weights_only=True))
    return(gen, enc)

def test(data_test, gen, enc, args, gen_t=None, enc_t=None):
    args.gumbel_t = args.init_t
    if args.cuda:
        gen = gen.cuda()
        enc = enc.cuda()
    if gen_t:
        gen_t, enc_t = gen_t.cuda(), enc_t.cuda()
        gen_t.eval()
        enc_t.eval()
    epoch_stat, _, rationales, texts, rationales_t = run_epoch(data_test, gen, enc, args, 
                                                               gen_t=gen_t, enc_t=enc_t)
    #metrics.save_rationales(texts, rationales, rationales_t)
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
    re_losses = []
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
                with torch.no_grad():
                    if "bert" in args.model_form:
                        mask_t, prob_t, log_prob_t = gen_t(x, att_mask=att_mask)
                    else:
                        mask_t, prob_t, log_prob_t = gen_t(x)
                kd_r_loss = learn_settings.get_kd_r_loss(args, prob, prob_t, log_prob, log_prob_t)
                kd_r_losses.append(kd_r_loss.item())
        else:
            mask = None
        if "bert" in args.model_form:
            logit = enc(x, att_mask=att_mask, mask=mask)
        else:
            logit = enc(x, mask=mask)
        pred_loss = learn_settings.get_loss(logit, y)
        if gen_t:
            with torch.no_grad():
                if "bert" in args.model_form:
                    logit_t = enc_t(x, att_mask=att_mask, mask=mask)
                else:
                    logit_t = enc_t(x, mask=mask)
            kd_y_loss = learn_settings.get_kd_y_loss(args, logit, logit_t, args.gumbel_t)
            kd_y_losses.append(kd_y_loss.item())
        if args.get_rationales:
            re_loss = pred_loss + args.select_lambda*select_loss
            loss = re_loss
            if gen_t:
                re_lambda = args.alpha_re
                if args.dist_part == 'both':
                    loss = re_lambda * re_loss + (1-re_lambda) * (args.kd_r_lambda*kd_r_loss + kd_y_loss*args.gumbel_t**2)
                elif args.dist_part == 'rationale':
                    loss = re_lambda * re_loss + (1-re_lambda) * (args.kd_r_lambda*kd_r_loss)
                elif args.dist_part == 'prediction':
                    loss = re_lambda * re_loss + (1-re_lambda) * (kd_y_loss*args.gumbel_t**2)
                else:
                    raise NotImplementedError("Distillation part {} not yet supported!".format(args.dist_part)) 
            re_losses.append(re_loss.item())
        else:
            loss = pred_loss
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        losses.append(loss.item())
        
        pred_losses.append(pred_loss.item())
        preds.extend(torch.argmax(logit, dim=-1).cpu().numpy())
        #texts.extend(text)
        golds.extend(y.cpu().numpy())
    epoch_metrics = metrics.get_metrics(preds, golds, args.num_class)
    if args.get_rationales:
        epoch_stat = {'loss' : float(np.round(np.mean(losses),3)), 'pred_loss': float(np.round(np.mean(pred_losses),3)), \
        'select_loss' : float(np.round(np.mean(select_losses),3)), 'select' : float(np.round(np.mean(selects),3))}
        epoch_stat['re_loss'] = float(np.round(np.mean(re_losses),3))
        if gen_t:
            epoch_stat["kd_r_loss"] = float(np.round(np.mean(kd_r_losses),3))
            epoch_stat["kd_y_loss"] = float(np.round(np.mean(kd_y_losses),3))
        else:
            epoch_stat["kd_r_loss"] = float(0)
            epoch_stat["kd_y_loss"] = float(0)
    else:
        epoch_stat = {'loss': float(np.mean(pred_losses))}
    epoch_stat.update(epoch_metrics)
    return(epoch_stat, step, rationales, texts, rationales_t)
