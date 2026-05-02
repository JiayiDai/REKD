import sklearn.metrics
import numpy, json, wandb, os, torch
import matplotlib.pyplot as plt

def get_metrics(preds, golds, num_classes):
    if num_classes == 2:
        avg = 'binary'
    else:
        avg = 'weighted'
    metrics = {}
    metrics['accuracy'] = sklearn.metrics.accuracy_score(y_true=golds, y_pred=preds)
    #metrics['confusion_matrix'] = sklearn.metrics.confusion_matrix(y_true=golds,y_pred=preds)
    metrics['precision'] = sklearn.metrics.precision_score(y_true=golds, y_pred=preds, average=avg)
    metrics['recall'] = sklearn.metrics.recall_score(y_true=golds,y_pred=preds, average=avg)
    metrics['f1'] = sklearn.metrics.f1_score(y_true=golds,y_pred=preds, average=avg)
    return(formatting(metrics))

def formatting(metrics, decimals=3):
    for key, value in metrics.items():
        if key != 'confusion_matrix':
            metrics[key] = float(numpy.round(value, decimals))
        else:
            metrics[key] = numpy.round(value, decimals)
    return(metrics)

def performance_log(args, metrics, file_name):
    hyperparams = {"seed":args.rand_seed, "select_lambda":args.select_lambda, "lr":args.init_lr, 
                   "epoch":args.epochs, "batch_size":args.batch_size, "init_t":args.init_t}
    performance_log = {"model":args.model_form, "dataset":args.dataset, "metrics":metrics,
                       "hyperparam":hyperparams}
    with open(file_name, "a") as f:
        json.dump(performance_log, f)
        f.write("\n")

def loss_log(train_losses, train_pred_losses, train_select_losses, kd_r_losses, kd_y_losses, gumbel_t, file_name, dev_min_critiria):
    all_losses = {"gumbel_t":gumbel_t, "loss":train_losses, "pred_loss":train_pred_losses, 
                  "select_loss":train_select_losses, "kd_r_loss":kd_r_losses, "kd_y_loss":kd_y_losses, "dev_min_critiria":dev_min_critiria}
    with open(file_name, "a") as f:
        json.dump(all_losses, f)
        f.write("\n")

def append_losses(task_id, epoch_stat, dev_losses=None, dev_re_losses=None, dev_pred_losses=None, dev_select_losses=None, kd_r_losses=None, kd_y_losses=None):
    dev_losses.append(epoch_stat['loss'])
    if "_re" in task_id or "_kd" in task_id:
        dev_pred_losses.append(epoch_stat['pred_loss'])
        dev_select_losses.append(epoch_stat['select_loss'])
        dev_re_losses.append(epoch_stat['re_loss'])
    if "_kd" in task_id:
        kd_r_losses.append(epoch_stat['kd_r_loss'])
        kd_y_losses.append(epoch_stat['kd_y_loss'])

def wandb_log(task_id, epoch_stat, epoch):
    wandb.log({"dev_loss":epoch_stat['loss']}, step=epoch)
    if "_re" in task_id or "_kd" in task_id: 
        wandb.log({"pred_loss":epoch_stat['pred_loss'],
                   "select_loss":epoch_stat['select_loss'],
                   "select":epoch_stat['select'],
                   "re_loss":epoch_stat['re_loss']}, step=epoch)
    if "_kd" in task_id:
        wandb.log({"kd_r_loss":epoch_stat['kd_r_loss'],
                   "kd_y_loss":epoch_stat['kd_y_loss']}, step=epoch)
        
def save_rationales(x, r, preds, y, dataset, model_form, run_id="not specified", text_filename=os.path.join("rationales", "text_rationale.txt"), img_folder_path=os.path.join("rationales", "images")):
    assert len(x) == len(r)
    if "vit" in model_form:#images
        for indx in range(len(x)):#
            visualize_patches(x[indx], r[indx], indx, os.path.join(img_folder_path, dataset), model_form)
    else:#texts
        with open(text_filename, 'w') as file:
            file.write(f"run_id: {run_id}\n")
            for i in range(len(x)):
                file.write(f"text: {x[i]}\n")
                file.write(f"m: {r[i]}\n")
                file.write(f"gold: {y[i]}\n")
                file.write(f"pred: {preds[i]}\n")
                file.write(f"r: {apply_mask_to_tokens(x[i], r[i])}\n")
                file.write("=========================================\n")
                if i < len(x) - 1:
                    file.write("\n")

def denormalize(tensor, model_form):
    """
    Reverses the ImageNet normalization for visualization.
    Input: Tensor (C, H, W)
    Output: Numpy Array (H, W, C) clipped to [0, 1]
    """
    if "vit" in model_form:
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
    img = tensor.clone().detach().cpu().numpy()
    img = img.transpose(1, 2, 0)
    img = (img * STD) + MEAN
    return numpy.clip(img, 0, 1)

def visualize_patches(pixel_values, mask, indx, img_folder_path, model_form, patch_size=16):
    """
    pixel_values: Tensor of shape (3, 224, 224)
    mask: Tensor or Array of shape (196,) or (1, 196). 
          1 = keep, 0 = mask out
    patch_size: 16 for 224 images
    """
    
    original_img = denormalize(pixel_values, model_form)
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    mask = mask.flatten()
    
    # Calculate grid size (sqrt(196) = 14)
    grid_size = int(numpy.sqrt(mask.shape[0])) # 14
    
    # Reshape mask to 2D grid (14, 14)
    mask_grid = mask.reshape(grid_size, grid_size)
    
    # Upsample mask from (14, 14) -> (224, 224)
    # We repeat the rows and columns by the patch_size
    mask_upsampled = mask_grid.repeat(patch_size, axis=0).repeat(patch_size, axis=1)
    
    # Expand dims to match image channels for multiplication: (224, 224, 1)
    mask_upsampled_3ch = mask_upsampled[:, :, numpy.newaxis]

    masked_img = original_img * mask_upsampled_3ch
    
    os.makedirs(img_folder_path, exist_ok=True)
    original_save_path = os.path.join(img_folder_path, f"{indx}_original.png")
    plt.imsave(original_save_path, original_img)
    masked_save_path = os.path.join(img_folder_path, f"{indx}_masked.png")
    plt.imsave(masked_save_path, masked_img)


def apply_mask_to_tokens(tokens, mask):
    """
    Replaces tokens with '_' where the mask is 0, keeping others intact.
    
    Args:
        tokens (list of str): The input tokens.
        mask (torch.Tensor): Binary tensor.
        
    Returns:
        str: Text with kept tokens and underscores.
    """
    if len(tokens) != len(mask):
        raise ValueError(f"Length mismatch: Tokens ({len(tokens)}) vs Mask ({len(mask)})")

    mask_values = mask.tolist()
    result_tokens = [
        token if val == 1 else "_" 
        for token, val in zip(tokens, mask_values)
    ]
    
    return " ".join(result_tokens)