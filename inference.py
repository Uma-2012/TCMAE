from collections.abc import Iterable
import numpy as np
import torch
from sklearn import metrics
from util import misc
from util.abnormal_utils import filt

def inference(model: torch.nn.Module, data_loader: Iterable,
                    device: torch.device,
                    log_writer=None, args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Testing '

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    predictions = []
    labels = []
    videos = []
    
    for data_iter_step, (samples, grads, targets, label, vid, frame_name) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        videos += list(vid)
        labels += list(label.detach().cpu().numpy())
        samples = samples.to(device)
        grads = grads.to(device)
        targets = targets.to(device)
        
        # Standard reconstruction inference
        _, _, _, recon_error = model(samples, targets=targets, grad_mask=grads, mask_ratio=args.mask_ratio)
        
        # Frame-level score is mean of patch errors
        recon_error = recon_error.mean(dim=1).detach().cpu().numpy()
        predictions += list(recon_error)

    predictions = np.array(predictions)
    labels = np.array(labels)
    videos = np.array(videos)

    evaluate_model(predictions, labels, videos, 
                   range=38 if args.dataset == 'avenue' else 900,
                   mu=11 if args.dataset == 'avenue' else 282,
                   normalize_scores=(args.dataset != 'avenue'))

def evaluate_model(predictions, labels, videos,
                   range=302, mu=21, normalize_scores=False):

    aucs = []
    filtered_preds = []
    filtered_labels = []
    for vid in np.unique(videos):
        pred = predictions[np.array(videos) == vid]
        pred = filt(pred, range=range, mu=mu)
        if normalize_scores:
            pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))

        pred = np.nan_to_num(pred, nan=0.)

        filtered_preds.append(pred)
        lbl = labels[np.array(videos) == vid]
        filtered_labels.append(lbl)
        lbl = np.array([0] + list(lbl) + [1])
        pred = np.array([0] + list(pred) + [1])
        fpr, tpr, _ = metrics.roc_curve(lbl, pred)
        res = metrics.auc(fpr, tpr)
        aucs.append(res)

    macro_auc = np.nanmean(aucs)

    filtered_preds = np.concatenate(filtered_preds)
    filtered_labels = np.concatenate(filtered_labels)

    fpr, tpr, _ = metrics.roc_curve(filtered_labels, filtered_preds)
    micro_auc = metrics.auc(fpr, tpr)
    micro_auc = np.nan_to_num(micro_auc, nan=1.0)

    print(f"MicroAUC: {micro_auc}, MacroAUC: {macro_auc}")
    return micro_auc, macro_auc