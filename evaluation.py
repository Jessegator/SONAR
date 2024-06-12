import sys
import os
import torch
import numpy as np
from data_utils import get_custom_loader
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from audio_feature_extraction import LFCC
from torchaudio.transforms import Spectrogram
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


def compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return frr, far, thresholds


def compute_eer(ground_truth, predictions):
    """
    Expecting ground_truth and predictions to be numpy arrays of the same length;
    Defining deepfakes (ground_truth == 1) as target scores and bonafide (ground_truth == 0) as nontarget scores.
    """
    assert ground_truth.shape == predictions.shape, "ground_truth and predictions must have the same shape"
    assert len(ground_truth.shape) == 1, "ground_truth and predictions must be 1D arrays"

    target_scores = predictions[ground_truth == 1]
    nontarget_scores = predictions[ground_truth == 0]

    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]

def run_validation(
        config: Dict,
        data_loader: DataLoader,
        model,
        device: torch.device):
    outputs_list = []
    labels_list = []
    predictions_list = []
    ensemble_out = []
    num_total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    if config["model_config"]["architecture"] == "LCNN":
        lfcc = LFCC(480, 240, 512, 24000, 20,with_energy=False)
    elif config["model_config"]["architecture"] == "ResNet":
        spectrogram = Spectrogram(n_fft=512)
    model.eval()
    names_list = []
    with torch.no_grad():
        for batch_x, batch_y, name in tqdm(data_loader, desc="Evaluating", mininterval=1.0):
            batch_size = batch_x.size(0)
            num_total += batch_size
            if config["model_config"]["architecture"] == "LCNN":
                batch_x = torch.unsqueeze(lfcc(batch_x).transpose(1, 2), 1).to(device)
            elif config["model_config"]["architecture"] == "ResNet":
                batch_x = torch.unsqueeze(spectrogram(batch_x),1).to(device)
            else:
                batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            batch_out = model(batch_x)
            ensemble_out.append(batch_out.detach().cpu().numpy())
            batch_loss = criterion(batch_out, batch_y)
            # batch_out, = torch.max(batch)
            _, predicted = torch.max(batch_out.data, 1)
            predictions_list.extend(predicted.detach().cpu().numpy().tolist())
            batch_prob = F.softmax(batch_out, dim=1).detach().to('cpu').numpy()
            batch_label = batch_y.detach().to('cpu').numpy().tolist()
            outputs_list.extend(batch_prob[:, 1].tolist())
            labels_list.extend(batch_label)
            names_list.extend(name)
            running_loss += batch_loss.item() * batch_size

    acc = np.mean(np.array(predictions_list) == np.array(labels_list))
    print(f"Acc: {acc}")
    prec = precision_score(labels_list, predictions_list)
    recall = recall_score(labels_list, predictions_list)
    f1 = f1_score(labels_list, predictions_list)
    auroc = roc_auc_score(labels_list, outputs_list)
    eer, _ = compute_eer(np.array(labels_list), np.array(outputs_list))
    print(f"Precision: {prec}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    print(f"AUROC: {auroc}")
    print(f"EER: {eer}")

    # np.save(f'./outputs/AASIST_{args.eval_dataset}.npy', np.vstack(ensemble_out))

    return auroc, eer, running_loss

def custom_validation(
        args,
        config,
        model,
        device: torch.device):
    for dataset in ['elevenlabs', 'openai', 'valle','prompttts2', 'neuralspeech3', 'mixed_fake_audio', 'xtts','voicebox','flashspeech']:

        data_loader = get_custom_loader(args.seed, config, dataset=dataset)  # Custom dataset

        outputs_list = []
        labels_list = []
        predictions_list = []
        ensemble_out = []
        num_total = 0
        running_loss = 0.0
        correct = 0.0
        criterion = nn.CrossEntropyLoss()
        model.eval()
        names_list = []
        with torch.no_grad():
            for batch_x, batch_y, name in tqdm(data_loader, desc="Evaluating", mininterval=1.0):
                batch_size = batch_x.size(0)
                num_total += batch_size
                batch_x = batch_x.to(device)
                batch_y = batch_y.view(-1).type(torch.int64).to(device)
                batch_out = model(batch_x)
                ensemble_out.append(batch_out.detach().cpu().numpy())
                batch_loss = criterion(batch_out, batch_y)
                # batch_out, = torch.max(batch)
                _, predicted = torch.max(batch_out.data, 1)
                predictions_list.extend(predicted.detach().cpu().numpy().tolist())
                # correct += (predicted == batch_y).sum().item()  # 和label比较
                batch_prob = F.softmax(batch_out, dim=1).detach().to('cpu').numpy()
                batch_label = batch_y.detach().to('cpu').numpy().tolist()
                outputs_list.extend(batch_prob[:, 1].tolist())
                labels_list.extend(batch_label)
                names_list.extend(name)
                running_loss += batch_loss.item() * batch_size

        acc = np.mean(np.array(predictions_list) == np.array(labels_list))
        print(f"{dataset} acc: {acc}")

        np.save(f'./outputs/Rawnet_{dataset}.npy', np.vstack(ensemble_out))

    return
