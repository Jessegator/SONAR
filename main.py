"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
from data_utils import get_loader, get_custom_loader, get_wavefake_loader, get_in_the_wild_loader
from utils import get_model, create_optimizer, seed_worker, set_seed, str_to_bool
from evaluation import compute_eer, run_validation
from collections import OrderedDict
from audio_feature_extraction import LFCC
from torchaudio.transforms import Spectrogram

warnings.filterwarnings("ignore", category=FutureWarning)

def main(args: argparse.Namespace) -> None:


    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    set_seed(args.seed, config)


    output_dir = Path(args.output_dir)
    database_path = Path(config["database_path"])

    model_tag = "{}_{}_ep{}_bs{}".format(
        args.dataset, os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"], config["batch_size"])
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"

    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    model = get_model(model_config, device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if args.resume:
        state_dict = torch.load(config["model_path"], map_location=device)
        if torch.cuda.device_count() > 1:
            model.load_state_dict(state_dict)
        else:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k 
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)


    if args.dataset == 'wavefake':
        trn_loader, dev_loader, eval_loader, _ = get_wavefake_loader('/home/mikiya/data/wavefake/', args.seed, config, dataset='wavefake')
    elif args.dataset == 'Librisevoc':
        trn_loader, dev_loader, eval_loader = get_loader('/home/mikiya/data/LibriSeVoc/', args.seed, config)
    elif args.dataset == 'combo':
        libri_trn_loader, libri_dev_loader, libri_eval_loader = get_loader(database_path, args.seed, config)
        wave_trn_loader, wave_dev_loader, wave_eval_loader, _ = get_wavefake_loader('/home/mikiya/data/wavefake/', args.seed, config, dataset='wavefake')
        trn_set = libri_trn_loader.dataset + wave_trn_loader.dataset
        dev_set = libri_dev_loader.dataset + wave_dev_loader.dataset
        eval_set = libri_eval_loader.dataset + wave_eval_loader.dataset
        gen = torch.Generator()
        gen.manual_seed(args.seed)
        trn_loader = DataLoader(trn_set,
                              batch_size=config["batch_size"],
                              shuffle=True,
                              drop_last=False,
                              pin_memory=True,
                              worker_init_fn=seed_worker,
                              generator=gen)
        dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            drop_last=False,
                            pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=gen)
        eval_loader = DataLoader(eval_set,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            drop_last=False,
                            pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=gen)


    if args.eval:
        state_dict = torch.load(config["model_path"], map_location=device)
        if torch.cuda.device_count() > 1:
            model.load_state_dict(state_dict)
        else:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k  # 移除`module.`前缀
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

        print("Model loaded : {}".format(config["model_path"]))
        print("Start evaluation...")

        if args.eval_dataset == 'LibriSeVoc':
            libri_trn_loader, libri_dev_loader, libri_eval_loader = get_loader(database_path, args.seed, config)
            eval_auroc, eval_err, eval_loss = run_validation(args, libri_eval_loader, model, device)
        elif args.eval_dataset == 'wavefake':
            wavefake_trn_loader, wavefake_dev_loader, wavefake_eval_loader, _ = get_wavefake_loader('/home/mikiya/data/wavefake/',
                                                                          args.seed, config, dataset='wavefake')
            eval_auroc, eval_err, eval_loss = run_validation(args, wavefake_eval_loader, model, device)
        elif args.eval_dataset == 'in_the_wild':
            eval_loader = get_in_the_wild_loader(args.seed, config, dataset="in_the_wild", use_name=True)
            eval_auroc, eval_err, eval_loss = run_validation(args, eval_loader, model, device)
        else:
            custom_validation(args, config, model, device)

        print("DONE.")
        sys.exit(0)

    if args.finetune:
        optim_config["base_lr"] = 0.00005
        optim_config["epochs"] = 40

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    best_dev_auroc = 0.0
    best_eval_auroc = 0.0

    n_swa_update = 0  # number of snapshots of model to use in SWA
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")

    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    '''Finetuning'''
    if args.finetune:

        trn_loader, eval_loader = get_in_the_wild_loader(args.seed, config, shots=60, dataset="in_the_wild", use_name=True, fewshot=True)
        run_validation(args, eval_loader, model, device)

        for epoch in range(1, optim_config["epochs"] + 1): #
            print("Start training epoch{:03d}".format(epoch))
            train_auroc, train_eer, train_loss = train_epoch(trn_loader, model, optimizer, device, scheduler, config)
            print(f"Epoch:[{epoch + 1}]  train loss: {train_loss} train AUROC: {train_auroc} train EER: {train_eer}")
            run_validation(args, eval_loader, model, device)
            torch.save(model.state_dict(),model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, train_auroc))
        
        print("DONE.")
        sys.exit(0)
        

    # Training
    for epoch in range(1,config["num_epochs"]+1):
        print("Start training epoch{:03d}".format(epoch))
        train_auroc, train_eer, train_loss = train_epoch(trn_loader, model, optimizer, device, scheduler, config)
        dev_auroc, dev_eer, dev_loss = run_validation(args, dev_loader, model, device)
        print(f"Epoch:[{epoch+1}]  train loss: {train_loss} train AUROC: {train_auroc} train EER: {train_eer} \n dev loss:: {dev_loss} dev AUROC: {dev_auroc} dev EER: {dev_eer}")
        writer.add_scalar("loss", train_loss, epoch)
        writer.add_scalar("auroc", dev_auroc, epoch)
        writer.add_scalar("eer", dev_eer, epoch)

        if dev_auroc >= best_dev_auroc:
            print("best model find at epoch", epoch)
            best_dev_auroc = dev_auroc
            torch.save(model.state_dict(), model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_auroc))


            if str_to_bool(config["eval_all_best"]):
                eval_auroc, eval_err, eval_loss = run_validation(args, eval_loader, model, device)
                log_text = "epoch{:03d}, ".format(epoch)
                if eval_auroc > best_eval_auroc:
                    log_text += "best auroc, {:.4f}%".format(eval_auroc)
                    best_eval_auroc = eval_auroc
                    torch.save(model.state_dict(),
                               model_save_path / "best.pth")
                if len(log_text) > 0:
                    print(log_text)
                    f_log.write(log_text + "\n")

            print("Saving epoch {} for swa".format(epoch))
            optimizer_swa.update_swa()
            n_swa_update += 1
        writer.add_scalar("best_dev_auroc", best_dev_auroc, epoch)

    print("Start final evaluation")
    epoch += 1
    if n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(trn_loader, model, device=device)

    eval_auroc, eval_err, eval_loss = run_validation(args, eval_loader, model, device)
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")
    f_log.write("AUROC: {:.3f}".format(eval_auroc))
    f_log.write("EER: {:.3f}".format(eval_err))
    f_log.close()

    torch.save(model.state_dict(),model_save_path / "swa.pth")

    if eval_auroc >= best_eval_auroc:
        best_eval_auroc = eval_auroc
        torch.save(model.state_dict(),model_save_path / "best.pth")
    print("Job Finished. AUROC: {:.3f}".format(best_eval_auroc))


def custom_validation(
        args,
        config,
        model,
        device: torch.device):
    for dataset in ['elevenlabs', 'openai', 'valle','prompttts2', 'neuralspeech3', 'audiogen', 'xtts','voicebox','flashspeech', 'audiobox']:

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
                _, batch_out = model(batch_x)
                ensemble_out.append(batch_out.detach().cpu().numpy())
                batch_loss = criterion(batch_out, batch_y)
                # batch_out, = torch.max(batch)
                _, predicted = torch.max(batch_out.data, 1)
                predictions_list.extend(predicted.detach().cpu().numpy().tolist())
                # correct += (predicted == batch_y).sum().item() # 和label比较
                batch_prob = F.softmax(batch_out, dim=1).detach().to('cpu').numpy()
                batch_label = batch_y.detach().to('cpu').numpy().tolist()
                outputs_list.extend(batch_prob[:, 1].tolist())
                labels_list.extend(batch_label)
                names_list.extend(name)
                running_loss += batch_loss.item() * batch_size

        acc = np.mean(np.array(predictions_list) == np.array(labels_list))
        print(f"{dataset} acc: {acc}")

        np.save(f'./outputs/AASIST_{dataset}.npy', np.vstack(ensemble_out))

    return


def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()

    outputs_list = []
    labels_list = []
    # set objective (Loss) functions
    weight = torch.FloatTensor([0.15, 0.85]).to(device)
    criterion = nn.CrossEntropyLoss(weight)
    if config["model_config"]["architecture"] == "LCNN":
        lfcc = LFCC(480, 240, 512, 24000, 20,with_energy=False)
    elif config["model_config"]["architecture"] == "ResNet":
        spectrogram = Spectrogram(n_fft=512)

    for batch_x, batch_y, _ in tqdm(trn_loader, desc="Training", mininterval=1.0):
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        if config["model_config"]["architecture"] == "LCNN":
                batch_x = torch.unsqueeze(lfcc(batch_x).transpose(1, 2), 1).to(device)
        elif config["model_config"]["architecture"] == "ResNet":
            batch_x = torch.unsqueeze(spectrogram(batch_x),1).to(device)
        else:
            batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
        batch_loss = criterion(batch_out, batch_y)
        batch_prob = F.softmax(batch_out, dim=1).detach().to('cpu').numpy()
        batch_label = batch_y.detach().to('cpu').numpy().tolist()
        outputs_list.extend(batch_prob[:,1].tolist())
        labels_list.extend(batch_label)
        running_loss += batch_loss.item() * batch_size
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

    auroc = roc_auc_score(labels_list, outputs_list)
    eer = compute_eer(np.array(labels_list), np.array(outputs_list))
    running_loss /= num_total
    return auroc, eer, running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SONAR system")
    parser.add_argument("--config",
                        dest="config",
                        default="./config/AASIST.conf",
                        type=str,
                        help="configuration file",)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=12345,
                        help="random seed (default: 12345)")
    parser.add_argument(
        "--eval",
        action="store_true",
        type=bool,
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument(
        "--finetune",
        action="store_true",
        type=bool,
        help="finetuning?")
    parser.add_argument("--comment",
                        type=str,
                        default="",
                        help="comment to describe the saved model")
    parser.add_argument("--eval_checkpoint",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    parser.add_argument("--dataset",
                        type=str,
                        default='wavefake',
                        help="dataset")
    # parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--resume", type=bool, default=False, help="resume from checkpoint")
    parser.add_argument("--eval_dataset", type=str, default='audiobox', choices=['audiobox','flashspeech','xtts','voicebox','valle','neuralspeech3','prompttts2','openai', 'elevenlabs',  'wavefake','in_the_wild', 'LibriSeVoc'], help="dataset for evaluation")
    main(parser.parse_args())
