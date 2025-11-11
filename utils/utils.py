import os
import sys
import json
import pytz
import argparse
import datetime
from sklearn.metrics import classification_report, accuracy_score
import torch.nn.functional as F
from torch.nn import MSELoss, L1Loss, CosineSimilarity

import torch

import numpy as np
import pandas as pd

from .dataset import FewShotDataset

METHODS = ['zeroshot', 'palm']


def get_args():
    parser = argparse.ArgumentParser(description='PALM: Prompt-based Few-Shot Learning for Audio Language Models')
    parser.add_argument('--method_name', type=str, default='', help='Model Name (default: None)', required=True)
    parser.add_argument('--save_model', help='Save the trained model (default: False)', action='store_true')
    parser.add_argument('--save_model_path', type=str, default=None, help='Path to save the trained model (default: None)')
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to the pre-trained model (learnable context) weights (default: None)')
    parser.add_argument('--load_model_abs_path', type=str, default=None, help='Absolute path to the pre-trained model (learnable context) weights (default: None)')
    parser.add_argument('--dataset_root', type=str, default='', help='Path to the dataset root directory (default: None)', required=True)
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--start_epoch', type=int, default=0, help='Starting epoch (default: 0)')
    parser.add_argument('--freq_test_model', type=int, default=10, help='Frequency of testing the model (default: 10)')
    parser.add_argument('--test_model_last_epoch_only', help='Test the model only at the last epoch (default: False)', action='store_true')
    parser.add_argument('--spec_aug', help='Apply Spectrogram Augmentation (default: False)', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size (default: 16)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=0, help='Random Seed (default: 0)')
    parser.add_argument('--eval_only', help='Evaluate the model only (default: False)', action='store_true')
    parser.add_argument('--exp_name', type=str, default='', help='experiment name', required=True)
    parser.add_argument('--do_logging', help='Disable Logging (default: False)', action='store_true')
    parser.add_argument('--prompt_prefix', type=str, default='The is a recording of ', help='Prompt Prefix (default: The is a recording of )')
    
    # COOP/COCOOP and PALM Arguments
    parser.add_argument('--n_ctx', type=int, default=16, help='Number of context tokens (default: 16)')
    parser.add_argument('--ctx_dim', type=int, default=512, help='Dimension of the context vector (default: 512)')

    # Few-Shot Learning Arguments
    parser.add_argument('--num_shots', type=int, default=16, help='Number of shots (default: 16)')
    parser.add_argument('--resample', type=bool, default=True, help='Resample samples if needed (default: True)')
    parser.add_argument('--repeat', type=bool, default=False, help='Repeat samples if needed (default: False)')

    # Backdoor Attack Arguments
    parser.add_argument('--attack_name', type=str, default='trojanwave', help='Attack Name (default: trojanwave)', choices=['trojanwave', 'flowmur', 'nbad', 'nba', 'noattack'])
    parser.add_argument('--rho', type=float, default=0.1, help='Perturbation Budget')
    parser.add_argument('--eps', type=float, default=0.2, help='Perturbation Budget for Audio')
    parser.add_argument('--lambda_clean', type=float, default=1.0, help='Clean Loss Weight')
    parser.add_argument('--lambda_adv', type=float, default=1.0, help='Adversarial Loss Weight')
    parser.add_argument('--not_use_audio_noise', help='Use Audio Noise (default: False)', action='store_true')
    parser.add_argument('--not_use_spec_noise', help='Use Spectral Noise (default: False)', action='store_true')
    parser.add_argument('--poison_rate', type=float, default=5.0, help='Poison Rate (default: 5.0 %)')
    parser.add_argument('--target_label', type=int, default=0, help='Target Label for Backdoor Attack (default: 0)')
    parser.add_argument('--mask_region', type=str, default='all', help='Mask Type (default: low)', choices=['low', 'mid', 'high', 'all'])
    parser.add_argument('--blend_rate', type=float, default=1.0, help="Weight for Audio Trigger (default: 1)")
    parser.add_argument('--noise_duration', type=str, default='half', help='Duration of Audio Noise w.r.t to Audio Waveform (default: half)', choices=['quarter', 'half', 'three-quarters', 'full'])
    parser.add_argument('--trigger', type=str, default='distant-whistle', help='Trigger to Perturb the Audio')
        
    args = parser.parse_args()

    # Sanity check on Arguments
    if not os.path.exists(args.dataset_root):
        raise ValueError(f"\n\nDirectory '{args.dataset_root}' does not exist. Specify the correct path to the dataset.\n\n")
    if args.save_model and not os.path.exists(args.save_model_path):
        raise ValueError(f"\n\nDirectory '{args.save_model_path}' does not exist. Create or specify the correct the directory to save the trained model.\n\n")
    if args.eval_only: 
        load_model_path = get_load_model_path(args)
        if not os.path.exists(load_model_path): raise ValueError(f"\n\nEvaluation Mode: Model file '{load_model_path}' does not exist. Specify the correct path to the model file.\n\n")
    if args.method_name == 'zeroshot': args.eval_only = True

    if not args.test_model_last_epoch_only and (args.n_epochs) % args.freq_test_model != 0:
        raise UserWarning(f"Number of epochs '{args.n_epochs}' is not divisible by the frequency of testing the model '{args.freq_test_model}'.\nConsider changing the frequency of testing the model or the number of epochs.\nOtherwise, the model will not be tested at the end of training and results will not be saved.")


    args.use_spec_noise = not args.not_use_spec_noise
    args.use_audio_noise = not args.not_use_audio_noise
        
    return args





def get_model(args, pengi, methods):
    print(f"Using Method: '{args.method_name.upper()}'\n")

    if args.method_name == 'zeroshot':
        model = methods.ZeroShot(args, pengi)
    elif args.method_name == 'coop':
        model = methods.COOP(args, pengi)
    elif args.method_name == 'cocoop':
        model = methods.COCOOP(args, pengi)
    elif args.method_name == 'palm':
        model = methods.PALM(args, pengi)
    else:
        raise ValueError(f"Method '{args.method_name}' is not supported. Choose from: [{', '.join(METHODS)}]")
    
    return model


def get_dataloaders(args):
    num_workers = 2
    # setting args.num_shots to -1 (set bach to args.num_shots)
    train_dataset = FewShotDataset(args.dataset_root, 'train' , num_shots=args.num_shots, repeat=args.repeat , process_audio_fn=args.process_audio_fn, resample=args.resample, poison_rate=args.poison_rate, target_label=args.target_label)
    test_dataset_clean  = FewShotDataset(args.dataset_root, 'test', clean=True, num_shots=-1, repeat=args.repeat , process_audio_fn=args.process_audio_fn, resample=args.resample, poison_rate=args.poison_rate, target_label=args.target_label)
    test_dataset_poisoned  = FewShotDataset(args.dataset_root, 'test', clean=False, num_shots=-1, repeat=args.repeat , process_audio_fn=args.process_audio_fn, resample=args.resample, poison_rate=args.poison_rate, target_label=args.target_label)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader_clean = torch.utils.data.DataLoader(test_dataset_clean, batch_size=128, shuffle=False, num_workers=num_workers)
    test_dataloader_poisoned = torch.utils.data.DataLoader(test_dataset_poisoned, batch_size=128, shuffle=False, num_workers=num_workers)

    test_dataloader  = [test_dataloader_clean, test_dataloader_poisoned]


    args.classnames = train_dataloader.dataset.classnames
    assert train_dataloader.dataset.classnames == test_dataloader[0].dataset.classnames, "Classnames in train and test (clean) datasets are different."
    assert train_dataloader.dataset.classnames == test_dataloader[1].dataset.classnames, "Classnames in train and test (poisoned) datasets are different."

    return train_dataloader, test_dataloader



def save_model(args, model, save_model_path):
        print(f"Saving Context Weights for Method: '{args.method_name.upper()}'\n")
        if args.method_name in ['coop', 'cocoop', 'palm']:
            checkpoint = {'prompt_learner': model.prompt_learner.state_dict()}
            checkpoint['pengi_bn0_buffer'] = {'running_mean': model.audio_encoder.base.htsat.bn0.running_mean.clone(), 
                                              'running_var': model.audio_encoder.base.htsat.bn0.running_var.clone(),
                                              'num_batches_tracked': model.audio_encoder.base.htsat.bn0.num_batches_tracked.clone()}
                          
            if args.attack_name in ['trojanwave', 'flowmur']: 
                checkpoint['attack'] = {'audio_noise': model.attack.audio_noise, 'spec_noise':model.attack.spec_noise}
            
            torch.save(checkpoint, save_model_path)
        else:
            raise ValueError(f"Model '{args.method_name}' is not supported. Choose from: [{', '.join(METHODS)}]")


def load_model(args, model, backdoor=False):
        load_model_path = get_load_model_path(args)
        checkpoint = torch.load(load_model_path)
        # model.prompt_learner.load_state_dict(checkpoint['prompt_learner'])
        model.audio_encoder.base.htsat.bn0.running_mean.copy_(checkpoint['pengi_bn0_buffer']['running_mean'])
        model.audio_encoder.base.htsat.bn0.running_var.copy_(checkpoint['pengi_bn0_buffer']['running_var'])
        model.audio_encoder.base.htsat.bn0.num_batches_tracked.copy_(checkpoint['pengi_bn0_buffer']['num_batches_tracked'])
        
        if backdoor:
            model.prompt_learner.ctx_backdoor = checkpoint['prompt_learner']['ctx'].clone().detach()
            if args.attack_name in ['trojanwave', 'flowmur']:
                    model.attack.audio_noise_backdoor = checkpoint['attack']['audio_noise'].clone().detach()
                    model.attack.spec_noise_backdoor = checkpoint['attack']['spec_noise'].clone().detach()
                    model.attack.audio_noise = checkpoint['attack']['audio_noise'].clone().detach()
                    model.attack.spec_noise = checkpoint['attack']['spec_noise'].clone().detach()         
        else:
            if args.attack_name in ['trojanwave', 'flowmur']:
                model.attack.audio_noise = checkpoint['attack']['audio_noise']
                model.attack.spec_noise = checkpoint['attack']['spec_noise']            
        # raise NotImplementedError("\n\nLoading model is not implemented yet.\n\n")






def get_save_model_path(args):
        save_model_path = os.path.join(args.save_model_path, args.method_name, args.attack_name)
        os.makedirs(save_model_path, exist_ok=True)
        # if not os.path.exists(save_model_path): os.mkdir(save_model_path)
        save_model_path = os.path.join(save_model_path, f"{args.exp_name}-SEED_{args.seed}-TARGET_{args.target_label}.pth")
        return save_model_path


def get_load_model_path(args):
        if args.load_model_abs_path is not None:
            load_model_path = args.load_model_abs_path
        else:
            load_model_path = os.path.join(args.load_model_path, args.method_name, args.attack_name, f"{args.exp_name}-SEED_{args.seed}-TARGET_{args.target_label}.pth")
        
        if not os.path.exists(load_model_path): 
            raise ValueError(f"Model file '{load_model_path}' does not exist. Specify the correct path to the model file.")
        
        return load_model_path



def compute_loss(logits, labels, ctx, ctx_backdoor, backdoor_tags, lambda_clean=1.0, lambda_adv=1.0, multi_label=False):
    
    clean_exists = any(~backdoor_tags)
    backdoor_exists = any(backdoor_tags)
    
    # # compute cross-entropy loss for multi-label classification
    # if multi_label:

    #     if clean_exists:
    #         loss_ce_clean = F.binary_cross_entropy_with_logits(logits[~backdoor_tags], labels[~backdoor_tags])
            
    #     if backdoor_exists: 
    #         loss_ce_adv = F.binary_cross_entropy_with_logits(logits[backdoor_tags], labels[backdoor_tags]) 

    #     if clean_exists and backdoor_exists:
    #         loss_ce = lambda_clean*loss_ce_clean + lambda_adv*loss_ce_adv 
    #     elif clean_exists and not backdoor_exists:
    #         loss_ce = 1.0*loss_ce_clean
    #     elif not clean_exists and backdoor_exists:
    #         loss_ce = 1.0*loss_ce_adv
    #     else:
    #         raise ValueError("No clean or backdoor sample found. Check the backdoor_tags assignment in Dataset class.")
        
    #     loss = lambda_ce*loss_ce

    # # compute cross-entropy loss for single-label classification
    # else:

    if clean_exists:
        loss_ce_clean = F.cross_entropy(logits[~backdoor_tags], labels[~backdoor_tags])
        
    if backdoor_exists: 
        loss_ce_adv = F.cross_entropy(logits[backdoor_tags], labels[backdoor_tags]) 

    if clean_exists and backdoor_exists:
        loss_ce = lambda_clean*loss_ce_clean + lambda_adv*loss_ce_adv
    elif clean_exists and not backdoor_exists:
        loss_ce = 1.0*loss_ce_clean
    elif not clean_exists and backdoor_exists:
        loss_ce = 1.0*loss_ce_adv
    else:
        raise ValueError("No clean or backdoor sample found. Check the backdoor_tags assignment in Dataset class.")
    
 
    loss_cr = MSELoss()(ctx, ctx_backdoor) # context repulsion loss
    # loss_cr = L1Loss()(ctx, ctx_backdoor)
    # loss_cr = (1-CosineSimilarity(dim=1, eps=1e-6)(ctx, ctx_backdoor)).mean()
    
    
    # print(f"CE Loss: {loss_ce.item():.4f}, CR Loss: {loss_cr.item():.4f}")

    loss = 1.0*loss_ce - 1.0*loss_cr
    # loss = 1.0*loss_ce

    
    return loss
        


def print_total_time(now_start, now_end):
	print(f'\nEnd Time & Date = {now_end.strftime("%I:%M %p")} , {now_end.strftime("%d_%b_%Y")}\n')
	duration_in_s = (now_end - now_start).total_seconds()
	days  = divmod(duration_in_s, 86400)   # Get days (without [0]!)
	hours = divmod(days[1], 3600)          # Use remainder of days to calc hours
	minutes = divmod(hours[1], 60)         # Use remainder of hours to calc minutes
	seconds = divmod(minutes[1], 1)        # Use remainder of minutes to calc seconds
	print(f"Total Time => {int(days[0])} Days : {int(hours[0])} Hours : {int(minutes[0])} Minutes : {int(seconds[0])} Seconds\n\n")



def print_dataset_info(train_dataloader, test_dataloader):
	n_classes = train_dataloader.dataset.n_classes
	num_batches_train = len(train_dataloader)
	num_batches_test = len(test_dataloader)

	print("\n########################\nDataset Information\n########################\n")
	print("Length of the Train Dataset: ", len(train_dataloader.dataset))
	print("Length of the Test Dataset: ", len(test_dataloader.dataset))
	print("Train Batch Size: ", train_dataloader.batch_size)
	print("Test Batch Size: ", test_dataloader.batch_size)
	print("Number of Batches in Train Dataloader: ", num_batches_train)
	print("Number of Batches in Test Dataloader: ", num_batches_test)
	print("Number of Classes: ", n_classes)
     

def get_scores(actual_labels, predicted_labels, classnames):
    cls_report = classification_report(actual_labels, predicted_labels, labels=np.arange(0, len(classnames)), target_names=classnames, zero_division=1, output_dict=True)
    if 'accuracy' not in cls_report:
        cls_report['accuracy'] = accuracy_score(actual_labels, predicted_labels)
    accuracy = cls_report['accuracy']
    f1_score = cls_report['macro avg']['f1-score']
    precision = cls_report['macro avg']['precision']
    recall = cls_report['macro avg']['recall']
    return accuracy, f1_score, precision, recall


def print_scores(accuracy, f1_score, precion, recall, avg_loss):
    print(f"{'Accuracy':<15} = {accuracy:0.4f}")
    print(f"{'F1-Score':<15} = {f1_score:0.4f}")
    print(f"{'Precision':<15} = {precion:0.4f}")
    print(f"{'Recall':<15} = {recall:0.4f}")
    print(f"{'Average Loss':<15} = {avg_loss:0.4f}\n\n")


def save_scores(scores):
    seed=scores['seed']
    epoch=scores['epoch']
    accuracy_clean, f1_score_clean, precision_clean, recall_clean, avg_loss_clean = scores['clean']['accuracy'], scores['clean']['f1_score'], scores['clean']['precision'], scores['clean']['recall'], scores['clean']['loss']
    accuracy_backdoor, f1_score_backdoor, precision_backdoor, recall_backdoor, avg_loss_backdoor = scores['backdoor']['accuracy'], scores['backdoor']['f1_score'], scores['backdoor']['precision'], scores['backdoor']['recall'], scores['backdoor']['loss']
    json_file_path = scores['json_file_path']
    
    if not os.path.exists(json_file_path):
        # create the file if it doesn't exist
        with open(json_file_path, "w") as file:
            file.write("{}")
        
    # load existing results
    with open(json_file_path, "r") as file:
        scores_json = json.load(file)

    scores_json[f"seed_{seed}"] = {"accuracy_clean": f"{accuracy_clean:0.4f}", "f1_score_clean": f"{f1_score_clean:0.4f}", "precision_clean": f"{precision_clean:0.4f}", "recall_clean": f"{recall_clean:0.4f}", "avg_loss_clean": f"{avg_loss_clean:0.4f}", "accuracy_backdoor": f"{accuracy_backdoor:0.4f}", "f1_score_backdoor": f"{f1_score_backdoor:0.4f}", "precision_backdoor": f"{precision_backdoor:0.4f}", "recall_backdoor": f"{recall_backdoor:0.4f}", "avg_loss_backdoor": f"{avg_loss_backdoor:0.4f}", "epoch": epoch}

    for metric in scores_json[f"seed_{seed}"].keys():
        if metric != 'epoch': scores_json[f"seed_{seed}"][metric] = float(scores_json[f"seed_{seed}"][metric])


    # save updated results
    with open(json_file_path, "w") as file:
        json.dump(scores_json, file, indent=2) 



# Decorator to measure the time taken by a function
def timeit(func):
    import time
    def wrapper(*args, **kwargs):
        
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        
        duration_in_s = end - start
        days  = divmod(duration_in_s, 86400)   # Get days (without [0]!)
        hours = divmod(days[1], 3600)          # Use remainder of days to calc hours
        minutes = divmod(hours[1], 60)         # Use remainder of hours to calc minutes
        seconds = divmod(minutes[1], 1)        # Use remainder of minutes to calc seconds


        date_now = datetime.datetime.now(pytz.timezone('Asia/Dubai'))
        print(f'\n\nTime & Date = {date_now.strftime("%I:%M %p")} , {date_now.strftime("%d_%b_%Y")}  GST')
        print(f"\nTotal Time => {int(days[0])} Hours : {int(minutes[0])} Minutes : {int(seconds[0])} Seconds\n\n")
        return result
        
    return wrapper


############################################################################################################
# Logging Functions
############################################################################################################


# Define a Tee class to duplicate output to both stdout and a log file
class Tee:
    def __init__(self, *files):
        self.files = files
 
    def write(self, text):
        for file in self.files:
            file.write(text)
            file.flush()
 
    def flush(self):
        for file in self.files:
            file.flush()

# Define a function to redirect stdout and stderr to a log file
def redirect_output_to_log(log_file):
    # Open the log file in append mode
    log = open(log_file, 'a')
 
    # Duplicate stdout and stderr
    sys.stdout = Tee(sys.stdout, log)
    sys.stderr = Tee(sys.stderr, log)

    return log


# Define a function to setup logging
def setup_logging(args):
    log_dir = os.path.join('logs', args.method_name, args.attack_name) # log file dir
    args.log_dir = log_dir
    
    # adding poison rate
    if args.do_logging:
        if not os.path.exists(log_dir): os.makedirs(log_dir)
        file_name = get_file_name(args)
        log_file_path = os.path.join(log_dir, f"{file_name}.log")
        if os.path.exists(log_file_path): os.remove(log_file_path)
        json_file_path = os.path.join(log_dir, f"{file_name}.json")
        args.json_file_path = json_file_path
        print(f"\nLogging to '{log_file_path}'\n")
        log_file = redirect_output_to_log(log_file_path) # redirect terminal output to log file
    else:
        log_file =None

    return log_file



def get_file_name(args):
    if args.attack_name == "noattack":
        file_name = f"{args.exp_name}-SEED_{args.seed}"
    elif args.attack_name == "nba":
        file_name = f"{args.exp_name}-SEED_{args.seed}-POISON_{int(args.poison_rate)}-TARGET_{args.target_label}-TRIGGER_{args.trigger}"
    elif args.attack_name == "nbad":
        file_name = f"{args.exp_name}-SEED_{args.seed}-POISON_{int(args.poison_rate)}-TARGET_{args.target_label}-TRIGGER_{args.trigger}"
    elif args.attack_name == "flowmur":
        file_name = f"{args.exp_name}-SEED_{args.seed}-EPS_{args.eps}-POISON_{int(args.poison_rate)}-TARGET_{args.target_label}"
    elif args.attack_name == "trojanwave":
        file_name = f"{args.exp_name}-SEED_{args.seed}-EPS_{args.eps}-RHO_{args.rho}-POISON_{int(args.poison_rate)}-TARGET_{args.target_label}"
    else:
        raise ValueError(f"Attack name '{args.attack_name}' not recognized. Choose from ['noattack', 'nba', 'nbad', 'flowmur', 'trojanwave']")
    return file_name


############################################################################################################
############################################################################################################




def get_masks(time_dim, freq_dim, device, low_freq_ratio=0.30, mid_freq_ratio=0.30):
    """
    Get masks for low, medium, and high-frequency regions of a spectrogram.

    Args:
        time_dim (int): Size of the time dimension (t) of the spectrogram.
        freq_dim (int): Size of the frequency dimension (f) of the spectrogram.
        device (torch.device): Device to allocate the masks on.
        low_freq_ratio (float): Fraction of the frequency range to consider as "low-frequency".
        mid_freq_ratio (float): Fraction of the frequency range to consider as "mid-frequency".

    Returns:
        dict: Masks for low, medium, and high-frequency regions, each of shape [t, f].
    """
    assert low_freq_ratio + mid_freq_ratio <= 1.0, "Low and mid-frequency ratios must sum to <= 1.0"

    # Frequency ranges
    low_end = int(low_freq_ratio * freq_dim)
    mid_end = int((low_freq_ratio + mid_freq_ratio) * freq_dim)

    # Initialize masks for the entire spectrogram shape
    low_freq_mask = torch.zeros((time_dim, freq_dim), dtype=torch.bool, device=device)
    mid_freq_mask = torch.zeros((time_dim, freq_dim), dtype=torch.bool, device=device)
    high_freq_mask = torch.zeros((time_dim, freq_dim), dtype=torch.bool, device=device)

    # Apply masks to respective frequency ranges
    low_freq_mask[:, :low_end] = True                           # Low frequencies from bottom upwards
    mid_freq_mask[:, low_end:mid_end] = True                    # Mid frequencies above low frequencies
    high_freq_mask[:, mid_end:] = True                          # High frequencies from the top

    # Return masks as a dictionary
    masks = {
        'low': low_freq_mask,   # [t, f]
        'mid': mid_freq_mask,   # [t, f]
        'high': high_freq_mask  # [t, f]
    }

    return masks


# def get_2d_masks(side_lenght, device):
#     matrix_ones = np.ones( [side_lenght, side_lenght], dtype=np.float32)
#     low_mask = np.flip((np.triu(matrix_ones, side_lenght//2)),1).copy()
#     high_mask = np.fliplr(np.rot90(low_mask)).copy()
#     middle_mask = np.where((low_mask+high_mask)==1, 0 , 1).astype(dtype=np.float32)
#     return torch.from_numpy(low_mask).to(device), torch.from_numpy(middle_mask).to(device), torch.from_numpy(high_mask).to(device)
