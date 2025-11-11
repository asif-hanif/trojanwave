import os
import random
import numpy as np
import datetime
import pytz
from tqdm import tqdm
from pprint import pprint

import torch
import torch.nn as nn

import methods
from pengi import pengi


from utils import trainer
from utils.utils import print_total_time, get_args, get_dataloaders, get_model, setup_logging, get_scores, print_scores, save_scores, load_model

# to solve  the issue of : the current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = '6'


def main(args):

    print(f"\n\n{'Method:':<10}{args.method_name.upper()}")
    print(f"{'Attack:':<10}{args.attack_name}")
    print(f"{'Dataset:':<10}{args.dataset_root.split('/')[-1]}")
    print(f"{'Seed:':<10}{args.seed}\n\n")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    args.process_audio_fn = pengi.preprocess_audio

    # to ensure reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    
    train_dataloader, test_dataloader = get_dataloaders(args)

    model = get_model(args, pengi, methods)
    model.to(device)

    
    print("\nArguments:\n")
    for arg in vars(args): print(f"{arg:<25}: {getattr(args, arg)}")
    print("\n\n")


    if args.eval_only:
        if args.method_name != "zeroshot": load_model(args, model)
        scores = trainer.get_clean_backdoor_scores(test_dataloader, model, device, args)
    else:
        if args.method_name != "zeroshot": load_model(args, model, backdoor=True) # load the pre-trained backdoor-infected model
        optimizer = torch.optim.SGD(model.prompt_learner.parameters(), lr=args.lr, momentum=0.9)
        trainer.run_training(model, train_dataloader, test_dataloader, optimizer, device, epochs=args.n_epochs, args=args)

        

if __name__ == "__main__":

    args = get_args()
    log_file = setup_logging(args)

    print("\n\n########################################################################")
    print("TrojanWave: Backdoor Attacks on Audio Language Models during Prompt Learning")
    print("########################################################################\n\n")
    date_now = datetime.datetime.now(pytz.timezone('Asia/Dubai'))
    print(f'Time & Date = {date_now.strftime("%I:%M %p")} , {date_now.strftime("%d_%b_%Y")}  GST\n')

    main(args)

