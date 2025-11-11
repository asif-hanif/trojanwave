import os
import torch
import numpy as np
from tqdm import tqdm

from .utils import get_scores, print_scores, save_scores, timeit, save_model, get_save_model_path, compute_loss


def run_epoch(model, dataloader, optimizer, device, args=None):
    model.train()

    losses = []
    actual_labels = []
    predicted_labels = []

    for i, (audio, label, backdoor_tags) in enumerate(dataloader):

        audio = audio.to(device).squeeze(1)
        label = label.to(device)

        logits = model(audio, backdoor_tags)

        loss = compute_loss(logits, label, model.prompt_learner.ctx,  model.prompt_learner.ctx_backdoor, backdoor_tags, lambda_clean=args.lambda_clean, lambda_adv=args.lambda_adv)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        backdoor_exists = any(backdoor_tags)
        if (args.attack_name=='trojanwave' or args.attack_name=='flowmur') and backdoor_exists: 
            model.attack.update_noise()


        losses.append(loss.item())

        actual_labels.extend(label.cpu().numpy())
        predicted_labels.extend(logits.argmax(axis=1).cpu().numpy())

    avg_loss = sum(losses) / len(losses)

    return avg_loss, actual_labels, predicted_labels


@timeit
def run_evaluation(model, dataloader, device, args=None):
    model.eval()
    model.prompt_learner.ctx_backdoor = model.prompt_learner.ctx_backdoor.to(device)

    losses = []
    actual_labels = []
    predicted_labels = []
    
    print("\n\nEvaluating the model ...")
    with torch.no_grad():
        for i, (audio, label, backdoor_tags) in enumerate(dataloader):
        # for i, (audio, label) in tqdm(enumerate(dataloader), total=len(dataloader)):
            print(f"Batch {i+1}/{len(dataloader)}")

            audio = audio.to(device).squeeze(1)
            label = label.to(device)
            
            logits = model(audio, backdoor_tags)
            loss = compute_loss(logits, label, model.prompt_learner.ctx,  model.prompt_learner.ctx_backdoor, backdoor_tags, lambda_clean=args.lambda_clean, lambda_adv=args.lambda_adv)

            losses.append(loss.item())

            actual_labels.extend(label.cpu().numpy())
            predicted_labels.extend(logits.argmax(axis=1).cpu().numpy())

    avg_loss = sum(losses) / len(losses)

    return avg_loss, actual_labels, predicted_labels


@timeit
def run_training(model, train_dataloader, test_dataloader, optimizer, device, epochs=50, args=None):

    pre_attack_setup(args, train_dataloader)
    
    for epoch in tqdm(range(epochs), total=epochs):

        train_loss, actual_labels, predicted_labels = run_epoch(model, train_dataloader, optimizer, device, args=args)

        if (epoch+1)%5 == 0:
            accuracy, f1_score, precision, recall =  get_scores(actual_labels, predicted_labels, args.classnames)
            print(f"\n\n-------------------------------\nTrain Evaluation (Epoch {epoch + 1}/{epochs})\n-------------------------------\n")
            print_scores(accuracy, f1_score, precision, recall, train_loss) 


        # test model at the end of training
        if args.test_model_last_epoch_only and epoch == epochs - 1:
            scores = get_clean_backdoor_scores(test_dataloader, model, device, args)
            scores.update({'seed': args.seed, 'epoch': epoch, 'json_file_path': args.json_file_path})

            if args.do_logging: 
                print("\n\nFinal Evaluation\nSaving Results ...")
                save_scores(scores)
                print(f"Results Saved @ {scores['json_file_path']}\n\n")

        # test model at regular intervals
        elif not args.test_model_last_epoch_only and (epoch + 1) % args.freq_test_model == 0:
            scores = get_clean_backdoor_scores(test_dataloader, model, device, args)
            scores.update({'seed': args.seed, 'epoch': epoch, 'json_file_path': args.json_file_path})

            if args.do_logging and epoch == epochs - 1:
                print("\n\nFinal Evaluation\nSaving Results ...")
                save_scores(scores)
                print(f"Results Saved @ {scores['json_file_path']}\n\n")

    
    if args.save_model:
        save_model_path = get_save_model_path(args)
        save_model(args, model, save_model_path)
        print(f"Model Saved @ {save_model_path}")



def get_clean_backdoor_scores(test_dataloader, model, device, args):

    test_dataloader_clean, test_dataloader_poisoned = test_dataloader

    # Clean Dataset
    print("\n\nEvaluating the model on clean test dataset ...")

    test_loss_clean, actual_labels, predicted_labels = run_evaluation(model, test_dataloader_clean, device, args=args)
    accuracy_clean, f1_score_clean, precision_clean, recall_clean =  get_scores(actual_labels, predicted_labels, args.classnames)
    print(f"\n\n-------------------------------\nTest Evaluation (Clean)\n-------------------------------\n")
    print_scores(accuracy_clean, f1_score_clean, precision_clean, recall_clean, test_loss_clean)

    # Poisoned Dataset
    print("\n\nEvaluating the model on poisoned test dataset ...")
    test_loss_backdoor, actual_labels, predicted_labels = run_evaluation(model, test_dataloader_poisoned, device, args=args)
    accuracy_backdoor, f1_score_backdoor, precision_backdoor, recall_backdoor =  get_scores(actual_labels, predicted_labels, args.classnames)
    print(f"\n\n-------------------------------\nTest Evaluation (Backdoor)\n-------------------------------\n")
    print_scores(accuracy_backdoor, f1_score_backdoor, precision_backdoor, recall_backdoor, test_loss_backdoor)

    scores = {}
    scores['clean'] = {'accuracy': accuracy_clean, 'f1_score': f1_score_clean, 'precision': precision_clean, 'recall': recall_clean, 'loss': test_loss_clean}
    scores['backdoor'] = {'accuracy': accuracy_backdoor, 'f1_score': f1_score_backdoor, 'precision': precision_backdoor, 'recall': recall_backdoor, 'loss': test_loss_backdoor}
    return scores


def pre_attack_setup(args, train_dataloader):
    if args.attack_name == 'noattack':
        print("\n\n###############################################")
        print("Clean Model Setup")
        print(f"Model is being trained with 'Clean Setup' (i.e. no attack is being applied)")
        print(f"Poison Rate is being set to 0%")
        print("###############################################\n\n")
        train_dataloader.dataset.clear_backdoor_tags()


