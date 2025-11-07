import os
import json
import argparse
import numpy as np



def check_seed_existence(results, seed): 
    return f'seed_{seed}' in results.keys()

# Function to load JSON data from a file
def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)
    
def get_json_file_name(args):
    if args.attack_name == "noattack":
        json_file_name = f"{args.dataset}-SEED_{args.seed}.json"
    elif args.attack_name == "trojanwave":
        json_file_name = f"{args.dataset}-SEED_{args.seed}-EPS_{args.eps}-RHO_{args.rho}-POISON_{int(args.poison_rate)}-TARGET_{args.target_label}.json"
    elif args.attack_name == "flowmur":
        json_file_name = f"{args.dataset}-SEED_{args.seed}-EPS_{args.eps}-POISON_{int(args.poison_rate)}-TARGET_{args.target_label}.json"
    elif args.attack_name == "nbad":
        json_file_name = f"{args.dataset}-SEED_{args.seed}-POISON_{int(args.poison_rate)}-TARGET_{args.target_label}-TRIGGER_{args.trigger}.json"
    elif args.attack_name == "nba":
        json_file_name = f"{args.dataset}-SEED_{args.seed}-POISON_{int(args.poison_rate)}-TARGET_{args.target_label}-TRIGGER_{args.trigger}.json"
    else:
        raise ValueError(f"Attack name '{args.attack_name}' not recognized. Choose from ['noattack', 'nba', 'nbad', 'flowmur', 'trojanwave']")
    return json_file_name


# Function to get results for all seeds of a dataset and method   
def load_dataset_results(dataset, results_folder, args):
    args.dataset = dataset
    results = {}
    for target_label in range(DATASETS[dataset]):
        args.target_label = target_label
        json_file_name = get_json_file_name(args)
        json_file_path = f"{os.path.join(results_folder, json_file_name)}"

        if os.path.exists(json_file_path):
            result = load_json(json_file_path)
            seed_exist = check_seed_existence(result, args.seed)
            if not seed_exist: raise ValueError(f"Seed {args.seed} not found in {json_file_path} file. Get results for seed={args.seed} first in '{json_file_path}'.")
        else:
            raise ValueError(f"File {json_file_path} does not exist. Get results for Dataset='{dataset}'.") 
        results[f'target_{target_label}'] = result[f'seed_{args.seed}']

    return results


def get_results(results_folder, args):
    results = {}
    for dataset in DATASETS:
        results[dataset] = load_dataset_results(dataset, results_folder, args)
    return results

def average_results_taret_labels(dataset, results, args):
    ca_list = []
    ba_list = []
    
    for target_label in range(DATASETS[dataset]):
        ca_list.append(results[dataset][f'target_{target_label}'][f'seed_{args.seed}']['accuracy_clean'])
        ba_list.append(results[dataset][f'target_{target_label}'][f'seed_{args.seed}']['accuracy_backdoor'])
    
    return np.mean(ca_list), np.mean(ba_list)


if __name__ == "__main__":
    
    # Datasets 
    DATASETS = {
                'Beijing-Opera':4,
                'CREMA-D':6,
                'ESC50-Actions':10,
                'ESC50':50,
                'GT-Music-Genre':10,
                'NS-Instruments':10,
                'RAVDESS':8,
                'SESA':4,
                'TUT2017':15,
                'UrbanSound8K':10,
                'VocalSound':6,
    }

    parser = argparse.ArgumentParser(description='Print Backdoor Attack Results')
    parser.add_argument('--method', type=str, default='', help='Method Name (default: palm)', required=True, choices=['coop', 'cocoop', 'palm'] )

    parser.add_argument('--seed', type=int, default=0, help='Seed value (default: 0)')
    parser.add_argument('--rho', type=float, default=0.1, help='Rho value (default: 0.1)')
    parser.add_argument('--eps', type=float, default=0.2, help='Epsilon value (default: 0.2)')
    parser.add_argument('--poison_rate', type=int, default=5, help='Poison rate value (default: 5 %)')
    parser.add_argument('--target_label', type=int, default=0, help='Target label value (default: 0)')
    parser.add_argument('--trigger', type=str, default='distant-whistle', help='Trigger to Perturb the Audio')


    args = parser.parse_args()

    method = args.method
    seed = args.seed

    attacks = ['nba', 'nbad', 'flowmur', 'trojanwave']
    # attacks = ['flowmur', 'trojanwave']


    for attack in attacks:
        args.attack_name = attack

        results_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), method, attack)
        
        results = get_results(results_folder, args) # get results for all datasets and seed in the form of a dictionary where keys are the names of datasets
        
        accuracy_dict = {}

        for dataset in DATASETS:
            clean_acc, backdoor_acc = average_results_taret_labels(dataset, results, args)
            accuracy_dict[dataset] = [clean_acc, backdoor_acc]

        with open(os.path.join(results_folder,'accuracy_target_avg.json'), 'w') as f:
            json.dump(accuracy_dict, f, indent=2)
        
        print(f"Results saved in {os.path.join(results_folder, 'accuracy_target_avg.json')} file.")

    print("\n\nResults saved successfully.\n\n")