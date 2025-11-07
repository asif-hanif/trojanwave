import os
import json
import argparse
import numpy as np
from tabulate import tabulate
from collections import defaultdict



def format_value(value, significant_digits=4):
    formatted_nvalue = f"{value*100:.6f}"  # convert to percentage
    # len of formatted number without dot
    len_value = len(formatted_nvalue.replace('.', ''))
    if len_value > significant_digits: formatted_number = formatted_nvalue[:significant_digits+1]
    return formatted_number

def format_delta(value, significant_digits=3):
    if value >= 0: tri = " \\uptri{"
    else: tri = " \\downtri{"
    formatted_nvalue = f"{abs(value)*100:.6f}"  # convert to percentage
    # len of formatted number without dot
    len_value = len(formatted_nvalue.replace('.', ''))
    if len_value > significant_digits: formatted_number = formatted_nvalue[:significant_digits+1]
    formatted_number = tri + formatted_number + '}'
    return formatted_number



if __name__ == '__main__':

    DATASETS = [
            'Beijing-Opera',
            'CREMA-D',
            'ESC50-Actions',
            'ESC50',
            'GT-Music-Genre',
            'NS-Instruments',
            'RAVDESS',
            'SESA',
            'TUT2017',
            'UrbanSound8K',
            'VocalSound',
        ]

    parser = argparse.ArgumentParser(description='Print Backdoor Attack Results')
    parser.add_argument('--method', type=str, default='', help='Method Name (default: palm)', required=True, choices=['coop', 'cocoop', 'palm'] )
    
    args = parser.parse_args()

    method = args.method
    attacks = ['noattack', 'nba', 'nbad', 'flowmur', 'trojanwave']
    # attacks = ['noattack', 'flowmur', 'trojanwave']

    accuracy_dict_all = defaultdict(list)
    accuracy_list_all = []

    for attack in attacks:   
        results_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), method, attack)
        accuracy_dict = json.load(open(os.path.join(results_folder, 'accuracy.json')))

        for dataset in DATASETS: accuracy_dict_all[dataset].extend( accuracy_dict[dataset][:1] if attack == 'noattack' else accuracy_dict[dataset] )


    # average accuracy across all datasets
    for dataset in DATASETS: accuracy_list_all.append([accuracy for accuracy in accuracy_dict_all[dataset]])

    avg_accuracy_all = list(np.array(accuracy_list_all).mean(axis=0))

    accuracy_dict_all['AVERAGE'] = avg_accuracy_all

    accuracy_all = np.array(accuracy_list_all+[avg_accuracy_all])


    noattack_acc = accuracy_all[:,0]

    delta = (accuracy_all - noattack_acc[:,None]) 


    # generate latex table
    string_acc = ''
    for i, dataset in enumerate(DATASETS+['AVERAGE']):
        acc_list = []
        for j, accuracy in enumerate(accuracy_dict_all[dataset]):
            if j % 2 == 0: acc_list.append(format_value(accuracy))
            else: acc_list.append(format_value(accuracy) + format_delta(delta[i][j]))

        if dataset != 'AVERAGE': string_acc = string_acc +  f'{dataset} & ' + ' & '.join(acc_list) + ' \\\\\n'
        else: string_acc = string_acc +  f'\midrule\n{dataset} & ' + ' & '.join(acc_list) + ' \\\\\n'

    
    # string_acc = string_acc +  f'\midrule\nAVERAGE & ' + ' & '.join([format_value(accuracy) for accuracy in avg_accuracy_all]) + ' \\\\\n'
    
    # for dataset in enumerate(DATASETS): string_acc = string_acc +  f'{dataset} & ' + ' & '.join([format_value(accuracy) for accuracy in accuracy_dict_all[dataset]]) + ' \\\\\n'
    # string_acc = string_acc +  f'\midrule\nAVERAGE & ' + ' & '.join([format_value(accuracy) for accuracy in avg_accuracy_all]) + ' \\\\\n'


    top_row = f"DATASETS ↓ & NoAttack-CA & "
    for attack in attacks[1:]:  top_row = top_row + f"{attack.upper()}-CA & {attack.upper()}-BA &"
    top_row = top_row[:-1] + ' \\\\'


    print(f"\n\n########## ACCURACY (LaTeX Table)   METHOD={args.method.upper()} ##########")
    results_acc = top_row+"\n"+string_acc
    print(results_acc)



    # print table in terminal
    # breakpoint()
    # results_acc=results_acc.replace("\\downtri{", "(▼")
    # results_acc=results_acc.replace("\\uptri{", "(▲")
    # results_acc=results_acc.replace("}", ")")
    results_acc=results_acc.replace("\\downtri{", "(-")
    results_acc=results_acc.replace("\\uptri{", "(+")
    results_acc=results_acc.replace("}", ")")

    table_acc = []
    for i, row in enumerate(results_acc.split("\n")):
        row_list = row.split("&")
        col_list = []
        for j, col in enumerate(row_list):
            if col.endswith("\\\\"): col = col[:-3]
            col = col.strip()
            col_list.append(col)
        if '\\midrule' in col_list or '' in col_list: continue
        table_acc.append(col_list)
    print(f"\n\n########## METHOD={args.method.upper()} ##########")
    print(tabulate(table_acc, tablefmt="simple"))

    print("\n\n")


