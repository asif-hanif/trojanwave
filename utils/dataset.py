import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import ast


class FewShotDataset(Dataset):
    def __init__(self, root, split=None, clean=True, num_shots=-1, repeat=False, process_audio_fn=None, resample=True, poison_rate=5.0, target_label=0): 
        """
        Args:
            root (str): path to the dataset.
            num_shots (int): number of shots per class.
            repeat (bool): repeat samples if needed (default: False).
            process_audio_fn (function): function to process audio samples.
            resample (bool): resample audio samples (default: True).
            poison_rate (float): percentage of backdoor samples.
            target_label (int): target label for backdoor samples.
        """

        assert split is not None, "'split' cannot be None. Choose from ['train', 'test']"
        
        self.root = root
        self.split = split
        self.num_shots = num_shots
        self.repeat = repeat
        self.resample = resample
        self.poison_rate = poison_rate
        self.target_label = target_label

        df = pd.read_csv(os.path.join(root, f"{split}.csv"))
        
        self.classnames = df['classname'].unique().tolist()
        self.classnames.sort()
        self.label2classname = {i: classname for i, classname in enumerate(self.classnames)}
        self.classname2label = {classname: i for i, classname in enumerate(self.classnames)}
        
        self.data = self.generate_fewshot_dataset(df, num_shots=num_shots, repeat=repeat)

        if split == 'test' and not clean:
            self.data = self.drop_target_label(self.data)


        self.process_audio_fn = process_audio_fn


        self.num_samples = len(self.data)

        if split=='train':
            self.backdoor_tags = torch.from_numpy(np.random.choice([0, 1], size=(self.num_samples,), p=[1-(self.poison_rate/100),self.poison_rate/100]))
        elif split=='test' and clean:
            self.backdoor_tags = torch.zeros(self.num_samples)
        else:
            self.backdoor_tags = torch.ones(self.num_samples)


        self.backdoor_tags = self.backdoor_tags.bool()

        print("\n\n################## Dataset Information ##################")
        if num_shots>0: print("FewShot Dataset")
        print(f"{'Root':<25} : {root}")
        print(f"{'Split':<25} : {split}")
        print(f"{'Number of Classes':<25} : {len(self.classnames)}")
        print(f"{'Number of Shots':<25} : {num_shots}")
        print(f"{'Total Number of Samples':<25} : {len(self.data)}")
        print(f"{'Classnames':<25} : {self.classnames}")
        print(f"{'Label to Classname':<25} : {self.label2classname}")
        print(f"{'Classname to Label':<25} : {self.classname2label}")
        print(f"{'Poison Rate':<25} : {poison_rate} %")
        print(f"{'Number of Poisoned Samples:':<25} : {torch.sum(self.backdoor_tags)}")
        print(f"{'Target Label':<25} : {target_label} --> {self.label2classname[target_label]}")
        print("########################################################\n\n")

    def generate_fewshot_dataset(self, df, num_shots=-1, repeat=False):
        """
        Generate a few-shot dataset.
        Args:
            df (pd.DataFrame): dataframe containing the dataset.
            num_shots (int): number of shots per class.
            repeat (bool): repeat samples if needed.
        """

        if num_shots == -1:
            return df

        print(f"Creating a {num_shots}-shot dataset ...")
        df_fewshot = pd.DataFrame(columns=df.columns)

        for classname in self.classnames:

            df_class = df[df['classname'] == classname]

            if len(df_class) >= num_shots:
                df_fewshot = pd.concat([df_fewshot, df_class.sample(num_shots)])
            else:
                if repeat:
                    df_fewshot = pd.concat([df_fewshot, df_class.sample(num_shots, replace=True)])
                else:
                    df_fewshot = pd.concat([df_fewshot,df_class])


        df_fewshot = df_fewshot.sample(frac=1).reset_index(drop=True)

        return df_fewshot
    

    def drop_target_label(self, df):
        return df[df['classname']!=self.label2classname[self.target_label]]


    def clear_backdoor_tags(self):
        self.backdoor_tags = torch.zeros(self.num_samples).bool()
    
    def set_backdoor_tags(self):
        self.backdoor_tags = torch.ones(self.num_samples).bool()

    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_path = os.path.join(self.root, row['path'])
        audio = self.process_audio_fn([audio_path], self.resample) # [1,n_samples]
        label = self.classname2label[row['classname']]
        if self.backdoor_tags[idx]: label = self.target_label
        return audio, label, self.backdoor_tags[idx]
    


############################################################################################################
############################################################################################################



# class FewShotMultiLabelDataset(Dataset):    
#     def __init__(self, root, split=None, num_shots=-1, repeat=False, process_audio_fn=None, resample=True, poison_rate=5.0, target_label=0): 
#         """
#         Args:
#             root (str): path to the dataset.
#             num_shots (int): number of shots per class.
#             repeat (bool): repeat samples if needed (default: False).
#             process_audio_fn (function): function to process audio samples.
#             resample (bool): resample audio samples (default: True).
#             poison_rate (float): percentage of backdoor samples.
#             target_label (int): target label for backdoor samples.
#         """

#         assert split is not None, "'split' cannot be None. Choose from ['train', 'test']"
        
#         self.root = root
#         self.split = split
#         self.num_shots = num_shots
#         self.repeat = repeat
#         self.resample = resample

#         df = pd.read_csv(os.path.join(root, f"{split}.csv"))
        
#         self.classnames = df['classname'].apply(ast.literal_eval).explode().unique().tolist()
#         self.classnames.sort()
#         self.label2classname = {i: classname for i, classname in enumerate(self.classnames)}
#         self.classname2label = {classname: i for i, classname in enumerate(self.classnames)}
        
#         # one-hot encoded labels
#         self.labels = df['classname'].apply(ast.literal_eval).apply(lambda x: [1 if classname in x else 0 for classname in self.classnames]).tolist()
        
#         # append df with one-hot encoded labels with column names as classnames
#         df = pd.concat([df, pd.DataFrame(self.labels, columns=self.classnames)], axis=1)


#         self.data = self.generate_fewshot_dataset(df, num_shots=num_shots, repeat=repeat)

        
#         self.process_audio_fn = process_audio_fn

#         # Backdoor Attack
#         self.poison_rate = poison_rate
#         self.target_label = target_label

#         self.num_samples = len(self.data)

#         if split=='train':
#             self.backdoor_tags = torch.from_numpy(np.random.choice([0, 1], size=(self.num_samples,), p=[1-(self.poison_rate/100),self.poison_rate/100]))
#         else:
#             self.backdoor_tags = torch.zeros(self.num_samples)

#         self.backdoor_tags = self.backdoor_tags.bool()

#         print("\n\n################## Dataset Information ##################")
#         if num_shots>0: print("FewShot Dataset")
#         print(f"{'Root':<25} : {root}")
#         print(f"{'Split':<25} : {split}")
#         print(f"{'Number of Classes':<25} : {len(self.classnames)}")
#         print(f"{'Number of Shots':<25} : {num_shots}")
#         print(f"{'Total Number of Samples':<25} : {len(self.data)}")
#         print(f"{'Classnames':<25} : {self.classnames}")
#         print(f"{'Label to Classname':<25} : {self.label2classname}")
#         print(f"{'Classname to Label':<25} : {self.classname2label}")
#         print(f"{'Poison Rate':<25} : {poison_rate} %")
#         print(f"{'Number of Poisoned Samples:':<25} : {torch.sum(self.backdoor_tags)}")
#         print(f"{'Target Label':<25} : {target_label} --> {self.label2classname[target_label]}")
#         print("########################################################\n\n")

#     def onehot_to_classnames(self, label):
#         return [self.classnames[i] for i in range(len(label)) if label[i]==1]
    
#     def classnames_to_onehot(self, classnames):
#         return [1 if classname in classnames else 0 for classname in self.classnames]
    

#     def generate_fewshot_dataset(self, df, num_shots=-1, repeat=False):
#         """
#         Generate a few-shot dataset.
#         Args:
#             df (pd.DataFrame): dataframe containing the dataset.
#             num_shots (int): number of shots per class.
#             repeat (bool): repeat samples if needed.
#         """

#         if num_shots == -1:
#             return df

#         # print(f"Creating a {num_shots}-shot dataset ...")
#         # df_fewshot = pd.DataFrame(columns=df.columns)

#         # initialize an empty list to collect few-shot samples
#         few_shot_samples = []

#         # initialize an empty set to keep track of already selected indices
#         selected_indices = set()

#         for classname in self.classnames:

#             # filter out samples that have already been selected in previous iterations
#             available_samples = df[(df[classname] == 1) & (~df.index.isin(selected_indices))]
            
#             # sample the desired number of rows for the current label
#             class_samples = available_samples.sample(n=min(num_shots, len(available_samples)))
            
#             # add the sampled indices to the set of selected indices
#             selected_indices.update(class_samples.index)
            
#             # Append the samples to the few-shot list
#             few_shot_samples.append(class_samples)

#         # Concatenate all selected samples into a single DataFrame
#         df_fewshot = pd.concat(few_shot_samples).reset_index(drop=True)

#         return df_fewshot

#     def clear_backdoor_tags(self):
#         self.backdoor_tags = torch.zeros(self.num_samples).bool()
    
#     def set_backdoor_tags(self):
#         self.backdoor_tags = torch.ones(self.num_samples).bool()

#     def __len__(self):
#         return len(self.data)
    
    
#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]
#         audio_path = os.path.join(self.root, row['path'])
#         audio = self.process_audio_fn([audio_path], self.resample) # [1,n_samples]
        
#         if self.backdoor_tags[idx]:
#             label = [0]*len(self.classnames)
#             label[self.target_label] = 1 
#         else:
#             label = self.classnames_to_onehot(row['classname'])

#         return audio, label, self.backdoor_tags[idx]
    