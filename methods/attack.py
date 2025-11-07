import torch 
import os
import pandas as pd
import numpy as np
 

from pengi import pengi

def BackdoorAttack(args):
    
    ATTACKS = ["trojanwave", "flowmur", "nba", "nbad", "noattack"]

    if args.attack_name == "trojanwave":
        return TrojanWave(args)
    elif args.attack_name == "flowmur":
        return FlowMur(args)
    elif args.attack_name == "nba":
        return NBA(args)
    elif args.attack_name == "nbad":
        return NBAD(args)
    elif args.attack_name == "noattack":
        return NoAttack(args)
    else:
        raise ValueError(f"Model '{args.attack_name}' is not supported. Choose from: [{', '.join(ATTACKS)}]")


class TrojanWave():
    def __init__(self, args): 
        self.rho = args.rho 
        self.eps = args.eps
        self.blend_rate = args.blend_rate
        self.use_spec_noise = args.use_spec_noise
        self.use_audio_noise = args.use_audio_noise
        self.device = args.device
        noise_duration = {'quarter':1/4, 'half':2/4, 'three-quarters':3/4, 'full':1}[args.noise_duration]
        self.audio_len = 308700
        self.noise_len = int(self.audio_len * noise_duration)

        self.init_noise(args)

        print("\n\n###############################################")
        print("Attack: TrojanWave")
        print(f"Use Spec Noise: {self.use_spec_noise}")
        print(f"Use Audio Noise: {self.use_audio_noise}")
        print(f"Rho (Perturbation Budget for Spec_Noise): {self.rho}")
        print(f"Eps (Perturbation Budget for Audio_Noise): {self.eps}")
        print(f"Blend Rate: {self.blend_rate}")
        print(f"Duration of Audio Noise: {self.noise_len}")
        print("###############################################\n\n")



    def init_noise(self, args):

        if self.use_audio_noise:
            self.audio_noise = torch.normal(mean=0, std=0.3, size=(1, self.noise_len)).to(self.device).clamp(-self.eps, self.eps).detach()
            self.audio_noise.detach_()
            self.audio_noise.requires_grad = True
        else: 
            self.audio_noise = torch.zeros(1, self.noise_len).to(self.device).detach()
            self.audio_noise.requires_grad = False

        if self.use_spec_noise:
            self.spec_noise = torch.from_numpy(np.random.uniform(low=1-self.rho, high=1+self.rho, size=(1, 965, 64)).astype('float32') ).to(self.device).detach()
            self.spec_noise.detach_()
            self.spec_noise.requires_grad = True
        else: 
            self.spec_noise = torch.ones(1, 965, 64).to(self.device).detach()
            self.spec_noise.requires_grad = False



    def add_trigger(self, audio, backdoor_tag):
        noise_len = self.noise_len
        tau = np.random.randint(0, self.audio_len - noise_len) if self.audio_len > noise_len else 0
        audio[backdoor_tag, tau:tau+noise_len] = audio[backdoor_tag, tau:tau+noise_len] + self.blend_rate * self.audio_noise 
        audio = torch.clamp(audio, min=-1.0, max=1.0)
        return audio
    

    def update_noise(self):

        # update audio noise 
        if self.use_audio_noise:
            self.audio_noise = self.audio_noise - 0.01*self.audio_noise.grad.sign()
            self.audio_noise = torch.clamp(self.audio_noise, min=-self.eps, max=self.eps)
            self.audio_noise.detach_()
            self.audio_noise.requires_grad = True

        # update spec noise
        if self.use_spec_noise:
            self.spec_noise = self.spec_noise - 0.01*self.spec_noise.grad.sign()
            self.spec_noise = self.spec_noise.clamp(1-self.rho, 1+self.rho).detach()
            self.spec_noise.detach_()
            self.spec_noise.requires_grad = True



############################################################################################################
############################################################################################################



class FlowMur():
    def __init__(self, args):
        self.eps = args.eps
        self.blend_rate = args.blend_rate
        self.device = args.device
        self.audio_len = 308700
        noise_duration = {'quarter':1/4, 'half':2/4, 'three-quarters':3/4, 'full':1}[args.noise_duration]
        self.noise_len = int(self.audio_len * noise_duration)

        self.init_noise()

        print("\n\n###############################################")
        print("Attack: FlowMur")
        print(f"Eps (Perturbation Budget for Audio_Noise): {self.eps}")
        print(f"Blend Rate: {self.blend_rate}")
        print(f"Duration of Audio Noise: {self.noise_len}")
        print("###############################################\n\n")

    def init_noise(self):
        self.audio_noise = torch.normal(mean=0, std=0.3, size=(1, self.noise_len)).to(self.device).clamp(-self.eps, self.eps).detach()
        self.audio_noise.detach_()
        self.audio_noise.requires_grad = True

        self.spec_noise = torch.ones(1, 965, 64).to(self.device).detach()
        self.spec_noise.requires_grad = False

    def add_trigger(self, audio, backdoor_tag):
        tau = np.random.randint(0, self.audio_len - self.noise_len) if self.audio_len > self.noise_len else 0
        audio[backdoor_tag, tau:tau+self.noise_len] = audio[backdoor_tag, tau:tau+self.noise_len] + self.blend_rate * self.audio_noise
        audio = torch.clamp(audio, min=-1.0, max=1.0)
        return audio

    def update_noise(self):
        self.audio_noise = self.audio_noise - 0.01*self.audio_noise.grad.sign()
        self.audio_noise = torch.clamp(self.audio_noise, min=-self.eps, max=self.eps)
        self.audio_noise.detach_()
        self.audio_noise.requires_grad = True


############################################################################################################
############################################################################################################



class NBA():
    def __init__(self, args):
        args.lambda_clean = 1.0
        args.lambda_adv = 1.0
        self.audio_len = 308700
        self.load_audio_into_tensor = pengi.load_audio_into_tensor
        self.trigger = args.trigger 
        self.trigger_path = os.path.join('media', 'audio_triggers', 'audios', self.trigger) + '.wav'
        metadata = pd.read_csv(os.path.join('media', 'audio_triggers', 'metadata.csv'))
        self.duration = int(metadata[metadata['file'] == self.trigger + '.wav']['duration'])
        self.blend_rate = args.blend_rate
        self.resample = args.resample
        self.device = args.device
        
        self.init_noise()

        print("\n\n###############################################")
        print("Attack: NBA")
        print(f"Trigger: {self.trigger}")
        print(f"Blend Rate: {self.blend_rate}")
        print(f"Duration of Audio Noise: {self.noise_len}")
        print("###############################################\n\n")

    def init_noise(self):
        self.audio_noise = self._preprocess_trigger(self.trigger_path, self.resample, self.duration)
        self.noise_len = self.audio_noise.shape[1]
        if self.audio_len < self.noise_len:
            self.audio_noise = self.audio_noise[:, self.audio_len]

        self.spec_noise = torch.ones(1, 965, 64).to(self.device).detach()
        self.spec_noise.requires_grad = False

    def add_trigger(self, audio, backdoor_tag):
        noise_len = self.noise_len
        tau = 0
        audio[backdoor_tag, tau:tau+noise_len] = audio[backdoor_tag, tau:tau+noise_len] + self.blend_rate * self.audio_noise #* torch.sigmoid(self.audio_noise_weight)
        audio = torch.clamp(audio, min=-1.0, max=1.0)
        return audio

    def _preprocess_trigger(self, audio_file, resample, duration=1):
        r"""Load audio file and return raw audio"""
        audio_tensor = self.load_audio_into_tensor(audio_file, duration, resample)
        audio_tensor = audio_tensor.reshape(1, -1).to(self.device)
        return audio_tensor


############################################################################################################
############################################################################################################


class NBAD():
    def __init__(self, args):
        args.lambda_clean = 1.0
        args.lambda_adv = 1.0
        self.audio_len = 308700
        self.load_audio_into_tensor = pengi.load_audio_into_tensor
        self.trigger = args.trigger 
        self.trigger_path = os.path.join('media', 'audio_triggers','audios', self.trigger) + '.wav'
        metadata = pd.read_csv(os.path.join('media', 'audio_triggers', 'metadata.csv'))
        self.duration = int(metadata[metadata['file'] == self.trigger + '.wav']['duration'])
        self.blend_rate = args.blend_rate
        self.resample = args.resample
        self.device = args.device

        self.init_noise(args)

        print("\n\n###############################################")
        print("Attack: NBAD")
        print(f"Trigger: {self.trigger}")
        print(f"Blend Rate: {self.blend_rate}")
        print(f"Duration of Audio Noise: {self.noise_len}")
        print("###############################################\n\n")

    def init_noise(self, args):
        self.audio_noise = self._preprocess_trigger(self.trigger_path, self.resample, self.duration)
        self.noise_len = self.audio_noise.shape[1]
        if self.audio_len < self.noise_len:
            self.audio_noise = self.audio_noise[:, self.audio_len]

        self.spec_noise = torch.ones(1, 965, 64).to(self.device).detach()
        self.spec_noise.requires_grad = False

    def add_trigger(self, audio, backdoor_tag):
        noise_len = self.noise_len
        tau = np.random.randint(0, self.audio_len - noise_len) if self.audio_len > noise_len else 0
        audio[backdoor_tag, tau:tau+noise_len] = audio[backdoor_tag, tau:tau+noise_len] + self.blend_rate * self.audio_noise 
        audio = torch.clamp(audio, min=-1.0, max=1.0)
        return audio

    def _preprocess_trigger(self, audio_file, resample, duration=1):
        r"""Load audio file and return raw audio"""
        audio_tensor = self.load_audio_into_tensor(audio_file, duration, resample)
        audio_tensor = audio_tensor.reshape(1, -1).to(self.device)
        return audio_tensor
    

############################################################################################################
############################################################################################################


class NoAttack():     

    def __init__(self, args): 
        self.device = args.device
        self.audio_len = 308700
        self.noise_len = self.audio_len

        self.init_noise(args)

        print("\n\n###############################################")
        print("Clean Model Setup")
        print(f"Model is being trained with 'Clean Setup' (i.e. no attack is being applied)")
        print("###############################################\n\n")


    def init_noise(self, args):
        self.spec_noise = torch.ones(1, 965, 64).to(self.device).detach()
        self.spec_noise.requires_grad = False

        self.audio_noise = torch.zeros(1, self.noise_len).to(self.device).detach()
        self.audio_noise.requires_grad = False


    def add_trigger(self, audio, backdoor_tag):
        return audio
    

############################################################################################################
############################################################################################################








