# TrojanWave: Exploiting Prompt Learning for Stealthy Backdoor Attacks on Large Audio-Language Models (EMNLP'25)

> [**TrojanWave: Exploiting Prompt Learning for Stealthy Backdoor Attacks on Large Audio-Language Models**](https://aclanthology.org/2025.emnlp-main.940/)<br><br>
> [Asif Hanif](https://scholar.google.com/citations?hl=en&user=6SO2wqUAAAAJ), [Maha Tufail Agro](https://scholar.google.com/citations?user=FXJzma8AAAAJ), [Fahad Shamshad](https://scholar.google.com/citations?user=d7QL4wkAAAAJ), and
[Karthik Nandakumar](https://scholar.google.com/citations?user=2qx0RnEAAAAJ)


[![page](https://img.shields.io/badge/Project-Page-F9D371)](https://asif-hanif.github.io/trojanwave/)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://aclanthology.org/2025.emnlp-main.940/)




<hr />

| ![main figure](/media/trojanwave.png)|
|:--| 
| **TrojanWave**<p align="justify">This attack learns two triggers (temporal and spectral) to embed a backdoor into the audio-language model (ALM) during prompt learning. The ALM’s weights remain frozen, and only the learnable prompts are manipulated. At inference time, the ALM performs normally on clean inputs (performance on par with the backdoor-free setup) but predicts the adversary’s target label $y^{\prime}$ when input containing trigger is presented.</p> |

</br>

| ![main figure](/media/attack_pipeline.png)|
|:--| 
| **TrojanWave Attack Pipeline**<p align="justify">An adversary embeds a backdoor into the learned prompts during few-shot training and publishes the infected prompts online. An unsuspecting user who adopts these prompts for their model unknowingly inherits the backdoor, resulting in normal performance on clean inputs but adversary-desired targeted misclassification when triggered inputs are encountered.</p> |

</br>
<hr />
</br>

> **Abstract** <p align="justify"><i>
Prompt learning has emerged as an efficient alternative to full fine-tuning for adapting large audio-language models (ALMs) to downstream tasks. While this paradigm enables scalable deployment via Prompt-as-a-Service frameworks,  it also introduces a critical yet underexplored security risk of backdoor attacks. In this work, we present *TrojanWave*, the first backdoor attack tailored to the prompt-learning setting in frozen ALMs. Unlike prior audio backdoor methods that require training from scratch on full datasets, *TrojanWave* injects backdoors solely through learnable prompts, making it highly scalable and effective in few-shot settings. *TrojanWave* injects imperceptible audio triggers in both time and spectral domains to effectively induce targeted misclassification during inference. 
To mitigate this threat, we further propose *TrojanWave-Defense*, a lightweight prompt purification method that neutralizes malicious prompts without hampering the clean performance. Extensive experiments across 11 diverse audio classification benchmarks demonstrate the robustness and practicality of both the attack and defense. 
<br><br>
</i></p>

> <b>TLDR:</b> The paper presents TrojanWave, a novel backdoor attack on audio-language models that exploits prompt learning instead of model retraining.
It exposes the security risks of malicious prompts that inject imperceptible audio triggers causing hidden misclassifications.A lightweight defense, TrojanWave-Defense, is proposed to purify infected prompts while preserving normal model performance.

<br><br>


> <p align="justify"> <b>Goal:</b> <i> The paper aims to introduce and analyze a new type of backdoor attack—called TrojanWave—targeting large audio-language models (ALMs) that use prompt learning. Its main objective is to show that such attacks can be executed solely through learnable prompts, without modifying model parameters, making them highly stealthy and scalable. It also proposes a defense method, TrojanWave-Defense, to purify infected prompts and restore model safety without degrading normal performance.
</i></p>

> <p align="justify"> <b>Motivation:</b> <i> With the rise of prompt learning and “Prompt-as-a-Service” frameworks, users increasingly rely on third-party prompts to adapt models efficiently. However, this creates a serious security risk where adversaries can distribute malicious prompts that appear normal but contain hidden backdoors triggered by imperceptible sounds. Recognizing that such prompt-based attacks were largely unexplored, the paper seeks to expose this vulnerability and highlight the urgent need for protection in real-world ALM deployments.
</i></p>

> <p align="justify"> <b>Main Idea:</b> <i> TrojanWave introduces a stealthy attack that embeds imperceptible audio triggers—crafted in both time and spectral domains—into learnable prompts, which then cause targeted misclassification when triggered inputs are encountered. Unlike prior attacks that retrain models, it keeps the backbone model frozen, making the method lightweight and efficient. To counteract this, the authors propose TrojanWave-Defense, a prompt purification strategy that removes the correlation between malicious prompts and triggers while maintaining clean-task accuracy.
</i></p>


</br>
</br>

## Updates :rocket:
- **Aug 20, 2025** : Accepted in [EMNLP (Main) 2025](https://2025.emnlp.org/) &nbsp;&nbsp; :confetti_ball: :tada:
- **Nov 05, 2025** : Released code for TrojanWave-Attack
- **Nov 05, 2025** : Released instructions for preparing datasets  
- **Nov 10, 2025** : Released code for TrojanWave-Defense 

</br>
</br>

## Table of Contents
- [Installation](#installation)
- [Model](#model)
- [Datasets](#datasets)
- [Code Structure](#code-structure)
- [Run Experiments](#run-experiments)
- [Results](#results)
- [Citation](#citation)
- [Contact](#contact)
- [Acknowledgement](#acknowledgement)

</br>
</br>

<a name="installation"/>

## Installation :gear:
1. Create a conda environment. If already created (in attack setup), skip this step. 
```shell
conda create --name trojanwave python=3.8
conda activate trojanwave
```
2. Clone trojanwave-defense branch
```shell
git clone --branch trojanwave-defense --single-branch https://github.com/asif-hanif/trojanwave.git trojanwave-defense
```
3. Install PyTorch and other dependencies. If already done (in attack setup), skip this step.
```shell
cd trojanwave-defense
pip install -r requirements.txt
```

</br>
<a name="model"/>
    
## Model :white_square_button:
We have shown the results on TrojanWave and other baselines (NBA, NBA-D, FlowMur) using [PENGI](https://github.com/microsoft/Pengi) model. 

Download the pre-trained PENGI model using the link provided below and place the checkpoint file at path [`pengi/configs`](/pengi/configs) (after clonning the repo). 


| Model | Link | Size |
|:-- |:-- | :-- |
| PENGI | [Download](https://zenodo.org/records/8387083/files/base.pth) | 2.2 GB | 

<br>

PENGI checkpoint can also be downloaded with following command:
```bash
wget https://zenodo.org/records/8387083/files/base.pth
```

</br>

<a name="datasets"/>
    
## Datasets :page_with_curl:

We have performed experiments on 11 audio classification datasets.  Instructions for downloading/processing datasets used by our method have been provided in the [DATASETS.md](DATASETS.md). 

| Dataset | Type | Classes | Size | Link |
|:-- |:-- |:--: |--: |:-- |
| [Beijing-Opera](https://compmusic.upf.edu/bo-perc-dataset) | Instrument Classification | 4 | 69 MB | [Instructions](DATASETS.md#beijing-opera) |
| [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) | Emotion Recognition | 6 | 606 MB | [Instructions](DATASETS.md#crema-d) |
| [ESC50](https://github.com/karolpiczak/ESC-50) | Sound Event Classification | 50 | 881 MB | [Instructions](DATASETS.md#esc50) |
| [ESC50-Actions](https://github.com/karolpiczak/ESC-50) | Sound Event Classification | 10 | 881 MB | [Instructions](DATASETS.md#esc50-actions) |
| [GT-Music-Genre](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) | Music Analysis | 10 | 1.3 GB | [Instructions](DATASETS.md#gt-music-genre) |
| [NS-Instruments](https://magenta.tensorflow.org/datasets/nsynth) | Instrument Classification | 10 | 18.5 GB | [Instructions](DATASETS.md#ns-instruments) |
| [RAVDESS](https://zenodo.org/records/1188976#.YFZuJ0j7SL8) | Emotion Recognition | 8 | 1.1 GB | [Instructions](DATASETS.md#ravdess) |
| [SESA](https://zenodo.org/records/3519845) | Surveillance Sound Classification | 4 | 70 MB | [Instructions](DATASETS.md#sesa) |
| [TUT2017](https://zenodo.org/records/400515) | Acoustic Scene Classification | 15 | 12.3 GB | [Instructions](DATASETS.md#tut2017) |
| [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) | Sound Event Classification | 10 | 6.8 GB | [Instructions](DATASETS.md#urbansound8k) |
| [VocalSound](https://github.com/YuanGongND/vocalsound) | Vocal Sound Classification | 6 | 8.2 GB | [Instructions](DATASETS.md#vocalsound) |

</br>
</br>

All datasets should be placed in a directory named `Audio-Datasets` and the path of this directory should be specified in the variable `DATASET_ROOT` in the shell [`scripts`](/scripts/). The directory structure should be as follows:
```
Audio-Datasets/
    ├── Beijing-Opera/
    ├── CREMA-D/
    ├── ESC50/ 
    ├── ESC50-Actions/
    ├── GT-Music-Genre/
    ├── NS-Instruments/
    ├── RAVDESS/
    ├── SESA/
    ├── TUT2017/
    ├── UrbanSound8K/
    ├── VocalSound/
 ```


</br>

<a name="code-structure"/>

## Code Structure :snowflake:
There are three main folders in this repo: `pengi`, `methods`, `utils`. Code in [`pengi`](/pengi) folder is taken from [PENGI](https://github.com/microsoft/Pengi) repo for model instantiation. Implementation of baselines (`nba`, `nbad`, `flowmur`) and our method `trojanwave` is in [`methods`](/methods) folder. Class definitions of audio and text encoder of PENGI model can be found in [`methods/encoders.py`](/methods/encoders.py) file. Training and dataset related code is in [`utils`](/utils) folder.

</br>

<a name="run-experiments"/>

## Run Defense Experiments :zap:

We have performed all experiments on `NVIDIA A100-SXM4-40GB` GPU. Shell scripts to run experiments can be found in [`scripts`](/scripts/) folder. 

```shell
## General Command Structure
bash  <SHELL_SCRIPT>  <METHOD_NAME> <ATTACK_NAME>
```

Following methods (including `trojanwave`) are supported in this repository:

`noattack` `nba` `nbad` `flowmur`

Examples to run `trojanwave` method on different audio classifiction datasets have been provided below:

```shell
bash scripts/beijing_opera.sh palm trojanwave
bash scripts/crema_d.sh palm trojanwave
bash scripts/esc50_actions.sh palm trojanwave
bash scripts/esc50.sh palm trojanwave
bash scripts/gt_music_genre.sh palm trojanwave
bash scripts/ns_instruments.sh palm trojanwave
bash scripts/ravdess.sh palm trojanwave
bash scripts/sesa.sh palm trojanwave
bash scripts/tut.sh palm trojanwave
bash scripts/urban_sound.sh palm trojanwave
bash scripts/vocal_sound.sh palm trojanwave
```

Results are saved in `json` format in [`logs`](/logs) directory. To process results, run the following command (after running all experiments):

```bash
cd logs
bash results.sh
```

<details>
<summary>Sample Output</summary>

![main figure](/media/defense-results.png)

</details>

</br>
</br>

**Note** To simplify evaluation, we convert multi-fold datasets into a single train–test split rather than performing cross-validation.

</br>

<a name="results"/>

## Results :microscope:

<div class="content has-text-justified"><p>


![main figure](/media/attack_and_defense_results.png)

</br>
</br>

<a name="citation"/>

## Citation :star:
If you find our work, this repository, or pretrained models useful, please consider giving a star :star: and citation.
```bibtex
@inproceedings{hanif2025trojanwave,
  title={TrojanWave: Exploiting Prompt Learning for Stealthy Backdoor Attacks on Large Audio-Language Models},
  author={Hanif, Asif and Agro, Maha Tufail and Shamshad, Fahad and Nandakumar, Karthik},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={18628--18644},
  year={2025}
}
```

</br>

<a name="contact"/>

## Contact :mailbox:
Should you have any questions, please create an issue on this repository or contact us at **asif.hanif@mbzuai.ac.ae**

</br>

<a name="acknowledgement"/>

## Acknowledgement :pray:
We used [PENGI](https://github.com/microsoft/Pengi) for model instantiation and borrowed a part of code from NBA, NBA-D and FlowMur to implement baselines. We thank the respective authors for releasing the code.

<hr />

