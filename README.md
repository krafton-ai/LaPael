# Latent Paraphrasing: Perturbation on Layers Improves Knowledge Injection in Language Models

by [Minki Kang](https://nardien.github.io/)<sup>1,2</sup>, [Sung Ju Hwang](http://www.sungjuhwang.com/)<sup>2</sup>, [Gibbeum Lee](https://scholar.google.com/citations?user=I4X0TyYAAAAJ&hl=en)<sup>1</sup>, and [Jaewoong Cho](https://sites.google.com/view/jaewoongcho)<sup>1</sup>.

<sup>1</sup> [KRAFTON AI](https://www.krafton.ai/en/research/publications/), <sup>2</sup> KAIST


📚 This repository contains the official implementation of the paper [**Latent Paraphrasing: Perturbation on Layers Improves Knowledge Injection in Language Models**](https://arxiv.org/abs/2411.00686), presented at **NeurIPS 2024**.

⚠️ **Important Note**:  
This repository is intended for **research and prototype development only**. It is not suitable for direct production use. The code is not a product of **KRAFTON Inc.** and is provided solely for research purposes.

This project builds on version `0.0.1` of [llama-recipes](https://github.com/meta-llama/llama-recipes). The repository includes:

### ✨ Key Features:
- **Train latent paraphrasers** using the SQuAD training set.
    - Cache embeddings of paraphrases for training latent paraphrasers.
    - Includes pre-trained weights for the latent paraphrasers used in our experiments.
- **Fine-tune LLMs** using trained latent paraphrasers.

### 🛠️ Upcoming Updates:
- Code to **generate paraphrases** for training latent paraphrasers.
- Support for additional datasets used in experiments (currently, only SQuAD is included as a reference).
- Detailed guide on how to adapt and use this code with other LLMs than Vicuna.

---

## 📝 Abstract

As Large Language Models (LLMs) are increasingly deployed in specialized domains with continuously evolving knowledge, the need for timely and precise knowledge injection has become essential. Fine-tuning with paraphrased data is a common approach to enhance knowledge injection, yet it faces two significant challenges: high computational costs due to repetitive external model usage and limited sample diversity. To this end, we introduce **LaPael**, a latent-level paraphrasing method that applies input-dependent noise to early LLM layers. This approach enables diverse and semantically consistent augmentations directly within the model. Furthermore, it eliminates the recurring costs of paraphrase generation for each knowledge update. Our extensive experiments on question-answering benchmarks demonstrate that LaPael improves knowledge injection over standard fine-tuning and existing noise-based approaches. Additionally, combining LaPael with data-level paraphrasing further enhances performance.

---

## 🛠️ Installation

### ✅ Requirements:
- **Python** version `>=3.8`

### Installation Steps:
```bash
pip install -U pip setuptools
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 -e .
```

---

## 📂 Dataset Setup

The required datasets for experimentation are organized as follows:
- SQuAD
    - $D_{train}$: `Data_Preprocessing/oracle_textbook/squad_train_short/processed_sentences.json`
        - Paraphrases: `Knowledge_Generator/openai_outputs/squad_train_short/paraphrase_medium_suffix_c256_gen10_gpt-3.5-turbo.json`
    - $D_{K}$: `Data_Preprocessing/oracle_textbook/squad_test_short/processed_sentences.json`
    - $D_{QA}$: `Evaluator/ContextQA/squad_test`

We plan to add additional datasets to support the reproduction of experiments. If you want to test our methods on new datasets, you can refer to the format of each provided dataset and construct your own dataset following the same structure. Detailed descriptions of the dataset format, including examples, are included in the repository to guide you in this process.

---

## 🔐 Pre-trained Checkpoints for LaPael

We provide pre-trained checkpoints of latent paraphrasers used in our experiments.

**Download Links**: [Google Drive](https://drive.google.com/file/d/1gtvPxO6p-w-yAWLv0yHo1tL0P6wW6o3v/view?usp=drive_link)

---

## 📖 Usage Guide

### Training Latent Paraphrasers

1. Cache Embeddings: Generate paraphrase embeddings by running:
```bash
python Analysis/get_lm_embeddings.py --lm vicuna --augtype medium --domain squad_train_short
```

2. Train Latent Paraphrasers: Use the training script provided:
```bash
sh scripts/train_lapael.sh configs/perturb_base squad
```

---

### Fine-tuning LLMs for Knowledge Injection

Use trained latent paraphrasers to fine-tune LLMs:
```bash
sh scripts/finetune_perturbation.sh SOURCE TARGET LAPAEL_PATH
```

- **SOURCE**: Dataset used to train LaPael (e.g., `squad`).
- **TARGET**: Dataset for LLM fine-tuning (e.g., `squad`).
- **LAPAEL_PATH**: Path to the trained LaPael model (e.g., `1206-vicuna-epoch=10-config=perturb_base-short-seed42`).

The fine-tuning script also evaluates the fine-tuned model on the QA dataset to measure its effectiveness.

---

## 🙌 Acknowledgments

This project builds upon open-source contributions from the [llama-recipes](https://github.com/meta-llama/llama-recipes) repository.  
We extend our gratitude to the research community for providing valuable datasets and tools.

For questions or discussions, feel free to open an issue or submit a pull request.