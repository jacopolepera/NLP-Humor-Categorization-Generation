# NLP Project: Understanding, Evaluating, and Generating Humor with LLMs

This repository contains a comprehensive pipeline for categorizing humor styles using Large Language Models (LLMs), training an AlBERT regression model on those categories, and finally improving a generative model (Mistral) utilizing Proximal Policy Optimization (PPO) and LoRA.

---

## Project Overview & Objectives

Humor is highly subjective, culturally dependent, and challenging for LLMs to generate effectively without resorting to overused puns or repetitive structures. This project aims to address these limitations through a three-phase pipeline:
1. **Self-Supervised Labeling**: Using a mixture of open-source LLMs (LLaMA 3, Mistral, Gemma) to automatically annotate jokes across six predefined humor categories (Wordplay, Absurdity, CulturalReference, Relatable, Offensive, EdgyContent).
2. **Regression Modeling**: Fine-tuning an **ALBERT-base-v2** model to predict multi-label humor scores from raw joke text, eliminating the need for constant LLM API calls.
3. **Generative Optimization**: Utilizing **Reinforcement Learning (PPO + LoRA)** to fine-tune a Mistral-7B-Instruct model, incentivizing the generation of jokes that score highly in at least one designated humor category.

## 📊 Dataset & Metrics

We utilized a subset of 10,000 jokes from the **Short Jokes dataset** (Kaggle). Due to the lack of objective human annotations, we introduced the **LPR (Le Pera - Liparoti Ranking)** metric. 
The LPR metric calculates a synthetic consensus score based on the evaluations of our three baseline LLMs. It implements a controlled disagreement mechanism ensuring stable and robust scoring, acting as a reliable proxy for human evaluation.

---

## Repository Structure

The project is divided into four main notebooks and some supplementary scripts run locally:

### 1. `nlp-categorization.ipynb`
This notebook defines the humor categories and extracts baseline scores utilizing the chosen LLMs.
- **Model Checkpoints**: Refer to the `sample10000` folder.

### 2. `nlp-bert-categories.ipynb`
This notebook contains the training code for an **AlBERT model for a multi-label regression task** based on the LPR metric values obtained from the first phase.
- **Model Checkpoints**: `joke_regressor_state_dict.pt`

### 3. `nlp-generation-ppo.ipynb`
This notebook details the **improvement of the Mistral generation model** via **PPO + LoRA**. It uses a custom reward function based on the predicted category scores to guide the generative process.
- **Model Checkpoints**: Refer to the `run_v9_nlp_generation_ppo` folder.

### 4. `nlp-demo.ipynb` ([View Demo](https://www.kaggle.com/code/jacopolepera/nlp-demo))
A **demonstration** notebook integrating the developed models to interactively test and showcase the improved joke generation pipeline.

### 5. Additional Scripts
- `consensus.py`: A Python script to calculate agreement measures between classification models (Cohen’s kappa, correlations, differences) and generate aggregated scores (mean, median, mode). It outputs the final LPR score for each joke into a final CSV artifact.

---

## Execution & Environment

All notebooks were primarily developed and tested on **Kaggle Notebooks** due to GPU constraints, utilizing low-rank adaptation techniques to maximize efficiency. However, they can be seamlessly adapted to any other Jupyter-based environment given the correct dependencies are installed.

---

## Main Requirements
- Python 3.x
- PyTorch (used extensively for training loops and RLHF)
- Hugging Face Transformers & PEFT (for LoRA)
- Scikit-learn
- Pandas, NumPy, Matplotlib/Seaborn

---

## Authors
- **Jacopo Alfonso Le Pera**
- **Silvio Emanuele Liparoti**
