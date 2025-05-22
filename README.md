# Adapting While Learning: Grounding LLMs for Scientific Problems with Tool Usage Adaptation

Code for paper "Adapting While Learning: Grounding LLMs for Scientific Problems with Tool Usage Adaptation", presented at ICML2025.

## Installation

To set up the environment, you'll need Python 3.9 and the required dependencies. You can create the environment and install the required packages using the following commands:

```bash
conda create -n awl python=3.9.19
conda activate awl
pip install -r requirements.txt
```

The emulators used in climate and epidemiology scenarios are respectively adapted from [MFRNP](https://github.com/Rose-STL-Lab/MFRNP) and [INP](https://github.com/Rose-STL-Lab/Interactive-Neural-Process). You can download the pre-trained surrogate neural nets from [this link](https://drive.google.com/drive/folders/1Q-KwQnrxME3txfut0sGgbiRhJy9PCWPp?usp=share_link), and put the tools for climate and epidemiology respectively under `src/Climate` and `src/Epidemiology`.

The tool-calling in this work is achieved through Llama-3.1-8B-Instruct's [chat template](https://huggingface.co/docs/transformers/main/chat_templating).

## Data Preparation and Training

![pipeline](assets/pipeline.png)

The workflow for training and evaluating the model, as illustrated in the paper and above figure, includes the following steps:

1. LLMs interact with external tools to solve scientific problems
2. Based on the interaction with tools, a solution is generated.
3. The LLMs are evaluated on the questions to categorize the dataset into easy and hard problems.
4. The World Knowledge Learning (WKL) and Tool Usage Adaptation (TUA) data are combined to further train the model.

Each sub-folder in the repository corresponds to a dataset, and each sub-folder contains scripts that execute these components in sequence.

```bash
# Generate a set of problems for training and testing.
python make_problems.py

# Use an LLM-Agent to answer the generated questions.
python answer_agent.py

# Generate solutions for the questions
python generate_solution.py

# Create training data based on the input file and model
python make_training_data.py

# Generate answers using the LLM for the given input questions.
python answer_llm.py
```

Additionally, the Climate and Epidemiology folders include code related to open-ended questions.

## Evaluation

We provide the testing data in the `test_set` folder. For Questions with Definite Answers we provide a standard answers along with the questions. For open-ended questions, we provide corresponding evaluation scripts (`Climate/evaluate_open.py` and `Epidemiology/eval_open.py`).

## Utils

The `utils` folder contains utility scripts related to the following components:

1. **Model utilities**: Scripts for loading and interacting with open-source and closed-source (proprietary) language models. You should replace the `API_KEY` with your own one.
2. **Dataset construction**: Tools for preprocessing, formatting, and generating datasets used in training and evaluation.
3. **Mathematical evaluation**: Utilities to assess and evaluate mathematical outputs.

