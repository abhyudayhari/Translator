# English to French Transformer Translator

This repository contains an implementation of a bare encoder and decoder transformer for translating English sentences to French. The transformer model achieves an accuracy of approximately 79%. It is designed without using PyTorch's transformer module, making it a lightweight alternative for those seeking to understand the inner workings of transformers.

## Prerequisites

Ensure the following dependencies are installed:

- Python 3.x
- Pandas
- TensorFlow
- Matplotlib
- NumPy

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/english-to-french-transformer.git
    ```

2. Navigate to the project directory:

    ```bash
    cd english-to-french-transformer
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Preprocessing
Run `preprocessor.py` to preprocess the dataset. This step is essential to prepare the data for training.
  ```bash
  python preprocessor.py
  ```

### 2. Training
Execute `trainer.py` to train the transformer model. This script will train the model using the preprocessed dataset. Adjust the batch size and number of iterations to ensure correct epochs and efficient resource management.

 ```bash
  python trainer.py
  ```

### 3. Generating Translations
After training, use `generator.py` to translate English sentences to French. Provide the saved model's file path as an argument.

 ```bash
  python generatory.py path_to_saved_model
  ```

## EXAMPLE TRANSLATIONS:

**Input:** "This is the first book I've ever done."
**Output:** "C'est le premier livre que j'ai jamais fait."

**Input:** "So I'll just share with you some stories very quickly of some magical things that have happened."
**Output:** "Je vais donc partager avec vous très rapidement quelques histoires de choses magiques qui se sont produites."

### TINKERING WITH CONFIGURATIONS

Feel free to explore and modify the configuration files `config.ini`  to adjust hyperparameters and experiment with different settings. The dataset (~844k sentences) is substantial, so be mindful of computational resources. Optimize batch sizes and iteration counts to balance training time and accuracy.

### CONTRIBUTING
We welcome contributions to this project! If you'd like to contribute, follow these steps:

1. Fork the repository.
2. Clone your forked repository to your local machine.
3. Create a new branch for your feature or fix.
4. Make your changes and commit them with descriptive messages.
5. Push your changes to your forked repository.
6. Create a pull request to the main repository.

### LICENSE
This project is licensed under the MIT License
