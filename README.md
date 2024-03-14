English to French Transformer Translator


This repository contains an implementation of a bare encoder and decoder transformer for translating English 
sentences to French. The transformer achieves an accuracy of approximately 79%. The translation model is trained 
without using PyTorch transformer module, making it a lightweight alternative for those seeking to understand 
the inner workings of transformers.

Prerequisites

Python 3.x
Pandas
TensorFlow
Matplotlib
NumPy

Usage

1. Preprocessing: Run preprocessor.py to preprocess the dataset. This step is necessary to prepare the data for training.

>>>> python preprocessor.py

2. Training: Execute trainer.py to train the transformer model. This script will train the model using the preprocessed dataset. Ensure to configure the batch size and number of iterations appropriately to maintain correct epochs and manage computational resources efficiently.

>>>> python trainer.py

3. Generating Translations: After training, utilize generator.py to translate English sentences to French. Pass the saved model's file path as an argument to this script.

>>>> python generator.py path_to_saved_model

Example Translations:

Input: "This is the first book I've ever done."
Output: "C'est le premier livre que j'ai jamais fait."

Input: "So I'll just share with you some stories very quickly of some magical things that have happened."
Output: "Je vais donc partager avec vous très rapidement quelques histoires de choses magiques qui se sont produites."

4. Tinkering with Configurations: Feel free to explore and modify the configuration files (config.ini) to adjust hyperparameters and experiment with different settings. Given the substantial size of the dataset (~844k sentences), be mindful of computational resources and choose batch sizes and iteration counts accordingly.


Dataset


The dataset used for training is extensive (~844k sentences), enabling the model to learn effectively. Due to its size, computing resources may be strained during preprocessing and training. Ensure sufficient resources and consider optimizing batch sizes and iterations to balance training time and accuracy.

Acknowledgments
Inspiration for this project came from the seminal work on transformers by Vaswani et al. (2017).
Thanks to the open-source community for providing invaluable resources and frameworks for natural language processing tasks.

