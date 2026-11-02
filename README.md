# CI3201: Deep Learning - Lab Assignments
**Vishwakarma Institute of Technology, Pune**  
**Department of Computer Engineering (Artificial Intelligence)**

This repository contains the complete practical assignments for the Deep Learning course, focusing on building, training, and analyzing various Neural Network architectures including MLPs, RNNs, CNNs, Transformers, and Multimodal Voice Systems.

## 📂 Repository Structure

The repository is structured by assignment. Each folder contains the Jupyter Notebook, a detailed explanation Markdown file, a formatted `.docx` report, and designated folders for datasets and generated plots.

### Assignments Index

* **[Assignment 1: Data Preprocessing](./Ass1/EXPLANATION.md)**
  * Environment setup, TensorFlow/Keras basics, and data normalization using the Fashion MNIST dataset.
* **[Assignment 2: Perceptron Implementation](./Ass2/EXPLANATION.md)**
  * Single-Layer Perceptron built from scratch using NumPy to understand weights, biases, and activation functions.
* **[Assignment 2.2: Multilayer Perceptron](./Ass2_2/EXPLANATION.md)**
  * Deep Feedforward MLP implemented using PyTorch on the Wine dataset, evaluating via accuracy metrics and a confusion matrix.
* **[Assignment 3: Forward/Backpropagation & Hyperparameters](./Ass3/EXPLANATION.md)**
  * Deep Neural Network on CIFAR-10. Analyzes the impact of varying learning rates and epoch counts on model convergence.
* **[Assignment 4: LSTM Time-Series Forecasting](./Ass4/EXPLANATION.md)**
  * Recurrent Neural Network built with Keras Functional API. Uses historical Apple (AAPL) stock data to predict future closing prices via Long Short-Term Memory cells.
* **[Assignment 5: Sequence Classification (RNN vs LSTM vs GRU)](./Ass5/EXPLANATION.md)**
  * Evaluates and compares performance across three major Recurrent Neural Network architectures on a sentiment analysis dataset (Amazon Alexa Reviews) using PyTorch.
* **[Assignment 6: Custom CNN for Image Classification](./Ass6/EXPLANATION.md)**
  * A 4-block Convolutional Neural Network with data augmentation to classify a unique Mango Leaf & Fruit Pathology dataset.
* **[Assignment 7: Transfer Learning (AlexNet, VGG16, ResNet50, EfficientNetB0)](./Ass7/EXPLANATION.md)**
  * Reusing ImageNet-pretrained CNN feature extractors by freezing early layers and fine-tuning standard architectural heads on new image domains.
* **[Assignment 8: Pre-trained BERT for Sentiment Analysis](./Ass8/EXPLANATION.md)**
  * Fine-tuning `bert-base-uncased` with bidirectional self-attention on the Stanford Sentiment Treebank (SST-2) dataset for binary text classification.
* **[Assignment 9: Vision Transformer (ViT) vs. CNN Baseline](./Ass9/EXPLANATION.md)**
  * Comparative image classification benchmarking pre-trained **ViT-B/16** against a **ResNet18** CNN on the Intel Natural Scenes dataset.
* **[Assignment 10: Audio Classification with STT & TTS](./Ass10/EXPLANATION.md)**
  * Spoken command audio recognition using a 2D CNN trained on 64-bin log-Mel spectrograms, paired with **OpenAI Whisper** for Speech-to-Text and **pyttsx3** for Text-to-Speech synthesis on Google Speech Commands.

## 🛠️ Typical Folder Layout

```text
├── .gitignore
├── README.md
├── Ass1/
├── Ass2/
├── Ass2_2/
├── Ass3/
├── Ass4/
├── Ass5/
├── Ass6/
├── Ass7/
├── Ass8/
├── Ass9/
│   ├── 60_Assignment-9_DL.docx
│   ├── Assignment9.ipynb
│   ├── EXPLANATION.md
│   ├── Datasets/
│   └── Plots/
└── Ass10/
    ├── 60_Assignment-10_DL.docx
    ├── Assignment10.ipynb
    ├── EXPLANATION.md
    ├── Datasets/
    └── Plots/
```
