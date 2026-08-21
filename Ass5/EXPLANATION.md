# Assignment 5: RNN vs LSTM vs GRU

## 🎯 Objective
Implement and compare Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM), and Gated Recurrent Units (GRU) for binary sentiment classification using the **Amazon Alexa Reviews** dataset.

## 🧠 Concepts Covered
- **Text Preprocessing**: Tokenization, Vocabulary Indexing, Sequence Padding.
- **Vanilla RNNs**: Temporal sequence passing and the vulnerability to vanishing gradients.
- **LSTM Networks**: Cell states, Forget/Input/Output gating mechanisms.
- **GRU Networks**: Reset and Update gates for a more computationally efficient temporal architecture.
- **PyTorch `nn.Embedding`**: Converting integer sequences to dense continuous feature vectors.
- **Performance Analysis**: Empirical comparison of accuracy and training speed between the architectures.

## 📁 Folder Structure
- `Assignment5.ipynb`: Core implementation notebook (PyTorch).
- `60_Assignment-5_DL.docx`: Submission-ready lab report.
- `Datasets/`: Contains `amazon_alexa.tsv`.
- `Plots/`: Contains benchmark visualization `rnn_lstm_gru_comparison.png`.
