# Assignment 4: LSTM Time-Series Forecasting

## 🎯 Objective
Develop an LSTM-based model for time-series forecasting. This assignment implements a Long Short-Term Memory network to predict Apple (AAPL) stock prices using historical sequence data.

## 🧠 Concepts Covered
- Extracting and analyzing historical financial time-series data (`yfinance`).
- Time-series preprocessing: Data scaling (`MinMaxScaler`) and generating rolling sequence windows.
- Resolving vanishing gradients using LSTM gating architectures.
- Building models via the **Keras Functional API** (`Input -> LSTM -> Dropout -> Dense`).
- Implementing `EarlyStopping` callbacks to optimize training epochs dynamically.
- Evaluating regression performance using Root Mean Squared Error (RMSE) and plotting trends.

## 📁 Folder Structure
- `Assignment4.ipynb`: Core implementation notebook.
- `60_Assignment-4_DL.docx`: Submission-ready lab report.
- `Datasets/`: Folder reserved for local data files (e.g., saved `AAPL_stock_prices.csv`).
- `Plots/`: Folder reserved for saving evaluation graphs and prediction trends.
