import os
import json
import numpy as np

def run_experiment_script():
    script_path = r"d:\Vit\Academics Sem-5\Deep Learning (DL)\Lab\Ass3\run_experiment.py"
    
    code = '''import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

print("Starting CIFAR-10 Practical 3 Experiment...")

# 1. Load Data
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

X_train = X_train_raw.astype('float32') / 255.0
X_test = X_test_raw.astype('float32') / 255.0

y_train = to_categorical(y_train_raw, 10)
y_test = to_categorical(y_test_raw, 10)

def build_model():
    model = Sequential([
        Flatten(input_shape=(32, 32, 3)),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

# 2. Experiment 1: Learning Rates
learning_rates = [0.1, 0.01, 0.001, 0.0001]
lr_results = {}

for lr in learning_rates:
    print(f"Training for Learning Rate: {lr}")
    model = build_model()
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, 
                        validation_data=(X_test, y_test),
                        epochs=15, batch_size=128, verbose=0)
    lr_results[str(lr)] = {
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']]
    }

# 3. Experiment 2: Epochs
print("Training for 50 Epochs with lr=0.001...")
best_model = build_model()
best_model.compile(optimizer=Adam(learning_rate=0.001),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
epoch_history = best_model.fit(X_train, y_train,
                               validation_data=(X_test, y_test),
                               epochs=50, batch_size=128, verbose=0)

epoch_results = {
    'loss': [float(x) for x in epoch_history.history['loss']],
    'val_loss': [float(x) for x in epoch_history.history['val_loss']],
    'accuracy': [float(x) for x in epoch_history.history['accuracy']],
    'val_accuracy': [float(x) for x in epoch_history.history['val_accuracy']]
}

# 4. Save results to JSON
results_path = r"d:\\Vit\\Academics Sem-5\\Deep Learning (DL)\\Lab\\Ass3\\experiment_results.json"
with open(results_path, 'w') as f:
    json.dump({'lr_results': lr_results, 'epoch_results': epoch_results}, f, indent=2)

print(f"Saved experiment results to {results_path}")
'''
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(code)
    print(f"Created {script_path}")

if __name__ == '__main__':
    run_experiment_script()
