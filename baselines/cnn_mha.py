# -*- coding: utf-8 -*-
"""CNN-MHA
#CNN + Multi-head Attention (Baseline Model)
"""

# ==== Imports ====
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, MaxPooling2D,
                                     Reshape, MultiHeadAttention, GlobalAveragePooling1D,
                                     Dense, Dropout)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, f1_score
from keras_tuner.tuners import BayesianOptimization

# ==== Load Your Data ====
X = np.array(connectomes).reshape(-1, 200, 200, 1)  # Shape: (samples, 200, 200, 1)
y = np.array(y)  # Binary labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==== Define Model Builder ====
def build_model(hp):
    inputs = Input(shape=(200, 200, 1))

    x = Conv2D(
        filters=hp.Int('conv1_filters', 32, 128, step=32),
        kernel_size=3, activation='relu', padding='same',
        kernel_regularizer=l2(hp.Choice('l2_reg1', [0.001, 0.01]))
    )(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(
        filters=hp.Int('conv2_filters', 64, 256, step=64),
        kernel_size=3, activation='relu', padding='same',
        kernel_regularizer=l2(hp.Choice('l2_reg2', [0.001, 0.01]))
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    conv3_filters = hp.Int('conv3_filters', 128, 256, step=64)
    x = Conv2D(conv3_filters, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Reshape((25*25, conv3_filters))(x)

    attention = MultiHeadAttention(
        num_heads=hp.Choice('num_heads', [2, 4, 8]),
        key_dim=hp.Choice('key_dim', [32, 64])
    )(x, x)

    x = GlobalAveragePooling1D()(attention)

    x = Dense(
        units=hp.Int('dense1_units', 32, 128, step=32),
        activation='relu'
    )(x)
    x = Dropout(hp.Float('dropout1', 0.2, 0.5, step=0.1))(x)

    x = Dense(
        units=hp.Int('dense2_units', 16, 64, step=16),
        activation='relu'
    )(x)
    x = Dropout(hp.Float('dropout2', 0.2, 0.5, step=0.1))(x)

    outputs = Dense(1, activation='sigmoid', dtype='float32')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('lr', [1e-2, 1e-3, 1e-4])),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# ==== Bayesian Tuner Setup ====
tuner = BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=10,  # Increase for better tuning
    directory='bayesian_tuner',
    project_name='cnn_attention_asd'
)

# ==== Run Hyperparameter Search ====
tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=1)

# ==== Best Model ====
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

# ==== Evaluation ====
y_pred_prob = best_model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)
f1 = f1_score(y_test, y_pred)

print(f"\nAccuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")
print(f"F1 Score: {f1:.4f}")
