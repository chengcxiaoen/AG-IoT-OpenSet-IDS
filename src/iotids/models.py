"""Neural models for the Agri-OpenSet-IDS experiments."""

from __future__ import annotations

import tensorflow as tf


def build_lightweight_dnn(
    input_dim: int,
    num_classes: int,
    learning_rate: float = 1e-4,
    dropout: float = 0.30,
) -> tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    """Build the paper-style lightweight DNN plus embedding/logit probes."""

    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_256")(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn_256")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout_256")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_128")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout_128")(x)
    embedding = tf.keras.layers.Dense(64, activation="relu", name="embedding")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout_embedding")(embedding)
    logits = tf.keras.layers.Dense(num_classes, name="logits")(x)
    outputs = tf.keras.layers.Activation("softmax", name="softmax")(logits)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="lightweight_dnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    embedding_model = tf.keras.Model(inputs=inputs, outputs=embedding, name="embedding_model")
    logits_model = tf.keras.Model(inputs=inputs, outputs=logits, name="logits_model")
    return model, embedding_model, logits_model


def build_benign_autoencoder(
    input_dim: int,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Build a small benign-only boundary model for near-normal attacks."""

    hidden_1 = max(32, min(128, input_dim))
    bottleneck = max(8, min(32, input_dim // 4))

    inputs = tf.keras.Input(shape=(input_dim,), name="benign_features")
    x = tf.keras.layers.Dense(hidden_1, activation="relu", name="encoder_dense")(inputs)
    x = tf.keras.layers.Dense(bottleneck, activation="relu", name="bottleneck")(x)
    x = tf.keras.layers.Dense(hidden_1, activation="relu", name="decoder_dense")(x)
    outputs = tf.keras.layers.Dense(input_dim, activation="linear", name="reconstruction")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="benign_autoencoder")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model


def reconstruction_error(
    model: tf.keras.Model,
    X: tf.Tensor | object,
    batch_size: int,
) -> object:
    reconstructed = model.predict(X, batch_size=batch_size, verbose=0)
    return tf.reduce_mean(tf.square(tf.convert_to_tensor(X) - reconstructed), axis=1).numpy()
