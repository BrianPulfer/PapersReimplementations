# Code from Tensorflow Examples

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa  # Community-made Tensorflow code (AdamW optimizer)


class Patches(layers.Layer):
    """Gets some images and returns the patches for each image"""

    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


def mlp(x, hidden_units, dropout_rate):
    """MLP with dropout and skip-connections"""
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class PatchEncoder(layers.Layer):
    """Adding (learnable) positional encoding to input patches"""

    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def main():
    # Downloading dataset
    num_classes = 100
    input_shape = (32, 32, 3)
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

    print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

    # Hyper-parameters
    learning_rate = 0.001
    weight_decay = 0.0001
    batch_size = 256
    num_epochs = 100
    image_size = 72  # We resize input images to this size
    patch_size = 6  # Size of each patch extracted from the image (12x12 patches of size 6x6)
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Size of the transformer layers
    transformer_layers = 8  # Nr of times we are repeating the transformer encoder
    mlp_head_units = [2048, 1024]

    # Data augmentation
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.Normalization(),
            layers.experimental.preprocessing.Resizing(image_size, image_size),
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(factor=0.02),
            layers.experimental.preprocessing.RandomZoom(height_factor=0.2, width_factor=0.2)
        ],
        name='data_augmentation'
    )

    # Normalizing based on training data
    data_augmentation.layers[0].adapt(x_train)

    def create_vit_classifier():
        # Creating classifier
        inputs = layers.Input(shape=input_shape)
        augmented = data_augmentation(inputs)
        patches = Patches(patch_size)(augmented)
        encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

        for _ in range(transformer_layers):
            # Layer normalization and self-attention
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            attention_output = layers.MultiHeadAttention(
                num_heads, key_dim=projection_dim, dropout=0.1
            )(x1, x1)

            # Residual conenction
            x2 = layers.Add()([attention_output, encoded_patches])

            # Normalization and MLP
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

            # Residual connection
            encoded_patches = layers.Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)

        # Add MLP
        features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)

        # Classify output
        logits = layers.Dense(num_classes)(features)

        model = keras.Model(inputs=inputs, outputs=logits)
        return model

    def run_experiment(model):
        optimizer = tfa.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )

        model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
                keras.metrics.SparseTopKCategoricalAccuracy(5, name='top-5 accuracy'),
            ],
        )

        checkpoint_filepath = './tmp/checkpoint'
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True
        )

        history = model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=num_epochs,
            validation_split=0.1,
            callbacks=[checkpoint_callback]
        )

        model.load_weights(checkpoint_filepath)
        _, accuracy, top5_accuracy = model.evaluate(x_test, y_test)
        print(f"Test Accuracy:       {accuracy}")
        print(f"Test Top-5 Accuracy: {top5_accuracy}")
        return history

    model = create_vit_classifier()
    run_experiment(model)


if __name__ == '__main__':
    main()
