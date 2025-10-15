import tensorflow as tf
import numpy as np

class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.output_layer = tf.keras.layers.Dense(vocab_size)
    
    def call(self, inputs):
        # inputs: target word indices
        x = self.embedding(inputs)
        logits = self.output_layer(x)
        return logits

# Example parameters
vocab_size = 10000
embedding_dim = 100
model = Word2Vec(vocab_size, embedding_dim)

# Example forward pass
target_word = tf.constant([42])  # word index
logits = model(target_word)
print("Logits shape:", logits.shape)  # (1, vocab_size)
