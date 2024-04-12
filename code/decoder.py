import tensorflow as tf

try: from transformer import TransformerBlock, PositionalEncoding
except Exception as e: print(f"TransformerDecoder Might Not Work, as components failed to import:\n{e}")

########################################################################################

class RNNDecoder(tf.keras.layers.Layer):

    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        # TODO:
        # Now we will define image and word embedding, decoder, and classification layers
        

        # Define feed forward layer(s) to embed image features into a vector 
        # with the models hidden size
        self.image_dense1 = tf.keras.layers.Dense(64,activation=None,kernel_initializer="glorot_uniform",bias_initializer="zeros" )
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.image_embedding = tf.keras.layers.Dense(self.hidden_size,activation=None,kernel_initializer="glorot_uniform",bias_initializer="zeros" )
        

        # Define english embedding layer:
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.hidden_size)
        print("Saved")


        # Define decoder layer that handles language and image context:     
        self.decoder = tf.keras.layers.GRU(self.hidden_size, return_sequences=True)

        # Define classification layer(s) (LOGIT OUTPUT)
        self.classifier_dense1 = tf.keras.layers.Dense(self.hidden_size,activation=None,kernel_initializer="glorot_uniform",bias_initializer="zeros" )
        self.classifier = tf.keras.layers.Dense(self.vocab_size,activation=None,kernel_initializer="glorot_uniform",bias_initializer="zeros" )

    def call(self, encoded_images, captions):
        # TODO:
        # 1) Embed the encoded images into a vector of the correct dimension for initial state
        # 2) Pass your english sentance embeddings, and the image embeddings, to your decoder 
        # 3) Apply dense layer(s) to the decoder to generate prediction **logits**
        denseimage_1 = self.image_dense1(encoded_images)
        #print(denseimage_1.shape)
        r1 = self.leaky_relu(denseimage_1)
        embedded_images = self.image_embedding(r1)
        embedded_captions = self.embedding(captions)
        #initial_state = [embedded_images, tf.zeros_like(embedded_images)]
        #print(embedded_captions.shape)
        decoded_images = self.decoder(embedded_captions,initial_state=embedded_images)
        #print(decoded_images)
        d1 = self.classifier_dense1(decoded_images)
        r2 = self.leaky_relu(d1)
        logits = self.classifier(r2)
     
        return logits


########################################################################################

class TransformerDecoder(tf.keras.Model):

    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        # TODO: Define image and positional encoding, transformer decoder, and classification layers

        # Define feed forward layer(s) to embed image features into a vector 
        self.image_embedding = tf.keras.Sequential([tf.keras.layers.Dense(64,activation='relu',kernel_initializer="glorot_uniform",bias_initializer="zeros" ), tf.keras.layers.Dense(self.hidden_size,activation=None,kernel_initializer="glorot_uniform",bias_initializer="zeros" )])

        # Define positional encoding to embed and offset layer for language:
        self.encoding = PositionalEncoding(self.vocab_size,self.hidden_size,self.window_size)
        print(self.encoding)

        # Define transformer decoder layer:
        self.decoder = TransformerBlock(self.hidden_size)

        # Define classification layer(s) (LOGIT OUTPUT)
        self.classifier = tf.keras.Sequential([tf.keras.layers.Dense(self.hidden_size,activation='relu',kernel_initializer="glorot_uniform",bias_initializer="zeros" ),tf.keras.layers.Dense(self.vocab_size,activation=None,kernel_initializer="glorot_uniform",bias_initializer="zeros" )])

    def call(self, encoded_images, captions):
        # TODO:
        # 1) Embed the encoded images into a vector (HINT IN NOTEBOOK)
        # 2) Pass the captions through your positional encoding layer
        # 3) Pass the english embeddings and the image sequences to the decoder
        # 4) Apply dense layer(s) to the decoder out to generate **logits**
        embedded_images = self.image_embedding(encoded_images)
        embedded_captions = self.encoding(captions)
        decoded_images = self.decoder(inputs=embedded_captions,context_sequence=embedded_images)
        logits = self.classifier(decoded_images)
        return logits
