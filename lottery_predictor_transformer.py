import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class LotteryPredictorTransformer:
    def __init__(self, data):
        self.data = data
        self.sequence_length = 10
        self.embed_dim = 32
        self.num_heads = 4
        self.ff_dim = 64
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model_regular = None
        self.model_complementary = None

    def prepare_data(self):
        # Convertir datos a matriz numérica
        numbers = []
        for row in self.data.values:
            row_str = str(row)
            row_str = row_str.replace('[', '').replace(']', '').replace('Name:', '').replace('dtype:', '').replace('object', '').strip()
            nums = [int(n.strip()) for n in row_str.split() if n.strip()]
            if nums:
                numbers.append(nums)
        
        numbers = np.array(numbers)
        
        # Separar datos regulares y complementario
        regular_data = numbers[:, :5]
        complementary_data = numbers[:, 5:]
        
        # Preparar secuencias para números regulares
        X_regular, y_regular = [], []
        for i in range(len(regular_data) - self.sequence_length):
            sequence = regular_data[i:i + self.sequence_length]
            target = regular_data[i + self.sequence_length]
            X_regular.append(sequence)
            y_regular.append(target)
        
        # Preparar secuencias para complementario
        X_complementary, y_complementary = [], []
        for i in range(len(complementary_data) - self.sequence_length):
            sequence = complementary_data[i:i + self.sequence_length]
            target = complementary_data[i + self.sequence_length]
            X_complementary.append(sequence)
            y_complementary.append(target)
        
        return (np.array(X_regular), np.array(y_regular)), (np.array(X_complementary), np.array(y_complementary))

    def create_regular_model(self):
        # Definir las dimensiones
        seq_length = 10
        d_model = 32  # Debe coincidir con la dimensión de embedding
        
        # Capa de entrada
        inputs = layers.Input(shape=(seq_length, 5))
        
        # Proyectar la entrada a la dimensión del modelo
        x = layers.Dense(d_model)(inputs)  # Proyecta de 5 a 32 dimensiones
        
        # Crear embedding posicional
        positions = tf.range(start=0, limit=seq_length, delta=1)
        position_embedding = layers.Embedding(
            input_dim=seq_length, 
            output_dim=d_model
        )
        
        # Añadir embedding posicional
        pos_encoding = position_embedding(positions)
        pos_encoding = tf.expand_dims(pos_encoding, axis=0)  # Añadir dimensión de batch
        
        # Sumar embeddings
        x = x + pos_encoding
        
        transformer_block1 = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim)
        transformer_block2 = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim)
        
        x = transformer_block1(x)
        x = transformer_block2(x)
        
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(5, activation="sigmoid")(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def create_complementary_model(self):
        inputs = layers.Input(shape=(self.sequence_length, 1))
        
        position_embedding = layers.Embedding(
            input_dim=self.sequence_length,
            output_dim=self.embed_dim
        )
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        x = inputs + position_embedding(positions)
        
        transformer_block = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim)
        x = transformer_block(x)
        
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def train(self):
        # Preparar datos
        (X_regular, y_regular), (X_complementary, y_complementary) = self.prepare_data()
        
        # Crear y compilar modelos
        self.model_regular = self.create_regular_model()
        self.model_complementary = self.create_complementary_model()
        
        # Compilar modelo regular
        self.model_regular.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Compilar modelo complementario
        self.model_complementary.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Entrenar modelo regular
        split = int(len(X_regular) * 0.8)
        self.model_regular.fit(
            X_regular[:split], y_regular[:split],
            epochs=100,
            batch_size=32,
            validation_data=(X_regular[split:], y_regular[split:]),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Entrenar modelo complementario
        self.model_complementary.fit(
            X_complementary[:split], y_complementary[:split],
            epochs=100,
            batch_size=32,
            validation_data=(X_complementary[split:], y_complementary[split:]),
            callbacks=[early_stopping],
            verbose=0
        )

    def predict(self):
        if self.model_regular is None or self.model_complementary is None:
            return [1, 2, 3, 4, 5, 1]
        
        # Obtener últimas secuencias
        numbers = []
        for row in self.data.values[-self.sequence_length:]:
            row_str = str(row)
            row_str = row_str.replace('[', '').replace(']', '').replace('Name:', '').replace('dtype:', '').replace('object', '').strip()
            nums = [int(n.strip()) for n in row_str.split() if n.strip()]
            if nums:
                numbers.append(nums)
        
        numbers = np.array(numbers)
        
        # Predecir números regulares
        last_regular = numbers[:, :5].reshape(1, self.sequence_length, 5)
        pred_regular = self.model_regular.predict(last_regular, verbose=0)[0]
        regular_numbers = [max(1, min(43, round(x))) for x in pred_regular]
        
        # Asegurar números únicos y ordenados
        regular_numbers = list(set(regular_numbers))
        while len(regular_numbers) < 5:
            new_num = np.random.randint(1, 44)
            if new_num not in regular_numbers:
                regular_numbers.append(new_num)
        regular_numbers = sorted(regular_numbers[:5])
        
        # Predecir complementario
        last_complementary = numbers[:, 5:].reshape(1, self.sequence_length, 1)
        pred_complementary = self.model_complementary.predict(last_complementary, verbose=0)[0][0]
        complementary = max(1, min(16, round(pred_complementary)))
        
        # Asegurar que el complementario no está en los números regulares
        while complementary in regular_numbers:
            complementary = (complementary % 16) + 1
        
        return regular_numbers + [complementary] 