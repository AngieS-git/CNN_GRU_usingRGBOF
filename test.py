from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, GRU, Dense, TimeDistributed, Concatenate, Add, Activation, DepthwiseConv2D, Reshape, BatchNormalization
from tensorflow.keras.models import Model

def create_rgb_cnn_architecture(input_shape=(224, 224, 3)):  # Example simple RGB CNN
    input_tensor = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(input_tensor)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)  # Flatten *after* TimeDistributed will flatten frame-wise features
    return Model(inputs=input_tensor, outputs=x)

def create_optical_flow_cnn_architecture(input_shape=(224, 224, 2)): # Example simple OF CNN - or use MobileNet here!
    input_tensor = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(input_tensor)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x) # Flatten *after* TimeDistributed will flatten frame-wise features
    return Model(inputs=input_tensor, outputs=x)

# ----- Define Input Shapes -----
rgb_input_shape = (None, 224, 224, 3) # (Time steps, Height, Width, Channels) - None for variable time steps
of_input_shape = (None, 224, 224, 2) # (Time steps, Height, Width, OF Channels - e.g., 2 for x, y flow)
num_classes = 10 # Example: 10 action classes~

# ----- Input Layers -----
input_rgb = Input(shape=rgb_input_shape)
input_of = Input(shape=of_input_shape)

# ----- CNN Models -----
rgb_cnn_model = create_rgb_cnn_architecture()
of_cnn_model = create_optical_flow_cnn_architecture() # Or replace with MobileNet!

# ----- TimeDistributed CNNs (Frame-wise feature extraction) -----
rgb_features_sequence = TimeDistributed(rgb_cnn_model)(input_rgb)
of_features_sequence = TimeDistributed(of_cnn_model)(input_of)

# ----- GRU Layers -----
rgb_gru_output = GRU(units=128)(rgb_features_sequence) # Example GRU units
of_gru_output = GRU(units=128)(of_features_sequence) # Example GRU units

# ----- Fusion (Late Fusion - Concatenate) -----
merged_features = Concatenate()([rgb_gru_output, of_gru_output])

# ----- Classification Layers -----
dense_layer = Dense(units=256, activation='relu')(merged_features)
output_layer = Dense(units=num_classes, activation='softmax')(dense_layer)

# ----- Create the 2-Stream Model -----
model = Model(inputs=[input_rgb, input_of], outputs=output_layer)

# ----- Compile the Model -----
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ----- Model Summary (Optional) -----
model.summary()

# ----- Example Training (using dummy data - replace with your actual data) -----
import numpy as np
dummy_rgb_data = np.random.rand(100, 10, 224, 224, 3) # 100 samples, 10 time steps, RGB frames
dummy_of_data = np.random.rand(100, 10, 224, 224, 2) # 100 samples, 10 time steps, Optical Flow frames
dummy_labels = keras.utils.to_categorical(np.random.randint(0, num_classes, size=(100,)), num_classes=num_classes) # Example one-hot encoded labels

model.fit([dummy_rgb_data, dummy_of_data], dummy_labels, epochs=10, batch_size=32)
