import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, AveragePooling1D, Flatten, Reshape, BatchNormalization, Activation
from tensorflow.keras.optimizers.legacy import Adam
# IMPORT EARLY STOPPING
from tensorflow.keras.callbacks import EarlyStopping 

EPOCHS = args.epochs or 60
LEARNING_RATE = args.learning_rate or 0.0005
ENSURE_DETERMINISM = args.ensure_determinism
BATCH_SIZE = args.batch_size or 32

if not ENSURE_DETERMINISM:
    train_dataset = train_dataset.shuffle(buffer_size=BATCH_SIZE*4)
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False)
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=False)

# =====================================================================
# START OF 25-LAYER ARCHITECTURE (FROM PAPER)
# =====================================================================
model = Sequential()

# Layer 1: Input & Reshape (Splits the flat 15,000 array into 2500 timesteps across 6 axes)
model.add(Reshape((int(input_length / 6), 6), input_shape=(input_length, )))

# Layers 2-5: Convolution 1 Block
model.add(Conv1D(filters=8, kernel_size=10, strides=1, padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(AveragePooling1D(pool_size=2, strides=2, padding='same'))

# Layers 6-9: Convolution 2 Block
model.add(Conv1D(filters=16, kernel_size=10, strides=1, padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(AveragePooling1D(pool_size=2, strides=2, padding='same'))

# Layers 10-11: Convolution 3
model.add(Conv1D(filters=32, kernel_size=10, strides=1, padding='same'))
model.add(Activation('relu'))

# Layers 12-13: Convolution 4
model.add(Conv1D(filters=32, kernel_size=10, strides=1, padding='same'))
model.add(Activation('relu'))

# Layers 14-16: Convolution 5 Block
model.add(Conv1D(filters=32, kernel_size=10, strides=1, padding='same'))
model.add(Activation('relu'))
model.add(AveragePooling1D(pool_size=2, strides=2, padding='same'))

# Layers 17-22: Fully Connected Network
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

# Layers 23-25: Output Classification
model.add(Dense(classes, name='y_pred', activation='softmax'))
# =====================================================================
# END OF ARCHITECTURE
# =====================================================================

opt = Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999)

#  EARLY STOPPING CALLBACK
early_stopping = EarlyStopping(
    monitor='val_loss',         #
    patience=30,                
    restore_best_weights=True   
)

if 'callbacks' not in locals():
    callbacks = []

callbacks.append(BatchLoggerCallback(BATCH_SIZE, train_sample_count, epochs=EPOCHS, ensure_determinism=ENSURE_DETERMINISM))
callbacks.append(early_stopping) 

# Neural Network training
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
model.fit(train_dataset, epochs=EPOCHS, validation_data=validation_dataset, verbose=2, callbacks=callbacks)

disable_per_channel_quantization = False
