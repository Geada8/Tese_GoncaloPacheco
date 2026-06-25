import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, LSTM, Reshape, InputLayer,
    Conv1D, MaxPooling1D, BatchNormalization, Bidirectional
)
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tempfile, os

EPOCHS        = args.epochs        or 60
LEARNING_RATE = args.learning_rate or 0.0005
BATCH_SIZE    = args.batch_size    or 32

train_dataset      = train_dataset.batch(BATCH_SIZE, drop_remainder=False)
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=False)

TIMESTEPS = int(input_length / 6)
N_AXES    = 6

model = Sequential()
model.add(InputLayer(input_shape=(input_length,), name='x_input'))
model.add(Reshape((TIMESTEPS, N_AXES)))

model.add(Conv1D(filters=64, kernel_size=7, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=4))
model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=4))

model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(32, return_sequences=False)))
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(classes, name='y_pred', activation='softmax'))

opt = Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Save best epoch weights to a temp file, restore at end
best_weights_path = os.path.join(tempfile.gettempdir(), 'best_weights.h5')

callbacks = [
    ModelCheckpoint(
        filepath=best_weights_path,
        monitor='val_loss',       # tracks validation loss
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=30,              # stop if no improvement for 10 epochs
        restore_best_weights=True,
        verbose=1,
    ),
]

model.fit(train_dataset,
          epochs=EPOCHS,
          validation_data=validation_dataset,
          callbacks=callbacks,
          verbose=2)

# Restore best epoch weights in case EarlyStopping didn't trigger
model.load_weights(best_weights_path)

disable_per_channel_quantization = False