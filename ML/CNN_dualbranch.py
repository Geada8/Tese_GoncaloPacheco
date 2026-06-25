import tensorflow as tf  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import (Input, Reshape, Conv1D, BatchNormalization,  # type: ignore
                                     MaxPooling1D, GlobalAveragePooling1D,
                                     Flatten, Dense, Dropout, Concatenate, Cropping1D)
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore

# ── Hyperparameters ────────────────────────────────────────────────────────
EPOCHS        = 120
LEARNING_RATE = 0.001
BATCH_SIZE    = 32
DROPOUT_RATE  = 0.35
N_CLASSES     = 4

# Input shape: flattened log-magnitude spectrum
N_FREQ_BINS = 250
N_CHANNELS  = 6
SPEC_LEN    = N_FREQ_BINS * N_CHANNELS  # 1500

LF_BINS = 3

# ── Input ──────────────────────────────────────────────────────────────────
inp  = Input(shape=(SPEC_LEN,), dtype='float32')

spec = Reshape((N_FREQ_BINS, N_CHANNELS))(inp)

# ── Branch 1: 1x shaft-order branch ───────────────────────────────────────
lf = Cropping1D(cropping=(0, N_FREQ_BINS - LF_BINS))(spec)  
lf = Flatten()(lf)                                           
lf = Dense(64, activation='relu')(lf)
lf = Dense(32, activation='relu')(lf)

# ── Branch 2: Broadband Conv1D branch ─────────────────────────────────────
x = spec
for filters, kernel_size in [(64, 7), (128, 5), (128, 3)]:
    x = Conv1D(filters=filters, kernel_size=kernel_size,
               padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
x = GlobalAveragePooling1D()(x)
x = Dense(64, activation='relu')(x)

# ── Merge & classify ───────────────────────────────────────────────────────
z = Concatenate()([lf, x])
z = Dense(128, activation='relu')(z)
z = Dropout(DROPOUT_RATE)(z)
z = Dense(64, activation='relu')(z)
z = Dropout(DROPOUT_RATE)(z)
out = Dense(N_CLASSES, activation='softmax', name='y_pred')(z)

model = Model(inputs=inp, outputs=out)

# ── Training ───────────────────────────────────────────────────────────────
train_dataset      = None  # provided by Edge Impulse runtime
validation_dataset = None  # provided by Edge Impulse runtime

opt = Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=30,
    restore_best_weights=True,
    mode='max'
)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(train_dataset, epochs=EPOCHS, validation_data=validation_dataset,
          verbose=2, callbacks=[early_stopping])
