import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
import zipfile

# Unzipping the data file
zip_ref = zipfile.ZipFile('data.zip', 'r')
zip_ref.extractall("/Dataset")
zip_ref.close()


keras.backend.clear_session()

# Loading the weights
local_weights_file = "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

# Defining the pre_trained model
pre_trained_model = InceptionV3(input_shape=(150, 150, 3), include_top=False, weights=None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
    layer.trainable = False

# pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed5')
# print('last layer output shape ', last_layer.output_shape)
last_output = last_layer.output

# Flatten the output layer to 1D
x = layers.Flatten()(last_output)

# adding fc layer with 512 hidden units and relu activation
# Add a dropout rate of 0.2

x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.2)(x)

# Add a final sigmoid layer for classification wit 3 neurons
x = layers.Dense(3, activation='softmax')(x)

model = Model(pre_trained_model.input, x)

# Using Adam Optimizer
Adam = tf.keras.optimizers.Adam(learning_rate=0.0009, beta_1=0.9, beta_2=0.999)

# Compiling the Model
model.compile(optimizer=Adam, loss='categorical_crossentropy', metrics=['acc'])

model.summary()

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True)

# Note that validation  data should not be augmented
valid_datagen = ImageDataGenerator(rescale=1. / 255)

# FLOW TRAINING-EXAMPLES IMAGES IN BATCHES OF 20 USING TRAIN_DATAGEN GENERATOR
train_generator = train_datagen.flow_from_directory("/Dataset/train",
                                                    batch_size=20,
                                                    class_mode='categorical',
                                                    target_size=(150, 150),
                                                    shuffle=True)

# FLOW VALIDATION-EXAMPLES IMAGES IN BATCHES OF 20 USING VALID_DATAGEN GENERATOR
valid_generator = valid_datagen.flow_from_directory("/Dataset/valid",
                                                    batch_size=20,
                                                    class_mode='categorical',
                                                    target_size=(150, 150),
                                                    shuffle=False)


# Callbacks:
# Exponential decay
def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 ** (epoch / s)

    return exponential_decay_fn


exponential_decay_fn = exponential_decay(lr0=0.0009, s=5)

lr_scheduler_ed = keras.callbacks.LearningRateScheduler(exponential_decay_fn)

# Checkpoint callback
checkpoint_cb = keras.callbacks.ModelCheckpoint("model.h5", save_best_only=True)

# Early stopping callback
early_stopping_m = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

history = model.fit(train_generator,
                    validation_data=valid_generator,
                    steps_per_epoch=45,
                    epochs=30,
                    validation_steps=10,
                    callbacks=[checkpoint_cb, lr_scheduler_ed, early_stopping_m],
                    verbose=1)

# Graph between Training Accuracy and Validation Accuracy
acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(history.epoch, acc, 'r', label='Training accuracy')
plt.plot(history.epoch, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.grid(True)
plt.figure()

# Graph between Training Loss and Validation Loss

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(history.epoch, loss, 'r', label='Training Loss')
plt.plot(history.epoch, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.grid(True)
plt.show()

# Graph for Learning rate ~ Exponential Decay
plt.plot(history.epoch, history.history["lr"], "o-")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title(" exponential_decay", fontsize=14)
plt.grid(True)
plt.show()
