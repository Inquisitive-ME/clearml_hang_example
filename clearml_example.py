import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils as np_utils
from clearml import Task

task = Task.init(project_name="test",
                 task_name="CLEARML Example",
                 continue_last_task=False,
                 auto_connect_streams=True,
                 auto_resource_monitoring=False)

# Train a simple deep NN on the MNIST dataset.
# Gets to 98.40% test accuracy after 20 epochs
# (there is *a lot* of margin for parameter tuning).
# 2 seconds per epoch on a K520 GPU.

# the data, shuffled and split between train and test sets
nb_classes = 10
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_train = X_train.reshape(60000, 784).astype('float32') / 255.
X_test = X_test.reshape(10000, 784).astype('float32') / 255.
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = keras.models.Sequential()
model.add(keras.layers.Dense(512, input_shape=(784,)))
model.add(keras.layers.Activation('relu'))

model.add(keras.layers.Dense(512))
model.add(keras.layers.Activation('relu'))

model.add(keras.layers.Dense(10))
model.add(keras.layers.Activation('softmax'))

model2 = keras.models.Sequential()
model2.add(keras.layers.Dense(512, input_shape=(784,)))
model2.add(keras.layers.Activation('relu'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

# Advanced: setting model class enumeration and set it for the task
labels = dict(('digit_%d' % i, i) for i in range(10))
task.set_model_label_enumeration(labels)

output_folder = 'keras_example'

board = keras.callbacks.TensorBoard(histogram_freq=1, log_dir=output_folder, write_images=False)
model_store = keras.callbacks.ModelCheckpoint(filepath=os.path.join(output_folder, 'weight.{epoch}.keras'))

# Fit and evaluate the model

class RandomImageLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super(RandomImageLogger, self).__init__()
        self.log_dir = log_dir
        self.file_writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        # Create a random image
        image = np.random.rand(2560, 2560, 3) * 255
        image = image.astype(np.uint8)

        # Log the image
        with self.file_writer.as_default():
            tf.summary.image("Random Image", np.expand_dims(image, 0), step=epoch)

random_image_logger = RandomImageLogger(output_folder)

history = model.fit(X_train,
                    Y_train,
                    batch_size=128,
                    epochs=60,
                    callbacks=[board, model_store],
                    verbose=1,
                    validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])