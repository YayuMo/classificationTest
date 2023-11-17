import tensorflow as tf
from tensorflow.python.keras.optimizers import gradient_descent_v2
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from dataGenerator import dataGeneration
from datasetsOperation import dataOperation

base_dir = 'catdogdata/cats_and_dogs_filtered'
train_dir, valid_dir = dataOperation(base_dir)
train_generator, valid_generator = dataGeneration(train_dir, valid_dir)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer=RMSprop(learning_rate=0.00001),
              loss='categorical_crossentropy',
              metrics = ['acc'])

history = model.fit_generator(train_generator,
                              validation_data=valid_generator,
                              steps_per_epoch=100,
                              epochs=15,
                              validation_steps=50,
                              verbose=2)

