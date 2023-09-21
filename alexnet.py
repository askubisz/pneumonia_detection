import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
import numpy as np

################ LOAD AND SPLIT DATA
np.random.seed(1)
data=tf.keras.utils.image_dataset_from_directory('chest_xray/train', image_size=(227,227))
data=data.map(lambda x, y: (x/255, y))

len(data)
train_size=int(len(data)*0.8)
val_size=int(len(data)*0.2)+1

train=data.take(train_size)
val=data.skip(train_size).take(val_size)

############## ALEXNET
image_shape=(227,227,3)

model=Sequential()

# 1st Conv layer
model.add(Conv2D(filters=96, input_shape=image_shape, kernel_size=(11,11), strides=(4,4), padding='valid', activation='relu'))
model.add(BatchNormalization(axis=3))

# Max Pooling
model.add(MaxPooling2D(pool_size=(3,3), strides=(2, 2)))

# 2nd Conv layer
model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))
model.add(BatchNormalization(axis=3))

# Max Pooling
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

# 3rd Conv layer
model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(BatchNormalization(axis=3))

# 4th Conv layer
model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(BatchNormalization(axis=3))

# 5th Conv layer
model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(BatchNormalization(axis=3))

# Max Pooling
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

# Flatten
model.add(Flatten())

# 1st Fully Connected Layer
model.add(Dense(4096, input_shape=(227*227*3,), activation='relu'))
# Dropout to prevent overfitting
model.add(Dropout(0.4))

# 2nd Fully Connected Layer
model.add(Dense(4096, activation='relu'))
# Dropout to prevent overfitting
model.add(Dropout(0.4))

# Output layer
model.add(Dense(1, activation='sigmoid'))

# Compile
model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])


######### TRAIN MODEL
from datetime import datetime
logdir="logs/fit/alexnet" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir=logdir)
filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
hist=model.fit(train, epochs=10, validation_data=val, callbacks=[checkpoint, tensorboard_callback])

import visualkeras
visualkeras.layered_view(model, index_ignore=[14,16], type_ignore=[BatchNormalization, Flatten, Dropout], legend=True, to_file='alexnet.png')


####### SAVE MODEL
import os
model.save(os.path.join('models','alexnet.h5'))


########## LOAD MODEL

model = keras.models.load_model('models/alexnet.hdf5')


######### TEST MODEL
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy, F1Score
test_images=tf.keras.utils.image_dataset_from_directory('chest_xray/test', image_size=(227,227))
test_images=test_images.map(lambda x, y: (x/255, y))


######### SHOW METRICS
pre=Precision()
re=Recall()
acc=BinaryAccuracy()

for batch in test_images.as_numpy_iterator():
    x, y=batch
    yhat=model.predict(x)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)
print(pre.result().numpy())
print(re.result().numpy())
print(acc.result().numpy())

# PLOT RESULTS
fig=plt.figure()
plt.plot(range(1,11),hist.history['loss'], color='blue', label='loss')
plt.plot(range(1,11),hist.history['val_loss'], color='red', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend()
plt.savefig('alex_loss.png')
plt.show()

fig=plt.figure()
plt.plot(range(1,11),hist.history['accuracy'], color='blue', label='accuracy')
plt.plot(range(1,11),hist.history['val_accuracy'], color='red', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend()
plt.savefig('alex_acc.png')
plt.show()

