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
data=tf.keras.utils.image_dataset_from_directory('chest_xray/train', image_size=(224,224))
data=data.map(lambda x, y: (x/255, y))

len(data)
train_size=int(len(data)*0.8)
val_size=int(len(data)*0.2)+1

train=data.take(train_size)
val=data.skip(train_size).take(val_size)


############## VGG 16
pre_trained_model=tf.keras.applications.VGG16(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
for layer in pre_trained_model.layers:
    layer.trainable=False
x=Flatten()(pre_trained_model.output)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(pre_trained_model.input, x)
model.compile(optimizer='adam', loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])


######### TRAIN MODEL
from datetime import datetime
logdir="logs/fit/vgg16" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir=logdir)
filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
hist=model.fit(train, epochs=5, validation_data=val, callbacks=[checkpoint, tensorboard_callback])


####### SAVE MODEL
import os
model.save(os.path.join('models','vgg16.h5'))

########## LOAD MODEL

model = keras.models.load_model('models/vgg16.h5')

# import visualkeras
# visualkeras.layered_view(model, index_ignore=[0], type_ignore=[Flatten, BatchNormalization], legend=True, to_file='vgg16.png')


######### TEST MODEL
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
test_images=tf.keras.utils.image_dataset_from_directory('chest_xray/test', image_size=(224,224))
test_images=test_images.map(lambda x, y: (x/255, y))

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
plt.plot(range(1,6),hist.history['loss'], color='blue', label='loss')
plt.plot(range(1,6),hist.history['val_loss'], color='red', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend()
plt.savefig('vgg_loss.png')
plt.show()

fig=plt.figure()
plt.plot(range(1,6),hist.history['accuracy'], color='blue', label='accuracy')
plt.plot(range(1,6),hist.history['val_accuracy'], color='red', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend()
plt.savefig('vgg_acc.png')
plt.show()

