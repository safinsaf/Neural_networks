from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import SGD

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

def make_model():
  model = Sequential()
  model.add(Conv2D(filters=96, kernel_size=5, input_shape=(32,32,3), activation="relu"))
  model.add(MaxPool2D(pool_size=2, strides=1))
  model.add(Conv2D(filters=128, kernel_size=3, input_shape=(32,32,3), activation="relu"))
  model.add(MaxPool2D(pool_size=2))

  model.add(Flatten())
  model.add(Dense(512, activation='relu'))
  model.add(Dense(512, activation='relu'))
  model.add(Dense(10, activation='relu'))
  return model

model = make_model()
model.compile(optimizer=SGD(1e-4), loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=x_train, y=y_train, validation_split=0.15,
          epochs=30, verbose=2,
          batch_size=128)
print(model.evaluate(x_train, y_train))