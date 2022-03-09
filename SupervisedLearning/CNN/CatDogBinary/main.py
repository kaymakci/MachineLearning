import os, shutil
import matplotlib.pyplot as plt
import cv2

from keras import layers
from keras import models, optimizers
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers

original_dataset_dir = '/home/omer/PycharmProjects/CatDogBinary/dataset/PetImages'

# create a folder to store our small sample of images
base_dir = '../cats_and_dogs_small'
#os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
#os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')
#os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test')
#os.mkdir(test_dir)

# create sub-folders Cat and Dog
train_Cat_dir = os.path.join(train_dir, 'Cat')
#os.mkdir(train_Cat_dir)

train_Dog_dir = os.path.join(train_dir, 'Dog')
#os.mkdir(train_Dog_dir)

validation_Cat_dir = os.path.join(validation_dir, 'Cat')
#os.mkdir(validation_Cat_dir)

validation_Dog_dir = os.path.join(validation_dir, 'Dog')
#os.mkdir(validation_Dog_dir)

test_Cat_dir = os.path.join(test_dir, 'Cat')
#os.mkdir(test_Cat_dir)

test_Dog_dir = os.path.join(test_dir, 'Dog')
#os.mkdir(test_Dog_dir)

fnames = ['{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir + '/Cat', fname)
    dst = os.path.join(train_Cat_dir, fname)
    shutil.copyfile(src, dst)

# copy the following 500 cats images to the folder train_dir
fnames = ['{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir + '/Cat', fname)
    dst = os.path.join(validation_Cat_dir, fname)
    shutil.copyfile(src, dst)

# copy the following 500 cats images to the folder test_dir
fnames = ['{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir + '/Cat', fname)
    dst = os.path.join(test_Cat_dir, fname)
    shutil.copyfile(src, dst)

# copy the first 1000 dogs images to the folder train_dir
fnames = ['{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir + '/Dog', fname)
    dst = os.path.join(train_Dog_dir, fname)
    shutil.copyfile(src, dst)

# copy the following 500 dogs images to the folder train_dir
fnames = ['{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir + '/Dog', fname)
    dst = os.path.join(validation_Dog_dir, fname)
    shutil.copyfile(src, dst)

# copy the following 500 dogs images to the folder test_dir
fnames = ['{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir + '/Dog', fname)
    dst = os.path.join(test_Dog_dir, fname)
    shutil.copyfile(src, dst)


print("total training cat images :", len(os.listdir(train_Cat_dir)))
print("total training dog images :", len(os.listdir(train_Dog_dir)))
print("total validation cat images :", len(os.listdir(validation_Cat_dir)))
print("total validation dog images :", len(os.listdir(validation_Dog_dir)))
print("total test cat images :", len(os.listdir(test_Cat_dir)))
print("total test dog images :", len(os.listdir(test_Dog_dir)))

img = cv2.imread(train_Cat_dir + '/' + os.listdir(train_Cat_dir)[0])
img.shape

shutil.copyfile(train_Cat_dir + '/665.jpg', train_Cat_dir + '/666.jpg')

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                        train_dir,
                        target_size=(150,150),
                        batch_size=20,
                        class_mode='binary'
                        )

validation_generator = test_datagen.flow_from_directory(
                        validation_dir,
                        target_size=(150,150),
                        batch_size=20,
                        class_mode='binary'
                        )



model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu',
                       input_shape=(150,150, 3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',  # optimizers.RMSprop(lr=1e-4) ??
             loss='binary_crossentropy',
             metrics=['accuracy'])


model.summary()


history = model.fit_generator(
            train_generator,
            steps_per_epoch=100,
            epochs=30,
            validation_data=validation_generator,
            validation_steps=50
)


model.save('classifying_cats_dogs_small_1.h5')


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='training')
plt.plot(epochs, val_acc, 'b', label='validation')
plt.title('Accuracy during Training and Validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='training')
plt.plot(epochs, val_loss, 'b', label='validation')
plt.title('Loss during Training and Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()