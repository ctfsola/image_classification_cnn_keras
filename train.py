
import numpy as np
import matplotlib.pyplot as plt
from vectorize import load_data
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras import layers

# from tensorflow.keras import layers
# from tensorflow.keras import Model
train_images,train_labels,test_images,test_labels,classes = load_data()

print(len(train_images))
print(train_images.shape)
print(len(train_labels))
print(len(test_images))
print(len(test_labels))

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(classes[train_labels[i]])
# plt.show()

num_classes = len(set(classes))
print(num_classes)

height = 150
width = 150
channels = 3
batch_size = 128


from keras.layers import Conv2D , Activation, Flatten, Dense,Dropout

img_input = layers.Input(shape=(height, width, channels))

x = Conv2D(16, (3, 3), strides=(1, 1), padding='same', name='conv1')(img_input)
x = layers.MaxPooling2D(2)(x)
x = Activation('relu')(x)

x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv2')(x)
x = layers.MaxPooling2D(2)(x)
x = Activation('relu')(x)

x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv3')(x)
x = layers.MaxPooling2D(2)(x)
x = Activation('relu')(x)


x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv4')(x)
x = layers.MaxPooling2D(2)(x)
x = Activation('relu')(x)

x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv5')(x)
x = layers.MaxPooling2D(2)(x)
x = Activation('relu')(x)




x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dense(num_classes, activation='softmax')(x)

model = Model(img_input, x)





# img_input = layers.Input(shape=(height, width, channels))

# x = layers.Conv2D(16, 3, activation='relu')(img_input)
# x = layers.MaxPooling2D(2)(x)

# x = layers.Conv2D(32, 3, activation='relu')(x)
# x = layers.MaxPooling2D(2)(x)

# x = layers.Conv2D(64, 3, activation='relu')(x)
# x = layers.MaxPooling2D(2)(x)



# x = layers.Conv2D(64, 3, activation='relu')(x)
# x = layers.MaxPooling2D(2)(x)

# x = layers.Conv2D(128, 3, activation='relu')(x)
# x = layers.MaxPooling2D(2)(x)



# x = layers.Flatten()(x)
# x = layers.Dropout(0.5)(x)
# x = layers.Dense(512, activation='relu')(x)

# output = layers.Dense(num_classes, activation='softmax')(x)
# model = Model(img_input, output)




# model = Sequential()
# model.add(layers.SeparableConv2D(32, 3,
#                                  activation='relu',
#                                  input_shape=(height, width, channels,)))
# model.add(layers.SeparableConv2D(64, 3, activation='relu'))
# model.add(layers.MaxPooling2D(2))

# model.add(layers.SeparableConv2D(64, 3, activation='relu'))
# model.add(layers.SeparableConv2D(128, 3, activation='relu'))
# model.add(layers.MaxPooling2D(2))

# model.add(layers.SeparableConv2D(64, 3, activation='relu'))
# model.add(layers.SeparableConv2D(128, 3, activation='relu'))
# model.add(layers.GlobalAveragePooling2D())

# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dense(num_classes, activation='softmax'))


#model.compile(optimizer='rmsprop', loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) ,metrics=['acc']) # 整数标签要用 sparse_categorical_crossentropy
#model.compile(optimizer='rmsprop', loss= 'sparse_categorical_crossentropy' ,metrics=['acc']) # 整数标签要用 sparse_categorical_crossentropy

#model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc']) 
model.compile(optimizer= 'adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['acc']) # 增加from_logits=True，数据会更稳定
model.summary()
history = model.fit(train_images, train_labels,
                    epochs=20,
                    batch_size=batch_size,
                    validation_split=0.2,
                    )	


test_loss, test_acc = model.evaluate(test_images,test_labels)
print('test_acc',test_acc)

model.save_weights('model_weights.h5')
model.save('model.h5')

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


#绘制训练精度和验证精度
plt.clf()  #清空图像

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


#预测

probability_model = Sequential([model, layers.Softmax()])
predictions = probability_model.predict(test_images)

#在上例中，模型预测了测试集中每个图像的标签。我们来看看第一个预测结果：
plt.figure(figsize=(10,10))
# plt.subplot(1,1,1)
# plt.xticks([])
# plt.yticks([])
#plt.grid(False)
plt.imshow(test_images[0], cmap=plt.cm.binary)
plt.xlabel(classes[test_labels[0]])
plt.show()

pre = np.argmax(predictions[0])
print(pre)