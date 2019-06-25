# -*- coding: utf-8 -*-
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator


# 数据准备
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,# ((x/255)-0.5)*2  归一化到±1之间
    rotation_range=30, # 随机旋转的度数范围
    width_shift_range=0.2, #  除以总宽度的值
    height_shift_range=0.2, #  除以总宽度的值
    shear_range=0.2, # 剪切强度（以弧度逆时针方向剪切角度）
    zoom_range=0.2, # 随机缩放范围
    horizontal_flip=True, # 随机水平翻转
)
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

train_generator = train_datagen.flow_from_directory(directory='data/train',
                                                    target_size=(299,299),#Inception V3规定大小
                                                    batch_size=64,
                                                    classes = ['cat', 'dog'],
                                                    class_mode = 'binary')
val_generator = val_datagen.flow_from_directory(directory='data/val',
                                                target_size=(299,299),
                                                batch_size=64,
                                                classes = ['cat', 'dog'],
                                                class_mode = 'binary')
if __name__ == '__main__':
    # 构建不带分类器的预训练模型
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # 添加全局平均池化层
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # 添加一个全连接层
    x = Dense(1024, activation='relu')(x)

    # 添加一个分类器，假设我们有200个类
    predictions = Dense(1, activation='sigmoid')(x)

    # 构建我们需要训练的完整模型
    model = Model(inputs=base_model.input, outputs=predictions)

    print(model.summary())
    # 首先，我们只训练顶部的几层（随机初始化的层）
    # 锁住所有 InceptionV3 的卷积层
    for layer in base_model.layers:
        layer.trainable = False

    # 编译模型（一定要在锁层以后操作）
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    # 在新的数据集上训练几代
    model.fit_generator(generator=train_generator,
                    steps_per_epoch=10, # 800
                    epochs=2, # 2
                    validation_data=val_generator,
                    validation_steps=12, # 12
                    class_weight='auto'
                   )
    # class_weight = 'auto'

    # 现在顶层应该训练好了，让我们开始微调 Inception V3 的卷积层。
    # 我们会锁住底下的几层，然后训练其余的顶层。

    # 让我们看看每一层的名字和层号，看看我们应该锁多少层呢：
    for i, layer in enumerate(base_model.layers):
       print(i, layer.name)

    # 我们选择训练最上面的两个 Inception block
    # 也就是说锁住前面249层，然后放开之后的层。
    for layer in model.layers[:249]:
       layer.trainable = False
    for layer in model.layers[249:]:
       layer.trainable = True

    # 我们需要重新编译模型，才能使上面的修改生效
    # 让我们设置一个很低的学习率，使用 SGD 来微调
    from keras.optimizers import SGD
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])

    # 我们继续训练模型，这次我们训练最后两个 Inception block
    # 和两个全连接层
    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=10, # 800
                                  epochs=2, # 2
                                  validation_data=val_generator,
                                  validation_steps=12,
                                  class_weight='auto'
                                  )
    # class_weight = 'auto'
    # 精度
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    # 损失
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    # epochs的数量
    epochs = range(len(acc))
    print(acc)
    print(loss)
    print(val_acc)
    print(val_loss)
