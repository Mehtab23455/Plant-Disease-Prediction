from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, ReLU, Concatenate, GlobalAveragePooling2D, Dense

def build_densenet(input_shape=(224, 224, 3), num_classes=3):
    inputs = Input(shape=input_shape, name="input_layer_1")
    
    # First conv + pooling
    x = Conv2D(64, (3,3), padding='same', name='conv2d_4')(inputs)
    x = MaxPooling2D((2,2), name='max_pooling2d_1')(x)
    x = BatchNormalization(name='batch_normalization_1')(x)
    x = ReLU(name='re_lu_3')(x)
    
    # Dense block 1
    x1 = Conv2D(32, (3,3), padding='same', name='conv2d_5')(x)
    x = Concatenate(name='concatenate_3')([x, x1])
    x = BatchNormalization(name='batch_normalization_2')(x)
    x = ReLU(name='re_lu_4')(x)
    
    # Dense block 2
    x2 = Conv2D(32, (3,3), padding='same', name='conv2d_6')(x)
    x = Concatenate(name='concatenate_4')([x, x2])
    x = BatchNormalization(name='batch_normalization_3')(x)
    x = ReLU(name='re_lu_5')(x)
    
    # Dense block 3
    x3 = Conv2D(32, (3,3), padding='same', name='conv2d_7')(x)
    x = Concatenate(name='concatenate_5')([x, x3])
    
    # Classification head
    x = GlobalAveragePooling2D(name='global_average_pooling2d_1')(x)
    outputs = Dense(num_classes, activation='softmax', name='dense_1')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="DenseNet_Custom")
    return model
