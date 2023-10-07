import os
cwd = os.getcwd()
dir_path = os.path.dirname(os.path.realpath(__file__))
runfile('src/data/make_dataset.py')

tf.random.set_seed(42)

preprocess_input = tfl.Rescaling(scale=1/255, offset=0.0, name='preprocess')

class BottleneckBlock(tf.keras.layers.Layer):
    """
    Bottleneck block from MobileNetV2
    
    Constructor arguments:
    f -- integer, shape of the convolutional window
    nb_depth -- number of filters to expand to for the depthwise convolution
    nb_project -- number of filters for the projection
    
    """
    def __init__(self, f, nb_depth, nb_proj, **kwargs):
        super(BottleneckBlock, self).__init__(**kwargs)
        self.expconv = tfl.Conv2D(filters = nb_depth, 
                                  kernel_size = (1,1), 
                                  strides = (1,1), 
                                  padding = 'same', 
                                  kernel_initializer = glorot_uniform(seed=42),
                                  name = 'expconv'
                                  )
        self.expbatch = tfl.BatchNormalization(axis = 3, name = 'expbatch')
        self.depthconv = tfl.DepthwiseConv2D(kernel_size = (f,f), 
                                             strides = (1,1), 
                                             padding = 'same', 
                                             kernel_initializer = glorot_uniform(seed=42),
                                             name = 'depthconv'
                                             )
        self.depthbatch = tfl.BatchNormalization(axis = 3, name = 'depthbatch')
        self.projconv = tfl.Conv2D(filters = nb_proj, 
                                   kernel_size = (1,1), 
                                   strides = (1,1), 
                                   padding = 'same', 
                                   kernel_initializer = glorot_uniform(seed=42),
                                   name = 'projconv'
                                   )
        self.projbatch = tfl.BatchNormalization(axis = 3, name = 'projbatch')
        self.shortbatch = tfl.BatchNormalization(axis = 3, name = 'shortbatch')
        self.nb_depth = nb_depth
        self.nb_proj = nb_proj
        self.f = f
    
    def call(self, X):
        X_shortcut = X # Saved for the shortcut connection
    
        # Expansion Part
        X = self.expconv(X)
        X = self.expbatch(X)
        X = tfl.Activation('relu')(X)
        
        # Depthwise Convolution Part
        X = self.depthconv(X)
        X = self.depthbatch(X)
        X = tfl.Activation('relu')(X)
        
        # Projection Part
        X = self.projconv(X)
        X = self.projbatch(X)
        
        # Shortcut Connection
        if self.nb_proj == X_shortcut.shape[3]:
            X_shortcut = self.shortbatch(X_shortcut)
            X = tfl.add([X, X_shortcut])
        X = tfl.Activation('relu')(X)
    
        return X
    
    def get_config(self):
        config = super(BottleneckBlock, self).get_config()
        config.update({"f": self.f, "nb_depth": self.nb_depth, "nb_proj": self.nb_proj})
        return config


def flip_model(input_shape=(224,224), f=3, data_augmentation=tf.identity):
    """
    A custom architecture for the binary classification of flip/notflip images

    Arguments:
    input_shape -- shape of the images of the dataset
    f -- integer, shape of the convolutional window
    
    """

    X_input = tf.keras.Input(shape=input_shape, name = 'input')
    
    X = data_augmentation(X_input)
    
    X = preprocess_input(X)
    
    X = tfl.Conv2D(6, 
                   (3, 3), 
                   strides = (2, 2), 
                   padding = 'same', 
                   kernel_initializer = glorot_uniform(seed=42),
                   name = 'Conv2D_1'
                   )(X)
    X = tfl.BatchNormalization(axis = 3, name = 'BatchNorm_1')(X)
    X = tfl.Activation('relu', name = 'ReLU_1')(X)
    X = tfl.MaxPooling2D(pool_size = (2, 2), 
                         strides = (2, 2),
                         name = 'MaxPool2D_1'
                         )(X)
    X = tfl.Dropout(0.01, 
                    seed = 42,
                    name = 'Dropout_1'
                    )(X)
    
    Block1 = BottleneckBlock(f, 
                             24, 
                             6, 
                             name='Bottleneck1'
                             )
    X = Block1(X)
    X = tfl.MaxPooling2D(pool_size = (2, 2),
                         name = 'MaxPool2D_2')(X)
    X = tfl.Dropout(0.01, 
                    seed=42,
                    name = 'Dropout_2'
                    )(X)
    
    Block2 = BottleneckBlock(f, 
                             48, 
                             6, 
                             name='Bottleneck2'
                             )
    X = Block2(X)
    X = tfl.MaxPooling2D(pool_size = (2, 2),
                         name = 'MaxPool2D_3')(X)
    X = tfl.Dropout(0.1, 
                    seed = 42,
                    name = 'Dropout_3'
                    )(X)
    
    X = tfl.Flatten(name = 'Flatten')(X)
    X_output = tfl.Dense(units = 1, 
                         activation = 'sigmoid', 
                         kernel_initializer = glorot_uniform(seed=42),
                         name = 'Dense'
                         )(X)
    
    model = tf.keras.Model(inputs = X_input,
                           outputs = X_output,
                           name = 'flip_model'
                           )
    
    return model
    

# IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)
f = 3 # kernel size

# Creating the model instance
augment_data = tf.identity
custom_model = flip_model(input_shape=IMG_SHAPE, f=f, data_augmentation=augment_data)
custom_model.trainable = True
# print(f'\nThe custom model has {len(custom_model.layers)} layers.')
print('\n A summary of the model follows:\n')
summary = custom_model.summary()

# Compiling the model
loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=False)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.001,
                                                             decay_steps = len(train_dataset), 
                                                             decay_rate = 0.97, 
                                                             staircase = True
                                                             )
optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
metrics = ['accuracy', f1_score]
 
custom_model.compile(loss = loss_function,
                     optimizer = optimizer,
                     metrics = metrics
                     )

# Training the model
loss_callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                                 min_delta = 0.0002,
                                                 patience = 10
                                                 )

acc_callback = tf.keras.callbacks.EarlyStopping(monitor ='val_accuracy', 
                                                min_delta = 0.001, 
                                                patience = 30
                                                )

checkpoints = tf.keras.callbacks.ModelCheckpoint('./checkpoints/custom_model',
                                                monitor = 'val_loss',
                                                save_best_only = True, 
                                                save_weights_only = True,
                                                save_freq = 'epoch'
                                                )

history = custom_model.fit(train_dataset,
                           epochs = 100,
                           validation_data = validation_dataset,
                           callbacks = [loss_callback, acc_callback, checkpoints]
                           )

# Saving the model and its weights
custom_model.load_weights('./checkpoints/custom_model') # weights from best epoch
custom_objects = {"BottleneckBlock": BottleneckBlock, "f1_score": f1_score}
# tf.keras.saving.register_keras_serializable(package="my_package", name="f1_score")
custom_model.save('./checkpoints/custom_model.h5')

# Plotting Loss and f1 score history
loss = history.history['loss']
val_loss = history.history['val_loss']
f1 = history.history['f1_score']
val_f1 = history.history['val_f1_score']

plt.figure()
plt.subplot(2,1,1)
plt.plot(loss, label = 'Training loss')
plt.plot(val_loss, label = 'Validation loss')
plt.legend(loc = 'upper right')
plt.title('Training and Validation loss')
plt.subplot(2,1,2)
plt.plot(f1, label = 'Training f1_score')
plt.plot(val_f1, label = 'Validation f1_score')
plt.ylim([0, 1])
plt.legend(loc = 'lower right')
plt.title('Training and Validation f1_score')
plt.show()

# Evaluate on the test set
print('\nEvaluating the model on the test set\n')
custom_model.evaluate(test_dataset)
test_predictions = custom_model.predict(test_dataset)

print('custom_network_train.py was run')




