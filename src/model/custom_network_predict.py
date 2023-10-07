import os
cwd = os.getcwd()
dir_path = os.path.dirname(os.path.realpath(__file__))
runfile('src/data/make_dataset.py')

# Define custom object BottleneckBlock before loading the model
class BottleneckBlock(tf.keras.layers.Layer):
    """
    Bottleneck block from MobileNetV2
    
    Arguments:
    f -- shape of the convolutional window
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

# Load the model with its custom objects
custom_objects = {"BottleneckBlock": BottleneckBlock, "f1_score": f1_score}
loaded_model = tf.keras.models.load_model('./checkpoints/custom_model.h5', custom_objects=custom_objects)

# Evaluate on the test set
print('\nEvaluating the model on the test set\n')
loaded_model.evaluate(test_dataset)
test_predictions = loaded_model.predict(test_dataset)

print('custom_network_predict.py was run')