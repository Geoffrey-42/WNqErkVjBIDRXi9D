import os
cwd = os.getcwd()
dir_path = os.path.dirname(os.path.realpath(__file__))
runfile('src/data/make_dataset.py')

preprocess_input = tf.keras.applications.resnet50.preprocess_input

## Now adapt the base model
def flip_model(image_shape=IMG_SIZE, data_augmentation=tf.identity):
    ''' Define a keras model for binary classification making use of ResNet50
    Arguments:
        image_shape -- Image width and height
        data_augmentation -- data augmentation function
    Returns:
        Keras model instance
    '''
    input_shape = image_shape + (3,)
    base_model = tf.keras.applications.ResNet50(input_shape=input_shape,
                                                include_top=False,
                                                weights='imagenet')
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=input_shape) 
    x = data_augmentation(inputs)
    x = preprocess_input(x) 
    x = base_model(x, training=False) 
    x = tfl.GlobalAveragePooling2D()(x) 
    x = tfl.Dropout(0.2)(x)
    outputs = tfl.Dense(1, activation='linear')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    return model

loaded_model = flip_model()
loaded_model.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
                     metrics = ['accuracy', f1_score])
loaded_model.load_weights('./checkpoints/ResNet50')
print('\nEvaluating the model on the test set\n')
loaded_model.evaluate(test_dataset)
test_predictions = loaded_model.predict(test_dataset)

print('resnet_predict.py was run')
