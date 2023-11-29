import os
cwd = os.getcwd()
dir_path = os.path.dirname(os.path.realpath(__file__))
runfile('src/data/make_dataset.py')

tf.random.set_seed(42)

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# First attempt: Only fine tune the top of the base model
def flip_model(image_shape=IMG_SIZE, data_augmentation=tf.identity):
    ''' Define a keras model for binary classification making use of MobileNetV2
    Arguments:
        image_shape -- Image width and height
        data_augmentation -- data augmentation function
    Returns:
        Keras model instance
    '''
    input_shape = image_shape + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
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

mobilenetv2_model = flip_model()

# Compile the model
mobilenetv2_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005),
                          loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
                          metrics = ['accuracy', f1_score])

# Train the model
initial_epochs = 5
history = mobilenetv2_model.fit(train_dataset, 
                                validation_data=validation_dataset, 
                                epochs=initial_epochs)

# Plot the training history
score = history.history['f1_score']
val_score = history.history['val_f1_score']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))

plt.subplot(2, 1, 1)
plt.plot(score, label='Training f1_score')
plt.plot(val_score, label='Validation f1_score')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.title('Training and Validation f1_score')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')

plt.show()

## Second attempt: Fine tune the last few dozens of layers
base_model = mobilenetv2_model.layers[4]
base_model.trainable = True
print(f'Number of layers in the base model: {len(base_model.layers)}')

# Fine-tune from this layer onwards
fine_tune_at = 120
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Compile the model
loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.0001,
                                                             decay_steps = len(train_dataset), 
                                                             decay_rate = 0.8, 
                                                             staircase = False
                                                             )
optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
metrics = ['accuracy', f1_score]

mobilenetv2_model.compile(loss=loss_function,
                          optimizer = optimizer,
                          metrics=metrics)

# Train the model
fine_tune_epochs = 5
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = mobilenetv2_model.fit(train_dataset,
                                     epochs=total_epochs,
                                     initial_epoch=history.epoch[-1],
                                     validation_data=validation_dataset)

mobilenetv2_model.save_weights('./checkpoints/MobileNetV2')

# Plot the fine-tuning
score += history_fine.history['f1_score']
val_score += history_fine.history['val_f1_score']
loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))

plt.subplot(2, 1, 1)
plt.plot(score, label='Training f1_score')
plt.plot(val_score, label='Validation f1_score')
plt.ylim([0, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation f1_score')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')

plt.show()

# Evaluate on the test set
print('\nEvaluating the model on the test set\n')
mobilenetv2_model.evaluate(test_dataset)
test_predictions = mobilenetv2_model.predict(test_dataset)

print('mobile_net_train.py was run')
