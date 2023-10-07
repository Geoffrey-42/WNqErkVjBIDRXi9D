runfile('setup.py')
import os
cwd = os.getcwd()
dir_path = os.path.dirname(os.path.realpath(__file__))
directory = cwd.replace("\\", "/")

BATCH_SIZE = 32
IMG_SIZE = (224, 224)

# It is assumed that the training and testing directories contain the same 
# distribution of data. 
# Hence the validation dataset is sampled from the training set.
print('Exploring the training dataset...')
train_dataset, validation_dataset = image_dataset_from_directory(directory+"/images/training/",
                                                                 shuffle=True,
                                                                 validation_split = 0.2,
                                                                 subset = 'both',
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE,
                                                                 seed=42)
print('Exploring the testing dataset...')
test_dataset = image_dataset_from_directory(directory+"/images/testing/",
                                            shuffle=False,
                                            batch_size=597,
                                            image_size=IMG_SIZE,
                                            seed=42)

class_names = train_dataset.class_names
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

# Display 9 training examples
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

print('make_dataset.py was run')