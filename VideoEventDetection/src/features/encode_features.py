import os
cwd = os.getcwd()
dir_path = os.path.dirname(os.path.realpath(__file__))
runfile('src/data/make_dataset.py')

def augment_data(X_input):
    X = tfl.RandomBrightness(factor = 0.2, seed = 42)(X_input, training = True)
    X = tfl.RandomContrast(factor = 0.2, seed = 42)(X, training = True)
    return X

print('encode_features.py was run')
