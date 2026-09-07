import numpy as np, os
d = 'data/processed'
print(len(os.listdir(d)), "files")
print(np.load(f'{d}/X_train.npy').shape)
print(np.unique(np.load(f'{d}/y_train_binary.npy')))
print(np.unique(np.load(f'{d}/y_train_multi.npy')))