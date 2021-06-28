import numpy as np
import matplotlib.pyplot as plt
import idx2numpy


x_train     = idx2numpy.convert_from_file('samples/train-images.idx3-ubyte')
x_labels    = idx2numpy.convert_from_file('samples/train-labels.idx1-ubyte')
test        = idx2numpy.convert_from_file('samples/t10k-images.idx3-ubyte')
test_labels = idx2numpy.convert_from_file('samples/t10k-labels.idx1-ubyte')
















idx = 100
print(x_labels[idx])

plt.imshow(np.asarray(x_train[idx]))
plt.show()

