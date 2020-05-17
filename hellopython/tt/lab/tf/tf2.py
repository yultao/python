#Note that when using from package import item,
# the item can be either a submodule (or subpackage) of the package, or some other name defined in the package,
# like a function, class or variable.

# import module as identifier
import tensorflow as tf
# from module import submodule
from tensorflow import keras


print(tf.version);
print("tf.__version__: "+ tf.__version__)
print("keras.__version__: "+keras.__version__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt