import numpy as np
from keras.layers import Input, Dense

pair=("actress", "actor") 
w1, w2 = pair
print(w1)
food = {"ham" : "yes", "egg" : "yes", "spam" : "no" }
a,b=food["ham"],food["spam"]


print(a)
print(b)

print(np.array(shape=(2), dtype='int32').shape)