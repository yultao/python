#from sklearn.feature_extraction import DictVectorizer
from sklearn import neighbors
from sklearn import datasets

knn = neighbors.KNeighborsClassifier()

iris = datasets.load_iris()
# digits = datasets.load_digits()
print(iris.data)
# print(iris)

 
knn.fit(iris.data, iris.target)

label = knn.predict([[0.1,0.2,0.3,0.4]])

print(label)