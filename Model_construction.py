import pandas
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

file_iris = pandas.read_csv(r'C:\Users\Daniel\Desktop\Git Projects\Project-Iris-Classification\Dataset\iris.csv')

# define features
iris_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = file_iris[iris_features]
print(X)

#Target variable
y = file_iris.species

# separate test from training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

print(X_test.shape)
print(X_train.shape)
print(y_test.shape)
print(y_train.shape)


# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 12) # seems like the heigher the number of neighbours, the more precise.
# Fit the classifier to the data
knn.fit(X_train,y_train)

#show first 5 model predictions on the test data
print(y_test[0:5])
print(knn.predict(X_test)[0:5])

# Codify target variable (species)
#file_iris = pandas.get_dummies(file_iris, columns=['species'])
#print(file_iris.head())


