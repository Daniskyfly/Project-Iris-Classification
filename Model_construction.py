import pandas
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

file_iris = pandas.read_csv(r'C:\Users\Daniel\Desktop\Git Projects\Project-Iris-Classification\Dataset\iris.csv')

# define features
iris_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = file_iris[iris_features]
#print(X)

#Target variable
y = file_iris.species

# separate test from training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

#print(X_test.shape)
#print(X_train.shape)
#print(y_test.shape)
#print(y_train.shape)

min_error = len(y_test)
n = 0
for i in range(1, 100, 1): #test which n_neighbors parameter works better
    #print(i)
    # Create KNN classifier
    knn = KNeighborsClassifier(n_neighbors = i) # each i value is tested
    # Fit the classifier to the data
    knn.fit(X_train,y_train)
    error = False
    sum_error = 0
    for x in range(len(y_test)): # this loops counts the errors in the test data

        if knn.predict(X_test)[x] == y_test.iloc[x]:
            error = False
        else:
            error = True
        sum_error += error # error count
    #print(sum_error)
    if sum_error < min_error:  # if the error count is smaller for a given i, we select the i as the n parameter
        min_error = sum_error
        n = i
#print(min_error, n)
print("For this data, the knn model is optimal with n_neighbors = " + str(n) + ", where it generates " +
      str(min_error) + " errors for the used test data" )
print()
print()

# apply the model for the calculated n_neighbors
knn = KNeighborsClassifier(n_neighbors = n) # each i value is tested
knn.fit(X_train,y_train)
print(y_test.iloc[0:5])
print(knn.predict(X_test)[0:5])

# Codify target variable (species)
#file_iris = pandas.get_dummies(file_iris, columns=['species'])
#print(file_iris.head())


