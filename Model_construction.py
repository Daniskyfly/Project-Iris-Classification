import pandas

file_iris = pandas.read_csv(r'C:\Users\Daniel\Desktop\Git Projects\Project-Iris-Classification\Dataset\iris.csv')

# define features
iris_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = file_iris[iris_features]
print(X)

# Define data for test. First 5 rows from every species
test_data_setosa = file_iris.loc[file_iris['species']=='setosa'].head(5)
test_data_versicolor = file_iris.loc[file_iris['species']=='versicolor'].head(5)
test_data_virginica = file_iris.loc[file_iris['species']=='virginica'].head(5)

# Define data for training. All except for the rows used for test
training_data = pandas_concat([file_iris, test_data_setosa, test_data_versicolor, test_data_virginica]).drop_duplicates(keep=False)

