import pandas
import matplotlib.pyplot as plt

#Python Built-in function
#file_iris = open(r'C:\Users\Daniel\Desktop\Git Projects\Project-Iris-Classification\Dataset\iris.csv', 'r')
#print(file_iris.read())

#Pandas function
file_iris = pandas.read_csv(r'C:\Users\Daniel\Desktop\Git Projects\Project-Iris-Classification\Dataset\iris.csv')
print(file_iris.head(4))

#gerenal data description
print(file_iris.describe())

#list of existing columns
print(file_iris.columns)

#existing plant species#
print(file_iris.species.unique())

#group by species
file_virginica = file_iris.loc[file_iris['species']=='virginica']
print(file_virginica.head())

#Scatter plots

plt.figure(1)
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.scatter(file_iris.sepal_length, file_iris.sepal_width)

plt.figure(2)
plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.scatter(file_iris.petal_length, file_iris.petal_width)

plt.figure(3)
plt.xlabel('petal_length')
plt.ylabel('sepal_length')
plt.scatter(file_iris.petal_length, file_iris.sepal_length)

plt.figure(4)
plt.xlabel('petal_width')
plt.ylabel('sepal_width')
plt.scatter(file_iris.petal_width, file_iris.sepal_width)



plt.show()