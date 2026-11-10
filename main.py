""" 
Decision Tree algorithm






"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("car.csv")

data.columns = ("sales","maintainence","doors","persons","boot_space","safety","class")
print(data.head())
print(data.info())

# label encoder  to convert string in to numbers 
from sklearn.preprocessing import LabelEncoder 

encoder = LabelEncoder()

data["sales"] = encoder.fit_transform(data["sales"])
data["maintainence"] = encoder.fit_transform(data["maintainence"])
data["doors"] = encoder.fit_transform(data["doors"])
data["persons"] = encoder.fit_transform(data["persons"])
data["boot_space"] = encoder.fit_transform(data["boot_space"])
data["safety"] = encoder.fit_transform(data["safety"])
data["class"] = encoder.fit_transform(data["class"])

print(data.info())

x = data[["sales","maintainence","doors","persons","boot_space","safety"]]
y = data["class"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=2)


from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion="entropy",random_state=0)

classifier.fit(x_train,y_train)
prediction = classifier.predict(x_test)


from sklearn.metrics import accuracy_score,confusion_matrix

accuracy = accuracy_score(y_test,prediction)
accuracy =round(accuracy*100,2)

print("the accuracy is ",accuracy,"% acurate")

matrix = confusion_matrix(y_test,prediction)

#annot = True  =  annotation    fmt  formate = the format of 0and1
sns.heatmap(matrix,annot= True,fmt="d")

plt.title("matrix")
plt.xlabel("the predicted amount")
plt.ylabel("actual amount")
plt.show()
