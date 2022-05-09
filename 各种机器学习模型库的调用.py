#线性回归
from sklearn import linear_model
model = linear_model.LinearRegression()
#逻辑回归【softmax:1/(1+e^-x)】
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
#决策树
from sklearn import tree
model = tree.DecisionTreeClassifier(criterion='gini') 
#随机森林
from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier()
#

#
model.fit(Xtrain, ytrain)
model.score(Xtrain, ytrain)
predicted= model.predict(x_test)
