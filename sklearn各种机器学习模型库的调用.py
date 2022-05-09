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
#线性svm
from sklearn.svm import SVC
model = SVC(kernel='linear', C=1.0, random_state=0) # 用线性核，你也可以通过kernel参数指定其它的核。
#非线性svm
model = SVC(kernel='rbf', random_state=0, gamma=x, C=1.0) # 令gamma参数中的x分别等于0.2和100.0
#朴素贝叶斯
'''朴素贝叶斯根据特征是否离散，分为三种模型，如下所示：
贝叶斯估计/多项式模型：当特征是离散的时候，使用该模型；

高斯模型：特征是连续的时候采用该模型；

伯努利模型：特征是离散的，且取值只能是 0 和 1。'''
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB()
#KNN
from sklearn.neighbors import KNeighborsClassifier
KNeighborsClassifier(n_neighbors=6)
#boosting(AdaBoost)
#bagging
#GBDT
#
model.fit(Xtrain, ytrain)
model.score(Xtrain, ytrain)
predicted= model.predict(x_test)
