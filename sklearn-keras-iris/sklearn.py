from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris

# 定义一个保存模型的字典，根据 key 来选择加载哪个模型
models={ "knn": KNeighborsClassifier(n_neighbors=1),
    "naive_bayes": GaussianNB(),
    "logit": LogisticRegression(solver="lbfgs", multi_class="auto"),
    "svm": SVC(kernel="rbf", gamma="auto"),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "mlp": MLPClassifier()}

# 载入 Iris 数据集，然后进行训练集和测试集的划分，75%数据作为训练集，其余25%作为测试集
print("[INFO] loading data...")
dataset = load_iris()
#print(dataset)

(trainX, testX, trainY, testY) = train_test_split(dataset.data, dataset.target, random_state=3, test_size=0.25)
'''
X_train, X_test, y_train, y_test =train_test_split(train_data,train_target,test_size=0.4, random_state=0)
参数解释：
train_data：所要划分的样本特征集
train_target：所要划分的样本结果
test_size：样本占比，如果是整数的话就是样本的数量
random_state：是随机数的种子
'''
# 训练模型
print("[INFO] using '{}' model".format("knn"))
model = models["knn"]
model.fit(trainX, trainY)

# 预测并输出一份分类结果报告
print("[INFO] evaluating")
predictions = model.predict(testX)
print(classification_report(testY, predictions))
#classification_report参考:https://blog.csdn.net/qq_40765537/article/details/105344798 and https://blog.csdn.net/comway_Li/article/details/102758972
