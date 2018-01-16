from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO
import csv

data = []
label = []
featurelist = []
filename = "E:\\pycharm\\machine_learning\\Titanic\\train.csv"
with open(filename,'r') as file:
    titanic = csv.reader(file)
    for row in titanic:
        data.append(row)
        label.append(row[1])         #储存survived标签  带表头
label = label[1:]                   #去除表头
for item in data[1:]:
    head_dic = {}
    head_dic[data[0][2]]=item[2]    #Pclass
    head_dic[data[0][4]]=item[4]    #Sex
    #age为连续变量  最好阈值化
    if item[5]!='':                 #age 存在缺省值
        try:
            if int(item[5])<=20:
                head_dic[data[0][5]]='0'    #小于20岁
            elif int(item[5])<=40:
                head_dic[data[0][5]]='1'    #大于20岁 小于40岁
            else:
                head_dic[data[0][5]]='2'    #大于40岁
        except:
            age = item[5].split('.')        #age存在小数的情况
            if int(age[0])<=20:
                head_dic[data[0][5]]='0'
            elif int(age[0])<=40:
                head_dic[data[0][5]]='1'
            else:
                head_dic[data[0][5]]='2'
    else:                                   # age为缺省
        head_dic[data[0][5]] = '1'
    featurelist.append(head_dic)            # 每一个样例都存入列表
vec = DictVectorizer()                      # 转化为one-hot
train = vec.fit_transform(featurelist).toarray()
print(vec.get_feature_names())
lb = preprocessing.LabelBinarizer()         #标签二值化
label = lb.fit_transform(label)
decision_tree = tree.DecisionTreeClassifier(criterion='entropy') #构建决策树 ID3
decision_tree.fit(train,label)              # 进行分类
# 决策树可视化
with open('E:\\pycharm\\machine_learning\\Titanic\\tree.dot','w') as output:
    output = tree.export_graphviz(decision_tree,feature_names=vec.get_feature_names(),out_file=output)
# 将测试集进行转化
filename = "E:\\pycharm\\machine_learning\\Titanic\\test.csv"
test_data = []
test_featurelist=[]
with open(filename,'r') as file:
    titanic = csv.reader(file)
    for row in titanic:
        test_data.append(row)

for item in test_data[1:]:
    test_head_dic = {}
    test_head_dic[test_data[0][1]]=item[1]
    test_head_dic[test_data[0][3]]=item[3]
    if item[4]!='':
        try:
            if int(item[4])<=20:
                test_head_dic[test_data[0][4]]='0'
            elif int(item[4])<=40:
                test_head_dic[test_data[0][4]]='1'
            else:
                test_head_dic[test_data[0][4]]='2'
        except:
            age = item[4].split('.')
            if int(age[0])<=20:
                test_head_dic[test_data[0][4]]='0'
            elif int(age[0])<=40:
                test_head_dic[test_data[0][4]]='1'
            else:
                test_head_dic[test_data[0][4]]='2'
    else:
        test_head_dic[test_data[0][4]] = '1'
    test_featurelist.append(test_head_dic)
vec = DictVectorizer()
test = vec.fit_transform(test_featurelist).toarray()
predict = decision_tree.predict(test)                   #输出测试结果
print(predict)
#将测试结果输出csv
submission_dir = "E:\\pycharm\\machine_learning\\Titanic\\gender_submission.csv"
with open(submission_dir,'w',newline='') as file:
    submission = csv.writer(file)
    submission.writerow(['PassengerId','Survived'])
    for i in range(len(predict)):
        submission.writerow([892+i,predict[i]])



