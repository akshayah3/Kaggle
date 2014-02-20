__author__ = 'Akshay'


from sklearn import svm
from numpy import*
from scipy import*

#read data
train1 = genfromtxt('train.csv', delimiter=',', skip_header=1)
test1 = genfromtxt('test.csv', delimiter=',', skip_header=1)
train = train1[:, 1:]
result = train1[:, 0]

clf = svm.SVC(C=100000.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
gamma =0.0, kernel='poly', max_iter=-1, probability=False, random_state=None,
shrinking=True, tol=0.001, verbose=False)

clf.fit(train, result)
c = clf.predict(test1)
#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
#gamma=0.0, kernel='rbf', max_iter=-1, probability=False, random_state=None,
#shrinking=True, tol=0.001, verbose=False)
savetxt('d.csv', c, fmt='%d')

