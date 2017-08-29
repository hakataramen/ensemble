#!/usr/bin/python3
# -*- coding:utf-8 -*-

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import Majority
import compare_various_classifiers
import numpy as np

colors = ['black', 'orange', 'blue', 'green']
linestyles = [':', '--', '-.', '-']

for clf, label, clr, ls, in zip(compare_various_classifiers.all_clf, compare_various_classifiers.clf_labels, colors, linestyles):
    #陽性ラベルのクラスは1であることが前提
    y_pred = clf.fit(compare_various_classifiers.X_train, compare_various_classifiers.y_train).predict_proba(compare_various_classifiers.X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=compare_various_classifiers.y_test, y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr,
             color=clr,
             linestyle=ls,
             label='%s (auc = %0.2f)' % (label, roc_auc))

plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1],
         linestyle='--',
         color='gray',
         linewidth=2)
plt.xlim([-0.1, 1.1])
plt.xlim([-0.1, 1.1])
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

sc = StandardScaler()
X_train_std = sc.fit_transform(compare_various_classifiers.X_train)

from itertools import product
#決定木領域を描画する最小値、最大値を生成
x_min = X_train_std[:, 0].min() - 1
x_max = X_train_std[:, 0].max() + 1
y_min = X_train_std[:, 0].min() - 1
y_max = X_train_std[:, 0].max() + 1

#グリッドポイントを生成
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

#描画領域を2行2列に分割
f, axarr = plt.subplots(nrows=2, ncols=2,
                        sharex='col',
                        sharey='row',
                        figsize=(7, 5))
#決定領域のプロット、青や赤の散布図の作成などを実行
#変数idxは各分類器を描画する行と列の位置を表すタプル
for idx, clf, tt in zip(product([0,1], [0,1]), compare_various_classifiers.all_clf, compare_various_classifiers.clf_labels):
    clf.fit(X_train_std, compare_various_classifiers.y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx[0], idx[1]].scatter(X_train_std[compare_various_classifiers.y_train==0, 0],
                                  X_train_std[compare_various_classifiers.y_train==0, 1],
                                  c='blue',
                                  marker='^',
                                  s=50)
    axarr[idx[0], idx[1]].scatter(X_train_std[compare_various_classifiers.y_train==1, 0],
                                  X_train_std[compare_various_classifiers.y_train==1, 1],
                                  c='red',
                                  marker='o',
                                  s=50)
    axarr[idx[0], idx[1]].set_title(tt)

plt.text(-3.5, -4.5,
         s='Special width [standardized]',
         ha='center', va='center', fontsize=12)
plt.text(-10.5, 4.5,
         s='Petal length', va='center',
         fontsize=12, rotation=90)
plt.show()


compare_various_classifiers.mv_clf.get_params()
 
from sklearn.grid_search import GridSearchCV
params = {'decisiontreeclassifier__max_depth': [1, 2],
          'pipeline-1__clf__C': [0.001, 0,1, 100.0]}

grid = GridSearchCV(estimator=compare_various_classifiers.mv_clf,
                    param_grid=params,
                    cv=10,
                    scoring='roc_auc')
grid.fit(compare_various_classifiers.X_train, compare_various_classifiers.y_train)


for params, mean_score, scores in grid.grid_scores_:
          print("%0.3f+/-%0.2f %r" % (mean_score, scores.std() / 2, params))

print('Best parameters: %s' % grid.best_params_)
print('Accuracy: %2f' % grid.best_score_)
