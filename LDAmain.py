#　1.　データの作成
myData = data.classification(negLabel=-1,posLabel=1)
myData.selectData(dataType=2)

#　3.　線形判別モデルの学習
myModel = lda.LDA(Xtr,Ytr)
myModel.train()

#　4.　線形判別モデルの評価
print(f"モデルパラメータ:\nw={myModel.w},\n平均:m={myModel.m}")
print(f"正解率={myModel.accuracy(Xte,Yte):.2f}")

#　アンダーサンプリング

#　最小のカテゴリのデータ数
minNum = np.min([np.sum(Ytr==-1),np.sum(Ytr==1)])

#　各カテゴリのデータ
Xneg = Xtr[Ytr[:,0]==-1]
Xpos = Xtr[Ytr[:,0]==1]

#　最小データ数分だけ各カテゴリから選択し結合
Xtr = np.concatenate([Xpos[:minNum],Xneg[:minNum]],axis=0)
Ytr = np.concatenate([-1*np.ones(shape=[minNum,1]),1*np.ones(shape=[minNum,1])])