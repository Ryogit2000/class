import LDA as lda
import data


#　1.　学習データの設定と、全体及び各カテゴリの平均の計算
#　X:　入力データ　（データ数×次元数のnumpy.ndarray）
#　Y:　出力データ　（データ数×1のnumpy.ndarray）
def __init__(self,X,Y):
    #学習データの設定
    self.X = X
    self.Y = Y
    self.dNum = X.shape[0] # 学習データ数
    self.xDim = X.shape[1] # 入力の次元数

    #　各カテゴリに属す入力データ
    self.Xneg = X[Y[:,0]==-1]
    self.Xpos = X[Y[:,0]==1]

    #　全体及び各カテゴリに属すデータの平均
    self.m = np.mean(self.X,axis=0,keepdims=True)
    self.mNeg = np.mean(self.Xneg,axis=0,keepdims=True)
    self.mPos = np.mean(self.Xpos,axis=0,keepdims=True)

#　2.　固有値問題によるモデルパラメータの最適化
def train(self):
    #　カテゴリ間分散共分散行列Sinterの計算
    Sinter = np.matmul((self.mNeg-self.mPos).T,self.mNeg-self.mPos)

    #　カテゴリ内分散共分散行列和Sintraの計算
    Xneg = self.Xneg - self.mNeg
    Xpos = self.Xpos - self.mPos
    Sintra = np.matmul(Xneg.T,Xneg) + np.matmul(Xpos.T,Xpos)

    #　固有値問題を解き、最大固有値の固有ベクトルを獲得
    [L,V] = np.linalg.eig(np.matmul(np.linalg.inv(Sintra),Sinter))
    self.w = V[:,[np.argmax(L)]]

#　3.　予測
#　X:　入力データ　（データ数×次元数のnumpy.ndarray）
def predict(self,x):
    return np.sign(np.matmul(x-self.m,self.w))
