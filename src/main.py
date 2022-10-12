import numpy as np
import pandas as pd
pd.set_option('display.max_rows',10) # 调整pandas行的显示限制 

from utils import *
from model import DNN
from train import *

AllNode = pd.read_csv('../data/AllNode_DrPr.csv',names=[0,1],skiprows=1)
Alledge = pd.read_csv('../data/DrPrNum_DrPr.csv',header=None)
features = pd.read_csv('../data/AllNodeAttribute_DrPr.csv', header = None)
features = features.iloc[:,1:]

labels = pd.DataFrame(np.random.rand(len(AllNode),1))
labels[0:549]=0 
labels[549:]=1 
labels = labels[0]

adj, features, labels, idx_train, idx_val, idx_test  = load_data(Alledge,features,labels)

# set parameter
class item:
    def __init__(self):
        self.epochs = 200
        self.lr = 1e-1
        self.k1 = 200  
        self.k2 = 10
        self.epsilon1 = 0.03 
        self.epsilon2 = 0.05
        self.hidden = 64
        self.dropout = 0.5
        self.runs = 1

args = item()

node_sum = adj.shape[0]
edge_sum = adj.sum()/2
row_sum = (adj.sum(1) + 1)
norm_a_inf = row_sum/ (2*edge_sum+node_sum)

adj_norm = sparse_mx_to_torch_sparse_tensor(aug_random_walk(adj))

features = F.normalize(features, p=1)
feature_list = []
feature_list.append(features)
for i in range(1, args.k1):
    feature_list.append(torch.spmm(adj_norm, feature_list[-1]))

norm_a_inf = torch.Tensor(norm_a_inf).view(-1, node_sum)
norm_fea_inf = torch.mm(norm_a_inf, features)

hops = torch.Tensor([0]*(adj.shape[0]))
mask_before = torch.Tensor([False]*(adj.shape[0])).bool()

for i in range(args.k1):
    dist = (feature_list[i] - norm_fea_inf).norm(2, 1)
    mask = (dist<args.epsilon1).masked_fill_(mask_before, False)
    mask_before.masked_fill_(mask, True)
    hops.masked_fill_(mask, i)
mask_final = torch.Tensor([True]*(adj.shape[0])).bool()
mask_final.masked_fill_(mask_before, False)
hops.masked_fill_(mask_final, args.k1-1)
print("Local Smoothing Iteration calculation is done.")

input_feature = aver(hops, adj, feature_list)
print("Local Smoothing is done.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_class = 64#(labels.max()+1).item()

input_feature = input_feature.to(device)

print("Start training...")  

model = DNN(features.shape[1], args.hidden, n_class, args.dropout).to(device)
model.eval()
output, Emdebding = model(input_feature)#.cpu()
Emdebding_GCN = pd.DataFrame(Emdebding.detach().cpu().numpy())
# Emdebding_GCN.to_csv('../data/Emdebding_GCN2_DrPr.csv', header=None,index=False)

Positive = Alledge #pd.read_csv('../data/DrPrNum_DrPr.csv',header=None)
AllNegative = pd.read_csv('../data/AllNegative_DrPr.csv',header=None) 
Negative = AllNegative.sample(n=1923, random_state=520)
Positive[2] = Positive.apply(lambda x: 1 if x[0] < 0 else 1, axis=1)
Negative[2] = Negative.apply(lambda x: 0 if x[0] < 0 else 0, axis=1)
result = pd.concat([Positive,Negative]).reset_index(drop=True)
X = pd.concat([Emdebding_GCN.loc[result[0].values.tolist()].reset_index(drop=True),Emdebding_GCN.loc[result[1].values.tolist()].reset_index(drop=True)],axis=1)
Y = result[2]

from sklearn.model_selection import  StratifiedKFold,KFold
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from scipy import interp


NmedEdge = 499
DmedEdge = 7
SmedEdge = 0.85
k_fold = 10
print("%d fold CV"% k_fold)
i=0
tprs=[]
aucs=[]
mean_fpr=np.linspace(0,1,1000)
colorlist = ['red','firebrick','magenta', 'gold','limegreen','royalblue', 'purple', 'green','magenta', 'blue', 'black']
AllResult = []

skf = StratifiedKFold(n_splits=k_fold,random_state=0, shuffle=True)
for train_index, test_index in skf.split(X,Y):
  
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    model = GradientBoostingClassifier(n_estimators=NmedEdge,max_depth=DmedEdge,subsample=SmedEdge)
    model.fit(np.array(X_train), np.array(Y_train))
    y_score0 = model.predict(np.array(X_test))
    y_score_RandomF = model.predict_proba(np.array(X_test))
    fpr,tpr,thresholds=roc_curve(Y_test,y_score_RandomF[:,1])
    tprs.append(interp(mean_fpr,fpr,tpr))
    tprs[-1][0]=0.0
    #auc
    roc_auc=auc(fpr,tpr)
    aucs.append(roc_auc)
    print('ROC fold %d(AUC=%0.4f)'% (i,roc_auc))
 
mean_tpr=np.mean(tprs,axis=0)
mean_tpr[-1]=1.0
mean_auc=auc(mean_fpr,mean_tpr)
print('Mean ROC (AUC=%0.4f)'% (mean_auc)) 

