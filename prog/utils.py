import torch
import torch.nn.functional as F
import numpy as np
import os
import mlflow
import copy
import numpy.linalg as LA

class EarlyStopping():
    def __init__(self,patience,path="checkpoint.pt"):
        self.best_loss_score = None
        self.loss_counter =0
        self.patience = patience
        self.path = path
        self.val_loss_min =None
        self.epoch = 0
        
    def __call__(self,loss_val,model,epoch):
        if self.best_loss_score is None:
            self.best_loss_score = loss_val
            self.save_best_model(model,loss_val)
            self.epoch = epoch
        elif self.best_loss_score > loss_val:
            self.best_loss_score = loss_val
            self.loss_counter = 0
            self.save_best_model(model,loss_val)
            self.epoch = epoch
        else:
            self.loss_counter+=1
            
        if self.loss_counter == self.patience:
            return True
        
        return False
    def save_best_model(self,model,loss_val):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = loss_val

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_train_label_per(data):
    cnt = 0
    for i in data.train_mask:
        if i == True:
            cnt+=1

    train_mask_label = cnt
    labels_num = len(data.train_mask)
    train_label_percent = train_mask_label/labels_num

    print(f"train_mask_label:{cnt},labels_num:{labels_num},train_label_percent:{train_label_percent}")

def log_artifacts(artifacts,output_path=None):
    if artifacts is not None:
        for artifact_name, artifact in artifacts.items():
            if isinstance(artifact, list):
                if output_path is not None:
                    artifact_name = f"{output_path}/{artifact_name}"
                    os.makedirs(output_path, exist_ok=True)
                np.save(artifact_name, artifact)
                mlflow.log_artifact(artifact_name)
            elif artifact is not None and artifact !=[]:
                if output_path is not None:
                    artifact_name = f"{output_path}/{artifact_name}"
                    os.makedirs(output_path, exist_ok=True)
                np.save(artifact_name, artifact.to('cpu').detach().numpy().copy())
                mlflow.log_artifact(artifact_name)




#隣接行列作成
def Adjacency_matrix(data):
    index = data.edge_index
    A = torch.zeros([len(data.x),len(data.x)]).to("cuda")
    #A = torch.zeros(len(data.x),len(data.x))
    for i in range(len(index[0])):
        A[index[0][i]][index[1][i]] = 1
    
    return A


#隣接行列からラプラシアン行列作成
def make_laplacian(data):
    A = Adjacency_matrix(data)
    D = torch.diag(torch.sum(A, axis=0))
    L = D - A
    return L

def make_normed_laplacian(data):
    A = Adjacency_matrix(data)
    D = np.diag(np.sum(A, axis=0))
    L = D - A
    normed_L = np.dot((np.sqrt(LA.inv(D))),np.dot(L, np.sqrt(LA.inv(D))))

    return normed_L

#グラフスペクトル作成
def make_feature(data):
    #次元数はdimで指定
    dim = 7
    L = make_laplacian(data)
    #L = make_normed_laplacian(A)
    #lam, v = torch.linalg.eig(L)
    lam, v = torch.linalg.eigh(L,UPLO='L')
    #print(v)
    Vec = []

    for i in range(dim):
        vec = []

        for j in range(len(v)):
            vec.append(v[j,i])

        Vec.append(vec)
    
    feature = torch.tensor(Vec,dtype=float).to("cuda")
    feature = feature.T

    #print(feature)

    return feature


def edge_TF(data):
    index = data.edge_index
    class_vec = torch.full((1,len(index[0])), False, dtype=bool).to("cuda")
    c_f = 0
    c_t = 0

    #print(data.y)
    #print(index[0][0])

    for i in range(len(index[0])):

        if(data.y[index[0][i]] == data.y[index[1][i]]):
            class_vec[0][i] = True
            c_t += 1
        else:
            class_vec[0][i] = False
            c_f += 1

    #print(c_t)
    #print(c_f)

    return class_vec

    



#エッジ存在　同じクラス
def edge_dist_sameclass(data):
    X = make_feature(data)
    class_vec = edge_TF(data) 
    index = data.edge_index
    t = torch.zeros((len(index[0]),1))
    vec1 = []
    for i in range(len(index[0])):
        if(class_vec[0][i] == True):
            t[i][0] = torch.dist(X[index[0][i]],X[index[1][i]])
            vec1.append(torch.dist(X[index[0][i]],X[index[1][i]]))

    return t

def edge_dist_diffclass(data):
    X = make_feature(data)
    class_vec = edge_TF(data) 
    index = data.edge_index
    t = torch.zeros((len(index[0]),1)).to("cuda")
    vec1 = []
    for i in range(len(index[0])):
        if(class_vec[0][i] == False):
            t[i][0] = torch.dist(X[index[0][i]],X[index[1][i]])
            vec1.append(torch.dist(X[index[0][i]],X[index[1][i]]))

    return t


def loss_dist(out,class_vec,data):
    #class_vec = edge_TF(data) 
    index = data.edge_index
    

    t = torch.zeros((len(index[0]),1)).to("cuda")
    #vec1 = []
    for i in range(len(index[0])):
        if(class_vec[i] == False):
            t[i][0] = torch.dist(out[index[0][i]],out[index[1][i]])
            #vec1.append(torch.dist(out[index[0][i]],out[index[1][i]]))


    return t

def loss_dist2(out,class_vec,data):
    #class_vec = edge_TF(data) 
    index = data.edge_index
    

    #t = torch.zeros((len(index[0]),1)).to("cuda")
    vec1 = []
    # for i in range(len(index[0])):
    #     if(class_vec[0][i] == False):
    #         t[i][0] = torch.dist(out[index[0][i]],out[index[1][i]])
    #         vec1.append(torch.dist(out[index[0][i]],out[index[1][i]]))

    vec1 = [torch.dist(out[index[0][i]],out[index[1][i]]) for i in range(len(index[0])) if class_vec[i] == False]

    return vec1



def loss_dist3(out,class_vec,data):
    dist_tensor = torch.sqrt(torch.mm(out, out.T))
    index = data.edge_index
    t = torch.zeros((len(index[0]),1)).to("cuda")

    for i in range(len(index[0])):
        if(class_vec[i] == False):
            t[i][0] = dist_tensor[index[0][i], index[1][i]]
    
    return t

def loss_dist4(out,class_vec,data):
    dist_tensor = F.pdist(out)
    index = data.edge_index
    t = torch.zeros((len(index[0]),1)).to("cuda")

    for i in range(len(index[0])):
        if(class_vec[i] == False):
            t[i][0] = dist_tensor[index[0][i], index[1][i]]
    
    return t

