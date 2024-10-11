import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from new_model import GAT
import torch_geometric.transforms as T
from tqdm import tqdm
from utils import EarlyStopping, set_seed
from utils_group import new_graph
import hydra
from hydra import utils
# import mlflow

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def random_splits(data, num_classes, lcc_mask):

    torch.manual_seed(42)
    indices = []
    if lcc_mask is not None:
        for i in range(num_classes):
            index = (data.y[lcc_mask] == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

    else:
        for i in range(num_classes):
            index = (data.y==i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

    val_size = 500 // num_classes
    test_size = 1000 // num_classes

    train_index = torch.cat([i[:40] for i in indices], dim=0)
    val_index = torch.cat([i[40:40+val_size] for i in indices], dim=0)
    test_index = torch.cat([i[40+val_size:40+val_size+test_size] for i in indices], dim=0)
    test_index = test_index[torch.randperm(test_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(test_index, size=data.num_nodes)

    return data, torch.cat((train_index, val_index), dim=0)

def train(data, data_group, group_index, model, optimizer):
    model.train()
    optimizer.zero_grad()
    print("2", type(group_index))


    if data_group.edge_index.numel() > 0 and data_group.edge_index.min() < 0:
        print(True)
    else:
        print(False)
    if data_group.edge_index.max() >= data.x.size(0):
        print(True)
    else:
        print(False)
    if data_group.edge_index.numel() == 0:
        print("index is empty!")

    # print("x")
    # print(data.x.device)
    # print(type(data.x))
    # print(data.x.shape)
    # print("edge_index")
    # print(data.edge_index.device)
    # print(type(data.edge_index))
    # print(data.edge_index.shape)
    # print("g_x")
    # print(data_group.x.device)
    # print(type(data_group.x))
    # print(data_group.x.shape)
    # print("g_edge_index")
    # print(data_group.edge_index.device)
    # print(type(data_group.edge_index))
    # print(data_group.edge_index.shape)

    out_train, hs, *_ = model(data.x, data.edge_index, data_group.x, data_group.edge_index, group_index)

    out_train_softmax =  F.log_softmax(out_train, dim=-1)
    loss_train  = F.nll_loss(out_train_softmax[data.train_mask], data.y[data.train_mask])

    loss_train.backward()
    optimizer.step()

    #validation
    model.eval()
    out_val, _, _ = model(data.x, data.edge_index)


    out_val_softmax = F.log_softmax(out_val, dim=-1)
    loss_val = F.nll_loss(out_val_softmax[data.val_mask], data.y[data.val_mask])

    return loss_val.item()

@torch.no_grad()
def test(data,model):
    model.eval()
    out,_,attention = model(data.x, data.edge_index)
    out_softmax = F.log_softmax(out, dim=1)
    acc = accuracy(out_softmax,data,'test_mask')
    attention = model.get_v_attention(data.edge_index,data.x.size(0),attention)
    return acc,attention,out

def accuracy(out,data,mask):
    mask = data[mask]
    acc = float((out[mask].argmax(-1) == data.y[mask]).sum() / mask.sum())
    return acc

def run(data,model,optimizer,cfg):
    early_stopping = EarlyStopping(cfg['patience'],path=cfg['path'])
    data_group, group_index = new_graph(data.x, data.edge_index)
    for epoch in range(cfg['epochs']):
        if epoch % 10 == 0:
            data_group, group_index = new_graph(data.x, data.edge_index)
            print("1", type(group_index))

        loss_val = train(data, data_group ,group_index, model, optimizer)
        if early_stopping(loss_val,model,epoch) is True:
            break
    model.load_state_dict(torch.load(cfg['path']))
    test_acc,attention,h = test(data,model)
    return test_acc,early_stopping.epoch,attention,h



@hydra.main(config_path='conf', config_name='config')
def main(cfg):

    print(utils.get_original_cwd())
    # mlflow.set_tracking_uri('http://127.0.0.1:5000')
    # mlflow.set_experiment("output")
    # mlflow.start_run()

    cfg = cfg[cfg.key]
    # for key,value in cfg.items():
    #     mlflow.log_param(key,value)

    root = utils.get_original_cwd() + '/data/' + cfg['dataset']
    dataset = Planetoid(root           = root,
                        name          = cfg['dataset'],
                        transform     = eval(cfg['transform']),
                        pre_transform = eval(cfg['pre_transform']))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)
    print(data)
    
    data,index = random_splits(data=data, num_classes=cfg["n_class"], lcc_mask=None)
    # check_train_label_per(data)

    artifacts,test_accs,epochs,attentions,hs = {},[],[],[],[]
    artifacts[f"{cfg['dataset']}_y_true.npy"] = data.y
    artifacts[f"{cfg['dataset']}_x.npy"] = data.x
    artifacts[f"{cfg['dataset']}_supervised_index.npy"] = index

    
    for i in tqdm(range(cfg['run'])):
        set_seed(i)
        #if cfg['mode'] == 'original':
        model = GAT(cfg).to(device)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg["learing_late"],weight_decay=cfg['weight_decay'])
        #test_acc,epoch,attention,h = run(data,model,optimizer,cfg,spectral_dist)
        test_acc,epoch,attention,h = run(data,model,optimizer,cfg)
        test_accs.append(test_acc)
        epochs.append(epoch)
        attentions.append(attention)
        hs.append(h)
        #print(hs[0])

    acc_max_index = test_accs.index(max(test_accs))
    artifacts[f"{cfg['dataset']}_{cfg['att_type']}_attention_L{cfg['num_layer']}.npy"] = attentions[acc_max_index]
    artifacts[f"{cfg['dataset']}_{cfg['att_type']}_h_L{cfg['num_layer']}.npy"] = hs[acc_max_index]

    test_acc_ave = sum(test_accs)/len(test_accs)
    epoch_ave = sum(epochs)/len(epochs)
    #log_artifacts(artifacts,output_path=f"{utils.get_original_cwd()}/DeepGAT/output/{cfg['dataset']}/{cfg['att_type']}/oracle/{cfg['oracle_attention']}")

    # mlflow.log_metric('epoch_mean',epoch_ave)
    # mlflow.log_metric('test_acc_min',min(test_accs))
    # mlflow.log_metric('test_acc_mean',test_acc_ave)
    # mlflow.log_metric('test_acc_max',max(test_accs))
    # mlflow.end_run()
    print("epoch_ave", epoch_ave)
    print("acc_ave", test_acc_ave)
    print("test_acc_min", min(test_accs))
    print("test_acc_max", max(test_accs))
    
    return test_acc_ave

if __name__ == "__main__":
    main()