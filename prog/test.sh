IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python main.py --multirun 'key=GAT_cora_tuned_DP' \
     'GAT_cora.n_head=8' \
     'GAT_cora.n_head_last=1' \
     'GAT_cora.mode=original' \
     'GAT_cora.run=10' \
     'GAT_cora.num_layer=3' \
     'GAT_cora.norm=choice(None,LayerNorm,BatchNorm1d)' \
     'GAT_cora.dropout=choice(0.,0.2,0.4,0.6)' \
     'GAT_cora.learing_late=choice(0.05,0.01,0.005,0.001)' \
     'GAT_cora.weight_decay=choice(0,1E-4,5E-4,1E-3)' \
     'GAT_cora.n_hid=5' \
     'GAT_cora.att_type=choice(DP)'\
     'GAT_cora.layer_loss=choice(supervised,unsupervised)'\
     ")
for STR in ${ary[@]}
do
    eval "${STR}"
done
