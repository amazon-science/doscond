# DosCond 

[KDD 2022] The implementation for ["Condensing Graphs via One-Step Gradient Matching"](https://arxiv.org/abs/2206.07746) on graph classification is shown below. For node classification, please refer to [link](https://github.com/ChandlerBang/GCond/tree/main/KDD22_DosCond).


Abstract
--
As training deep learning models on large dataset takes a lot of time and resources, it is desired to construct a small synthetic dataset with which we can train deep learning models sufficiently. There are recent works that have explored solutions on condensing image datasets through complex bi-level optimization. For instance, dataset condensation (DC) matches network gradients w.r.t. large-real data and small-synthetic data, where the network weights are optimized for multiple steps at each outer iteration. However, existing approaches have their inherent limitations: (1) they are not directly applicable to graphs where the data is discrete; and (2) the condensation process is computationally expensive due to the involved nested optimization. To bridge the gap, we investigate efficient dataset condensation tailored for graph datasets where we model the discrete graph structure as a probabilistic model. We further propose a one-step gradient matching scheme, which performs gradient matching for only one single step without training the network weights. 


## Requirements
All experiments are performed under `python=3.8.8`

Please see [requirements.txt](https://github.com/amazon-research/doscond/blob/main/requirements.txt).
```
numpy==1.20.1
ogb==1.3.0
pandas==1.2.3
scikit_learn==1.1.1
scipy==1.6.2
torch==1.8.1
torch_geometric==2.0.1
tqdm==4.60.0
```


## Run the code
Use the following code to run the experiment 
```
python main.py --dataset DD --init real  --gpu_id=0 --nconvs=3 --dis=mse --lr_adj=2  --lr_feat=0.01 --epochs=1000 --eval_init=1  --net_norm=none --pool=mean --seed=0 --ipc=1 
```

The hyper-parameter settings are listed in [`script.sh`](https://github.com/amazon-research/doscond/blob/main/script.sh). Run the following command to get the results.
```
bash script.sh
```

By specifying `save=1`, we can save the condensed graphs, e.g.,
```
python main.py --dataset DD --init real  --gpu_id=0 --nconvs=3 --dis=mse --lr_adj=2  --lr_feat=0.01 --epochs=1000 --eval_init=1  --net_norm=none --pool=mean --seed=0 --ipc=1 --save=1 
```
The condensed graphs will be saved under the `saved` directory. Run multiple seeds to get the multiple condesed datasets. Then use the following command to test the condensed dataset for multiple times.
```
python test_saved_graphs.py --filename DD_ipc1_s0_lra2.0_lrf0.01.pt --dataset DD
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.


