# GCN-pytorch
This is my pytorch's implementation of tkipf's GCN. 

For the original version:
[Tensorflow-Version](https://github.com/tkipf/gcn)

There are minor differences, including:

1. Add multiheads to each layers
2. Dropout is applied not only to the input of each layer, but also to the symmetrically normalized Laplacian matrix in each head
3. We leverage a GAT style early stop mechanism, which can greatly enhance the performance

For any questions, please feel free to open an issue. You may expect my response in less than 7 days.

## Future Works
1. Add argparse to the code to make it easier to try different hyperparameters
2. Tuning the model for a better performance on Cora
3. Might add some other aggregation based model to this repository.

## Notification
It should be notified that I am not the author of GCN. You might cite this work according to [Tensorflow-Version](https://github.com/tkipf/gcn).

## Performance
This model could achieve an average accuracy tested on Cora at <img src="https://latex.codecogs.com/gif.latex?82.0&space;\pm&space;0.8&space;\%" title="82.0 \pm 0.8 \%" />

## How to run our code

### Requirements
Before running our codes, please make sure these dependencies is installed in your environment

1. numpy==1.16.4
2. torch==1.1.0
3. scipy==1.3.0
4. networkx==2.3

by
```bash
python -m pip install -r requirements.txt
```

### Evaluating on Cora, Citeseer, Pubmed
You may run 
```bash
python gcn_main.py [-h] [-dataset [DATASET]] [-epoch EPOCH] [-early EARLY]
                   [-device [DEVICE]] [-lr LR] [-weight WEIGHT]
                   [-hidden [HIDDEN]] [-layer LAYER]
```

-dataset            you should use one of 'cora', 'pubmed', 'citeseer'. Default is cora
-epoch              maximum epoch, default is 1000
-early              early stop, default is 100
-device             which devices used to train, default is cpu
-lr                 learning rate, default is 0.01
-weight             weight decay, default is 5e-4
-hidden             hidden layer size, split by comma, default is 16
-layer              layer number, default is 2