# GCN-pytorch
This is my pytorch's implementation of tkipf's GCN. 

For the original version:
[Tensorflow-Version](https://github.com/tkipf/gcn)

There are minor differences, including:

1. Add multiheads to each layers
2. Dropout is applied not only to the input of each layer, but also to the symmetrically normalized Laplacian matrix in each head
3. We leverage a GAT style early stop mechanism, which can greatly enhance the performance

I have conducted experients on Tensorflow. It shows that additional dropout mechanism could enhance the performance of the original algorithm to 82% tested on Cora.

However, this version is not as good as I expected. Improvement might be done in the near future.

For any questions, please feel free to open an issue. You may expect my response in less than 7 days.

## Future Works
1. Add argparse to the code to make it easier to try different hyperparameters
2. Tuning the model for a better performance on Cora
3. Might add some other aggregation based model to this repository.

## Notification
It should be notified that I am not the author of GCN. You might cite this work according to [Tensorflow-Version](https://github.com/tkipf/gcn).

## Performance
This model could achieve an average accuracy tested on Cora at 

<img src="https://latex.codecogs.com/gif.latex?82.0&space;\pm&space;0.8&space;\%" title="82.0 \pm 0.8 \%" />

## How to run our code
