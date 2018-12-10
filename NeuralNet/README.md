# Neural Net Implementation

## Task 2.1 (python main.py -t 2.1 )
If a neural network with sigmoid activations and one hidden layer is given with

<img src="https://latex.codecogs.com/gif.latex?W&space;=&space;\begin{bmatrix}&space;-2&space;&&space;2&space;&&space;-1&space;\\&space;1&space;&&space;-2&space;&&space;0&space;\end{bmatrix}&space;,\quad&space;v&space;=&space;\begin{bmatrix}&space;3&space;\\&space;1&space;\end{bmatrix}" title="W = \begin{bmatrix} -2 & 2 & -1 \\ 1 & -2 & 0 \end{bmatrix} ,\quad v = \begin{bmatrix} 3 \\ 1 \end{bmatrix}" />

Then this calculates 

<img src="https://latex.codecogs.com/gif.latex?z_i,&space;h(z_i),&space;\text{and&space;}&space;\hat{y}_i&space;\text{&space;for&space;a&space;training&space;example&space;with&space;features&space;}&space;x^T_i=[-3\&space;-2\&space;\&space;2]." title="z_i, h(z_i), \text{and } \hat{y}_i \text{ for a training example with features } x^T_i=[-3\ -2\ \ 2]." />


## Task 2.2
This uses [the MNIST handwritten digits data file](http://deeplearning.net/data/mnist/mnist.pkl.gz) 
If `python main.py -t 2` is run, this will train a manually implemented one-hidden-layer neural network on the dataset. 
With `sgd=0`, this will do normal gradient descent (100 iterations).
![Normal gradient descent performance](./figs/output2_2_original.png)

With `sgd=1`, this will use a random minibatch size of 1% of the whole data set at a time and do 10 epochs of gradient descent with a constant learning rate of <img src="https://latex.codecogs.com/gif.latex?\inline&space;\alpha&space;=&space;0.001" title="\alpha = 0.001" />.
![Stochastic minibatch performance](./figs/output2_2_SGD.png)
