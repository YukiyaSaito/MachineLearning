#Fill lower halves of MNIST digit images
The three approaches are:
1. Inhomogeneous Markov Chain along the columns.
2. Tabular DAG. The element (i,j) depends on (i-2,j-2), (i-2,j-1), (i-2,j), (i-1,j-2), (i-1,j-1), (i-1,j), (i,j-2), (i,j-1) elements.
3. Sigmoid Belief Network. (i,j)-th element depends on all the previous elements.

##Results
1. Inhomogeneous Markov Chain:
![Inhomogeneous Markov Chain](inhom_Markov.png)
![Probability Distribution](inhom_Markov_prob.png)

2. Tabular DAG:
![Tabular DAG](tabularDAG.png)

3. Sigmoid Belief Network:
![Sigmoid Belief Network](sigBelNet.png)
