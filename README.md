# Neural-Networks
Contains Python Code for implemenation of various logic gates using BackPropogation in python
Example Implementation of AND Gate using this Module
## Code 
>import NeuralNetwork  
>b=NeuralNetwork.BackPropogate("and")  
>b.selectEpochs(10000)  
>b.setLearningRate(0.1)  
>b.train()  
>b.print("oo")  
>b.showerror()  
>b.plot()  

#### Print can take any of four parameter:-
* ow1-Output Weights1 Array
* ow2-Output Weights2 Array
* ob-Output Biases Array
* oo-Our Final Output 
### Functions it includes-
* sigmoid
* drivative
* train -Train Different Gates Model
* showerror -Show Error Percentage from Expected Output
* plot :- Plot different Grpahs for you
* print :- Print can take 4 inputs depending on which it provides output
* selectEpochs :- To select the number of epochs(Default :-1000)
* setLearningRate :- To select the learning rate of your Model(Default-0.1)
