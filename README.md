# This is assignment5 for **The School of AI**
- Here, you will find three files named as
    - model.py: It has the Model definition
    - utils.py: It has common utility functions definitions
    - S5.ipybn: Main notebook where the utility functions and Model is imported and the network is trained

<br>

## utils.py: has the following functions
- get_tfms: generate train/test transforms to be applied to the dataset
- get_ds: generate train/test datasets
- get_dls: generate train/test dataloaders
- GetCorrectPredCount: compute accuracy
- train_step: perform one training epoch
- test_step: perform one test epoch for validation
- plot_acc_loss: plotiing the statistics
- show_dls_samples: shows some samples from the dataloader
- get_optimizer: return SGD optimizer
- get_scheduler: returns StepLR scheduler for changing learning after pre-defined number of epochs
- getlossCriterion: returns CrossEntropy Loss

<br>

## Model.py: 
- Has the network architecture definitions for classification task having CNN network as backbone with couple of MLP layers for classification towards the end <br>
![image.png](attachment:8ec4d8da-cfbc-4d1e-a231-7dba6940f097.png)



