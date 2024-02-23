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


