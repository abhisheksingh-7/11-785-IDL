# HW1P2 - MLP for Phoneme Classification

## Setup
- Decompress the `.tar` file
- Download File `abhishe6_11785_HW1P2.ipynb`
- Go to https://colab.research.google.com/
- Upload this `.ipynb` file onto colab
- Enter your kaggle and WandB credentials in respective cells
- Go to Runtime -> Run all OR press [Ctrl+F9] to run all cells

## Experiment Details
- No of epochs trained: 28
- Optimizer: `AdamW`
- Architecture Details: View `HW1P2-wandb-ablations.csv` or `HW1P2-NAS.xlsx`
- WandB Experiment Screenshots: View `HW1P2-WandB-Charts.pdf`
- Final Submission Test Data Predictions: View `submission.csv`
- Final `training_accuracy` = 89.79%
- Final `validation_accuracy` = 88.78%

## Neural Net Architecture

```
=========================================================================
                         Kernel Shape  Output Shape     Params  Mult-Adds
Layer                                                                    
0_model.Linear_0         [1647, 2048]  [1024, 2048]  3.375104M  3.373056M
1_model.BatchNorm1d_1          [2048]  [1024, 2048]     4.096k     2.048k
2_model.GELU_2                      -  [1024, 2048]          -          -
3_model.Linear_3         [2048, 2048]  [1024, 2048]  4.196352M  4.194304M
4_model.BatchNorm1d_4          [2048]  [1024, 2048]     4.096k     2.048k
5_model.GELU_5                      -  [1024, 2048]          -          -
6_model.Dropout_6                   -  [1024, 2048]          -          -
7_model.Linear_7         [2048, 2048]  [1024, 2048]  4.196352M  4.194304M
8_model.BatchNorm1d_8          [2048]  [1024, 2048]     4.096k     2.048k
9_model.GELU_9                      -  [1024, 2048]          -          -
10_model.Dropout_10                 -  [1024, 2048]          -          -
11_model.Linear_11       [2048, 2048]  [1024, 2048]  4.196352M  4.194304M
12_model.BatchNorm1d_12        [2048]  [1024, 2048]     4.096k     2.048k
13_model.GELU_13                    -  [1024, 2048]          -          -
14_model.Dropout_14                 -  [1024, 2048]          -          -
15_model.Linear_15       [2048, 1288]  [1024, 1288]  2.639112M  2.637824M
16_model.BatchNorm1d_16        [1288]  [1024, 1288]     2.576k     1.288k
17_model.GELU_17                    -  [1024, 1288]          -          -
18_model.Dropout_18                 -  [1024, 1288]          -          -
19_model.Linear_19       [1288, 1024]  [1024, 1024]  1.319936M  1.318912M
20_model.BatchNorm1d_20        [1024]  [1024, 1024]     2.048k     1.024k
21_model.GELU_21                    -  [1024, 1024]          -          -
22_model.Dropout_22                 -  [1024, 1024]          -          -
23_model.Linear_23         [1024, 42]    [1024, 42]     43.05k    43.008k
-------------------------------------------------------------------------
                          Totals
Total params          19.987266M
Trainable params      19.987266M
Non-trainable params         0.0
Mult-Adds             19.966216M
=========================================================================

```

## Hyperparameters for Best Score
```
config = {
    'epochs'        : 30,
    'batch_size'    : 1024,
    'context'       : 30,
    'init_lr'       : 1e-3,
    'architecture'  : 'high-cutoff',
    'train_dataset' : 'train-clean-360',
    'val_dataset'   : 'dev-clean',
    'test_dataset'  : 'test-clean',
    'dropout_rate'  : 0.2,
    'weight_decay'  : 1e-2,
    'patience'      : 2
}
```

## Data loading scheme
```
train_loader = torch.utils.data.DataLoader(
    dataset     = train_data, 
    num_workers = 4,
    batch_size  = config['batch_size'], 
    pin_memory  = True,
    shuffle     = True
)

val_loader = torch.utils.data.DataLoader(
    dataset     = val_data, 
    num_workers = 2,
    batch_size  = config['batch_size'],
    pin_memory  = True,
    shuffle     = False
)

test_loader = torch.utils.data.DataLoader(
    dataset     = test_data, 
    num_workers = 2, 
    batch_size  = config['batch_size'], 
    pin_memory  = True, 
    shuffle     = False
)
```
