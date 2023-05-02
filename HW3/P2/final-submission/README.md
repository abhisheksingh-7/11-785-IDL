# HW2P2 - CNNs - Image Classification and Verification

## Setup
- Decompress the `.tar` file
- Download File `abhishe6_11785_HW3P2.ipynb`
- Go to https://colab.research.google.com/
- Upload this `.ipynb` file onto colab
- Enter your kaggle and WandB credentials in respective cells
- Go to Runtime -> Run all OR press [Ctrl+F9] to run all cells

## Experiment Details
- Architecture: Pyramid Bi-LSTM (pBLSTM)
- No of epochs trained: 52
- Optimizer: `AdamW`
- Loss Function: `CTC Loss`
- Scheduler: `ReduceLROnPlateau`
- Metric: `levenshtein distance`
- Mixed Precision Training: `Yes`
- Architecture Details: View `HW3P2-wandb-ablations.csv`
- WandB Experiment Screenshots: View `HW3P2-wandb-charts.pdf` or https://wandb.ai/deeper_learners/hw3p2-ablations
- Final `validation distance` = 2.59
- Final `test distance` = 2.98

## Neural Net Architecture

```
ASRModel(
  (augmentations): Sequential(
    (0): PermuteBlock()
    (1): FrequencyMasking()
    (2): TimeMasking()
    (3): PermuteBlock()
  )
  (encoder): Encoder(
    (embedding): Sequential(
      (0): PermuteBlock()
      (1): Conv1d(27, 256, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
      (2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): PermuteBlock()
    )
    (pBLSTMs): Sequential(
      (0): pBLSTM(
        (blstm): LSTM(512, 256, num_layers=2, batch_first=True, bidirectional=True)
      )
      (1): LockedDropout()
      (2): pBLSTM(
        (blstm): LSTM(1024, 256, num_layers=2, batch_first=True, bidirectional=True)
      )
      (3): LockedDropout()
    )
  )
  (decoder): Decoder(
    (mlp): Sequential(
      (0): PermuteBlock()
      (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): PermuteBlock()
      (3): Linear(in_features=512, out_features=2048, bias=True)
      (4): PermuteBlock()
      (5): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (6): PermuteBlock()
      (7): GELU(approximate='none')
      (8): Linear(in_features=2048, out_features=2048, bias=True)
      (9): PermuteBlock()
      (10): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (11): PermuteBlock()
      (12): GELU(approximate='none')
      (13): Dropout(p=0.2, inplace=False)
      (14): Linear(in_features=2048, out_features=41, bias=True)
    )
    (softmax): LogSoftmax(dim=2)
  )
)
========================================================================================
                                    Kernel Shape     Output Shape     Params  \
Layer                                                                          
0_augmentations.PermuteBlock_0                 -   [64, 27, 2001]          -   
1_augmentations.FrequencyMasking_1             -   [64, 27, 2001]          -   
2_augmentations.TimeMasking_2                  -   [64, 27, 2001]          -   
3_augmentations.PermuteBlock_3                 -   [64, 2001, 27]          -   
4_encoder.embedding.PermuteBlock_0             -   [64, 27, 2001]          -   
5_encoder.embedding.Conv1d_1        [27, 256, 3]  [64, 256, 2001]    20.736k   
6_encoder.embedding.BatchNorm1d_2          [256]  [64, 256, 2001]      512.0   
7_encoder.embedding.PermuteBlock_3             -  [64, 2001, 256]          -   
8_encoder.pBLSTMs.0.LSTM_blstm                 -     [40989, 512]   3.15392M   
9_encoder.pBLSTMs.LockedDropout_1              -     [40989, 512]          -   
10_encoder.pBLSTMs.2.LSTM_blstm                -     [27909, 512]  4.202496M   
11_encoder.pBLSTMs.LockedDropout_3             -     [27909, 512]          -   
12_decoder.mlp.PermuteBlock_0                  -   [64, 512, 500]          -   
13_decoder.mlp.BatchNorm1d_1               [512]   [64, 512, 500]     1.024k   
14_decoder.mlp.PermuteBlock_2                  -   [64, 500, 512]          -   
15_decoder.mlp.Linear_3              [512, 2048]  [64, 500, 2048]  1.050624M   
16_decoder.mlp.PermuteBlock_4                  -  [64, 2048, 500]          -   
17_decoder.mlp.BatchNorm1d_5              [2048]  [64, 2048, 500]     4.096k   
18_decoder.mlp.PermuteBlock_6                  -  [64, 500, 2048]          -   
19_decoder.mlp.GELU_7                          -  [64, 500, 2048]          -   
20_decoder.mlp.Linear_8             [2048, 2048]  [64, 500, 2048]  4.196352M   
21_decoder.mlp.PermuteBlock_9                  -  [64, 2048, 500]          -   
22_decoder.mlp.BatchNorm1d_10             [2048]  [64, 2048, 500]     4.096k   
23_decoder.mlp.PermuteBlock_11                 -  [64, 500, 2048]          -   
24_decoder.mlp.GELU_12                         -  [64, 500, 2048]          -   
25_decoder.mlp.Dropout_13                      -  [64, 500, 2048]          -   
26_decoder.mlp.Linear_14              [2048, 41]    [64, 500, 41]    84.009k   
27_decoder.LogSoftmax_softmax                  -    [64, 500, 41]          -   

                                     Mult-Adds  
Layer                                           
0_augmentations.PermuteBlock_0               -  
1_augmentations.FrequencyMasking_1           -  
2_augmentations.TimeMasking_2                -  
3_augmentations.PermuteBlock_3               -  
4_encoder.embedding.PermuteBlock_0           -  
5_encoder.embedding.Conv1d_1        41.492736M  
6_encoder.embedding.BatchNorm1d_2        256.0  
7_encoder.embedding.PermuteBlock_3           -  
8_encoder.pBLSTMs.0.LSTM_blstm       3.145728M  
9_encoder.pBLSTMs.LockedDropout_1            -  
10_encoder.pBLSTMs.2.LSTM_blstm      4.194304M  
11_encoder.pBLSTMs.LockedDropout_3           -  
12_decoder.mlp.PermuteBlock_0                -  
13_decoder.mlp.BatchNorm1d_1             512.0  
14_decoder.mlp.PermuteBlock_2                -  
15_decoder.mlp.Linear_3              1.048576M  
16_decoder.mlp.PermuteBlock_4                -  
17_decoder.mlp.BatchNorm1d_5            2.048k  
18_decoder.mlp.PermuteBlock_6                -  
19_decoder.mlp.GELU_7                        -  
20_decoder.mlp.Linear_8              4.194304M  
21_decoder.mlp.PermuteBlock_9                -  
22_decoder.mlp.BatchNorm1d_10           2.048k  
23_decoder.mlp.PermuteBlock_11               -  
24_decoder.mlp.GELU_12                       -  
25_decoder.mlp.Dropout_13                    -  
26_decoder.mlp.Linear_14               83.968k  
27_decoder.LogSoftmax_softmax                -  
----------------------------------------------------------------------------------------
                          Totals
Total params          12.717865M
Trainable params      12.717865M
Non-trainable params         0.0
Mult-Adds              54.16448M
========================================================================================
```

## Hyperparameters for Best Score
```
config = {
    "beam_width" : 2,
    "lr" : 1e-3,
    "epochs" : 50,
    "num_layers": 2,
    "factor": 0.5,
    "dropout": 0.2,
    "patience": 2, 
}
```

## Data loading scheme
```
BATCH_SIZE = 64
train_loader = torch.utils.data.DataLoader(
    dataset     = train_data, 
    num_workers = 4,
    batch_size  = BATCH_SIZE, 
    pin_memory  = True,
    shuffle     = True,
    collate_fn  = train_data.collate_fn
)
val_loader = torch.utils.data.DataLoader(
    dataset     = val_data, 
    num_workers = 2,
    batch_size  = BATCH_SIZE,
    pin_memory  = True,
    shuffle     = False,
    collate_fn  = val_data.collate_fn
)
test_loader = torch.utils.data.DataLoader(
    dataset     = test_data, 
    num_workers = 2, 
    batch_size  = BATCH_SIZE, 
    pin_memory  = True, 
    shuffle     = False,
    collate_fn  = test_data.collate_fn
)
```
