# 3DPyranet 

3DPyranet is a deep pyramidal neural network, based on biological pyramidal neurons developed in [[1]](#1);

## Usage

Check <a href="https://github.com/CVPRLab-UniParthenope/3DPyranet/blob/master/train.py">train.py</a> file 
for the correct pipeline and model usage.
Code uses *Sparse Softmax Cross Entropy* as loss function, it doesn't need *One Hot Encoding*.

### Dependencies

*   Python 3;
*   Tensorflow 1.2+;
*   TQDM;
*   Numpy;
*   Packaging;

### FLAGS

#### Checkpoint and evaluation

|      Option     |   Type  |     Default    |                       Description                      |
|:---------------:|:-------:|:--------------:|:-------------------------------------------------------|
|  evaluate_every |  float  |        1       | Number of epoch for each evaluation (decimals allowed) |
| test_milestones |   list  |    15,25,50    | Each epoch where performs test                         |
| save_checkpoint | boolean |      False     | Flag to save checkpoint or not                         |
| checkpoint_name |  string | 3dpyranet.ckpt | Name of checkpoint file                                   |


#### Input

|       Option      |  Type  | Default |            Description           |
|:-----------------:|:------:|:-------:|:---------------------------------|
|     train_path    | string |    //   | Path to npy training set         |
| train_labels_path | string |    //   | Path to npy training set labels  |
|      val_path     | string |    //   | Path to npy val/test set         |
|  val_labels_path  | string |    //   | Path to npy val/test set labels  |
|     save_path     | string |    //   | Path where to save network model |


#### Input parameters

|    Option    | Type | Default |          Description          |
|:------------:|:----:|:-------:|:------------------------------|
|  batch_size  |  int |   100   | Batch size                    |
| depth_frames |  int |    16   | Number of consecutive samples |
|    height    |  int |   100   | Sample height                 |
|     width    |  int |   100   | Sample width                  |
|  in_channels |  int |    1    | Sample channels               |
|  num_classes |  int |    6    | Number of classes             |


#### Hyper-parameters settings

|     Option    |  Type | Default |                                  Description                                 |
|:-------------:|:-----:|:-------:|:-----------------------------------------------------------------------------|
|  feature_maps |  int  |    3    | Number of maps to use (strict model shares the number of maps in each layer) |
| learning_rate | float | 0.00015 | Learning rate                                                                |
|  decay_steps  |  int  |    15   | Number of epoch for each decay                                               |
|   decay_rate  | float |   0.1   | Learning rate decay                                                          |
|   max_steps   |  int  |    50   | Maximum number of epoch to perform                                           |
|  weight_decay | float |   None  | L2 regularization lambda                                                     |


#### Optimization 

|    Option    |   Type  |  Default |                              Description                              |
|:------------:|:-------:|:--------:|:----------------------------------------------------------------------|
|   optimizer  |  string | MOMENTUM | Optimization algorthim (GD - MOMENTUM - ADAM)                         |
| use_nesterov | boolean |   False  | Flag to use Nesterov Momentum (it works only with MOMENTUM optimizer) |                                            |


## References

<a name="1">[1]</a> Ullah, Ihsan, and Alfredo Petrosino. "Spatiotemporal features learning with 3DPyraNet." International Conference on Advanced Concepts for Intelligent Vision Systems. Springer, Cham, 2016.
