### 1. model load
```python
# config.py,51
config.MODEL.NAME = 'pose_resnet'
```
```python
# train.py,92
model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
    config, is_train=True
) 
```
```python
# pose_resnet.py,312
def get_pose_net(cfg, is_train, **kwargs):
    ...
    model = PoseResNet(block_class, layers, cfg, **kwargs)
    ...
```
=> We have to edit `class PoseResNet()`!


### 2.training with model
for 1 epoch,
```python
    # train.py,169
    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        ...
        # train for one epoch
        train(config, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict) # line 173
        ...

```
```python
# lib/core/function.py,28
def train():
    ...
    output = model(input) # line 44
    target = target.cuda(non_blocking=True)
    target_weight = target_weight.cuda(non_blocking=True)

    loss = criterion(output, target, target_weight)
    ...

```

### 3. 
|           layer_name | output_size                   | layer                                                                                                  |
|---------------------:|-------------------------------|--------------------------------------------------------------------------------------------------------|
|                Input | 256x256x3                     |                                                                                                        |
|                conv1 | 128x128x64                    | 7x7, 64, stride 2, padding 3                                                                           |
|                  bn1 |                               |                                                                                                        |
|                 relu |                               |                                                                                                        |
|              maxpool | 64x64x64                      | 3x3, S:2, P:1                                                                                          |
| conv2_x (layer1) X 3 | 64x64x256                     | 1x1, 64, S:1   3x3, 64, S:1, P:1  1x1, 64→256 ( 1x1, 64→256, S:1 [downsample for input] )              |
|  conv3_x(layer2) X 4 | 32x32x512                     | 1x1, 256→128, S:1 3x3, 128, S:2, P:1 1x1, 128→512 ( 1x1, 256→512, S:2 [downsample for input] )         |
|  conv4_x(layer3) X 6 | 16x16x1024                    | 1x1, 512→256, S:1 3x3, 256, S:2, P:1 1x1, 256→1024 ( 1x1, 512→1024, S:2 [downsample for input] )       |
|  conv5_x(layer4) X 3 | 8x8x2048                      | 1x1, 1024→512, S:1 3x3, 512, S:2, P:1 1x1, 512→2048 ( 1x1, 1024→2048, S:2 [downsample for input] )     |
|               deconv | 16x16x256 32X32X256 64X64X256 | 4x4, 2048→256, S:2, Pin:1, Pout: 0 4x4, 256→256, S:2, Pin:1, Pout: 0 4x4, 256→256, S:2, Pin:1, Pout: 0 |
|                final | 64X64X16                      | 1X1, 256→16, S:1, P:0                                                                                  |





























