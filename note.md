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

### 3. layers





























