### cascade_rcnn.py
> 构建minibatches for cas-rcnn train
- get_cascade_rcnn_blob_names函数：
	- input:
	- output: blob_names
	- 作用：针对网络具体作用指定各blob的name

config.py中有：
cascade fg_thresh
cascade_bg_hi_thresh
......    _lo_thresh
#### Overlap threshold for an RoI to be considered foreground (if >= FG_THRESH)

__C.CASCADE_RCNN.FG_THRESHS = (0.5, 0.6, 0.7)

#### Overlap threshold for an RoI to be considered background (class = 0 if
#### overlap in [LO, HI))

__C.CASCADE_RCNN.BG_THRESHS_HI = (0.5, 0.6, 0.7)
__C.CASCADE_RCNN.BG_THRESHS_LO = (0.0, 0.0, 0.0)

_filter_boxes函数 这里的max_overlap不太懂
core.ScopedBlobReference 干嘛的？？？
distribute_cascade_proposals 中的forward中的训练阶段
