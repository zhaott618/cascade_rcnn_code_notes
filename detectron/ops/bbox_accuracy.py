# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

from detectron.core.config import cfg
import detectron.utils.boxes as box_utils

#### 又是一个核心类及其函数： BBoxAccuracyOp， 用于输出bbox的预测的IOU的精度
class BBoxAccuracyOp(object):
    """Output bbox prediction IoU accuracy, by Zhaowei Cai.
    """

    def __init__(self, bbox_reg_weights):
        self._bbox_reg_weights = bbox_reg_weights

    def forward(self, inputs, outputs):
        """See modeling.detector.AddBBoxAccuracy for inputs/outputs
        documentation.
        """

        # predicted bbox deltas, shape为（R, C*4）
        bbox_deltas = inputs[0].data
        # proposals的坐标集合, shape为（R, 5）
        bbox_data = inputs[1].data
        assert bbox_data.shape[1] == 5
        ### bbox_prior为所有的proposals坐标, shape为(R, 4)
        bbox_prior = bbox_data[:, 1:]

        # labels
        labels = inputs[2].data

        # mapped gt boxes
        mapped_gt_boxes = inputs[3].data
        gt_boxes = mapped_gt_boxes[:, :4]
        max_overlap = mapped_gt_boxes[:, 4]

        # bbox iou only for fg and non-gt boxes
        ###这里的labels指的是mapped_gt_bbox对应的labels吧？？？
        ###同时一移除所有的gt boxes
        ###相当于对这些gt bbox或proposals进行筛选
        keep_inds = np.where((labels > 0) & (max_overlap < 1.0))[0]
        ###所有符合要求的proposals个数
        num_boxes = keep_inds.size
        bbox_deltas = bbox_deltas[keep_inds, :]
        bbox_prior = bbox_prior[keep_inds, :]
        labels = labels[keep_inds]
        gt_boxes = gt_boxes[keep_inds, :]
        max_overlap = max_overlap[keep_inds]

        ### 关于AGNOSTIC_BBOX_REG 这个什么意思我始终云里雾里
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG or num_boxes == 0:

            bbox_deltas = bbox_deltas[:, -4:]
        else:
            ### 将bbox_deltas的数据结构重新组织，即只保留bbox_deltas中
            ### 每一组回归参数对应的类别和labels（对应的gt真值）类别相同的回归参数
            ### 处理后的bbox_deltas的shape为(num_boxes, 4)
            bbox_deltas = np.vstack(
                [
                    bbox_deltas[i, labels[i] * 4: labels[i] * 4 + 4]
                    for i in range(num_boxes)
                ]
            )

        ### 通过bbox_transform函数将得到的proposals经过回归参数回归后
        ### 得到预测框predicted_bboxes，注意_bbox_reg_weights
        pred_boxes = box_utils.bbox_transform(
            bbox_prior, bbox_deltas, self._bbox_reg_weights
        )

        #####平均iou初值为0
        avg_iou = 0.
        pre_avg_iou = sum(max_overlap)
        for i in range(num_boxes):
            ###第i个gt_box（对应于第i个pred_bbox）的坐标值
            gt_box = gt_boxes[i, :]
            ###第i个pred_box的坐标值
            pred_box = pred_boxes[i, :]
            ###计算gt_box与pred_box之间的IOU
            tmp_iou = box_utils.bbox_overlaps(
                gt_box [np.newaxis, :].astype(dtype=np.float32, copy=False),
                pred_box[np.newaxis, :].astype(dtype=np.float32, copy=False),
            )
            avg_iou += tmp_iou[0]
        if num_boxes > 0:
            avg_iou /= num_boxes
            pre_avg_iou /= num_boxes
        ### 即outputs【0】--->本stage的avg_iou
        ###  outputs[1]----->上一个stage的avg_iou
        outputs[0].reshape([1])
        outputs[0].data[...] = avg_iou
        outputs[1].reshape([1])
        outputs[1].data[...] = pre_avg_iou
