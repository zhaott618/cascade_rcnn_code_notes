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


class DecodeBBoxesOp(object):
    ####该类用于cascade rcnn某阶段输出预测框
    """Output predicted bbox, by Zhaowei Cai for Cascade R-CNN.
    """

    def __init__(self, bbox_reg_weights):
        self._bbox_reg_weights = bbox_reg_weights

    #### 该forward函数用于获取
    def forward(self, inputs, outputs):
        """See modeling.detector.DecodeBBoxes for inputs/outputs
        documentation.
        """
        ###输入是上一阶段的回归参数，以及proposals
        ###bbox_deltas的shape为(num of proposals, 8)
        bbox_deltas = inputs[0].data
        ####必须要是这种类型： CLS_AGNOSTIC_BBOX_REG
        assert cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
        assert bbox_deltas.shape[1] == 8
        bbox_deltas = bbox_deltas[:, -4:]
        bbox_data = inputs[1].data
        ###bbox_data的shape为（num of proposals, 5）
        assert bbox_data.shape[1] == 5
        batch_inds = bbox_data[:, :1]
        bbox_prior = bbox_data[:, 1:]

        # Transform bbox priors into proposals via bbox transformations
        ### 将bbox priors--->bbox predictions或proposals(bbox proposals在cascade 中
        # 即为输入给下一级的proposals)

        bbox_decode = box_utils.bbox_transform(
            bbox_prior, bbox_deltas, self._bbox_reg_weights
        )

        # remove mal-boxes with non-positive width or height and ground
        # truth boxes during training
        ###滤除具有负的宽/高的boxes
        if len(inputs) > 2:
            ###在训练阶段，inputs[2]为gt boxes, 推断时自然没有这一维，
            ### mapped_gt_boxes为某张图片（或某个batch??）的所有与proposals
            # 对应的gt boxes
            mapped_gt_boxes = inputs[2].data
            max_overlap = mapped_gt_boxes[:, 4]
            ###max_overlap什么作用？？？
            keep = _filter_boxes(bbox_decode, max_overlap)
            ####keep保留所有满足要求的bboxes
            bbox_decode = bbox_decode[keep, :]
            batch_inds = batch_inds[keep, :]

        bbox_decode = np.hstack((batch_inds, bbox_decode))
        ###outputs的shape为（1,）
        outputs[0].reshape(bbox_decode.shape)
        outputs[0].data[...] = bbox_decode


def _filter_boxes(boxes, max_overlap):
    """Only keep boxes with positive height and width, and not-gt.
    """
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    ###得到所有满足条件的decode_bboxes所在的行数，这里的max_overlap不太懂
    keep = np.where((ws > 0) & (hs > 0) & (max_overlap < 1.0))[0]

    return keep
