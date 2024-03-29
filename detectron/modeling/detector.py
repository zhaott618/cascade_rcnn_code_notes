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

"""Defines DetectionModelHelper, the class that represents a Detectron model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import logging

from caffe2.python import cnn
from caffe2.python import core
from caffe2.python import workspace
from caffe2.python.modeling import initializers
from caffe2.python.modeling.parameter_info import ParameterTags

from detectron.core.config import cfg
from detectron.ops.collect_and_distribute_fpn_rpn_proposals \
    import CollectAndDistributeFpnRpnProposalsOp
from detectron.ops.distribute_cascade_proposals import DistributeCascadeProposalsOp
from detectron.ops.generate_proposal_labels import GenerateProposalLabelsOp
from detectron.ops.generate_proposals import GenerateProposalsOp
from detectron.ops.decode_bboxes import DecodeBBoxesOp
from detectron.ops.bbox_accuracy import BBoxAccuracyOp
import detectron.roi_data.fast_rcnn as fast_rcnn_roi_data
import detectron.roi_data.cascade_rcnn as cascade_rcnn_roi_data
import detectron.utils.c2 as c2_utils

logger = logging.getLogger(__name__)


class DetectionModelHelper(cnn.CNNModelHelper):
    def __init__(self, **kwargs):
        # Handle args specific to the DetectionModelHelper, others pass through
        # to CNNModelHelper
        self.train = kwargs.get('train', False)
        self.num_classes = kwargs.get('num_classes', -1)
        assert self.num_classes > 0, 'num_classes must be > 0'
        for k in ('train', 'num_classes'):
            if k in kwargs:
                del kwargs[k]
        kwargs['order'] = 'NCHW'
        # Defensively set cudnn_exhaustive_search to False in case the default
        # changes in CNNModelHelper. The detection code uses variable size
        # inputs that might not play nicely with cudnn_exhaustive_search.
        kwargs['cudnn_exhaustive_search'] = False
        super(DetectionModelHelper, self).__init__(**kwargs)
        self.roi_data_loader = None
        self.losses = []
        self.metrics = []
        self.do_not_update_params = []  # Param on this list are not updated
        self.net.Proto().type = cfg.MODEL.EXECUTION_TYPE
        self.net.Proto().num_workers = cfg.NUM_GPUS * 4
        self.prev_use_cudnn = self.use_cudnn
        self.gn_params = []  # Param on this list are GroupNorm parameters
        self.stage_params = {}  # Param on this list are updated with scalars

    def TrainableParams(self, gpu_id=-1):
        """Get the blob names for all trainable parameters, possibly filtered by
        GPU id.
        """
        return [
            p for p in self.params
            if (
                p in self.param_to_grad and   # p has a gradient
                p not in self.do_not_update_params and  # not on the blacklist
                (gpu_id == -1 or  # filter for gpu assignment, if gpu_id set
                 str(p).find('gpu_{}'.format(gpu_id)) == 0)
            )]

    def AffineChannel(self, blob_in, blob_out, dim, inplace=False):
        ###  替代BN的affine transformation(因为此时minibatch尺寸太小)
        ###  可用于节省内存
        """Affine transformation to replace BN in networks where BN cannot be
        used (e.g., because the minibatch size is too small).

        The operations can be done in place to save memory.
        """
        blob_out = blob_out or self.net.NextName()
        param_prefix = blob_out

        scale = self.create_param(
            param_name=param_prefix + '_s',
            initializer=initializers.Initializer("ConstantFill", value=1.),
            tags=ParameterTags.WEIGHT,
            shape=[dim, ],
        )
        bias = self.create_param(
            param_name=param_prefix + '_b',
            initializer=initializers.Initializer("ConstantFill", value=0.),
            tags=ParameterTags.BIAS,
            shape=[dim, ],
        )
        if inplace:
            return self.net.AffineChannel([blob_in, scale, bias], blob_in)
        else:
            return self.net.AffineChannel([blob_in, scale, bias], blob_out)

    #### 这个函数相当相当重要，它相当于“极简faster rcnn实现中所描述的proposalcreator”
    #### 从20000个anchors选择2000（train）或300（inference）个作为proposals输入fast rcnn
    def GenerateProposals(self, blobs_in, blobs_out, anchors, spatial_scale):
        """Op for generating RPN porposals.

        blobs_in:

            ####'rpn_cls_probs': 4D tensor of shape (N, A, H, W)，每个anchor
            ####对应一个置信度
          - 'rpn_cls_probs': 4D tensor of shape (N, A, H, W), where N is the
            number of minibatch images, A is the number of anchors per
            locations, and (H, W) is the spatial size of the prediction grid.
            Each value represents a "probability of object" rating in [0, 1].

            #### 'rpn_bbox_pred': 4D tensor of shape (N, 4 * A, H, W)为bbox回归参数
            #### 每个anchor对应4个值
          - 'rpn_bbox_pred': 4D tensor of shape (N, 4 * A, H, W) of predicted
            deltas for transformation anchor boxes into RPN proposals.

            #### im_info的shape为(N, 3)代表每张输入图片的 [height, width, scale]，scale用来
            #### 将原图尺寸变换到网络输入尺寸
          - 'im_info': 2D tensor of shape (N, 3) where the three columns encode
            the input image's [height, width, scale]. Height and width are
            for the input to the network, not the original image; scale is the
            scale factor used to scale the original image to the network input
            size.

        blobs_out:
            ####rpn_rois中bbox的尺寸是相对于网络输入（即经过scaled的原图尺寸），因此
            #### proposals尺寸要乘上scale以恢复到原图尺寸
          - 'rpn_rois': 2D tensor of shape (R, 5), for R RPN proposals where the
            five columns encode [batch ind, x1, y1, x2, y2]. The boxes are
            w.r.t. the network input, which is a *scaled* version of the
            original image; these proposals must be scaled by 1 / scale (where
            scale comes from im_info; see above) to transform it back to the
            original input image coordinate system.
            ####rpn_roi_probs为置信度
          - 'rpn_roi_probs': 1D tensor of objectness probability scores
            (extracted from rpn_cls_probs; see above).
        """
        name = 'GenerateProposalsOp:' + ','.join([str(b) for b in blobs_in])
        # spatial_scale passed to the Python op is only used in convert_pkl_to_pb

        #### 网络forward
        self.net.Python(
            GenerateProposalsOp(anchors, spatial_scale, self.train).forward
        )(blobs_in, blobs_out, name=name, spatial_scale=spatial_scale)

        return blobs_out

    def GenerateProposalLabels(self, blobs_in):
        #### 该函数用于为rpn的proposals产生训练标签，用于rpn与fast rcnn端到端训练
        """Op for generating training labels for RPN proposals. This is used
        when training RPN jointly with Fast/Mask R-CNN (as in end-to-end
        Faster R-CNN training).

        ####网络输入：
        blobs_in:
          - 'rpn_rois': 2D tensor of RPN proposals output by GenerateProposals
          - 'roidb': roidb entries that will be labeled
          - 'im_info': See GenerateProposals doc.

        blobs_out:
          - (variable set of blobs): returns whatever blobs are required for
            training the model. It does this by querying the data loader for
            the list of blobs that are needed.
        """
        name = 'GenerateProposalLabelsOp:' + ','.join(
            [str(b) for b in blobs_in]
        )

        # The list of blobs is not known before run-time because it depends on
        # the specific model being trained. Query the data loader to get the
        # list of output blob names.
        blobs_out = fast_rcnn_roi_data.get_fast_rcnn_blob_names(
            is_training=self.train
        )
        blobs_out = [core.ScopedBlobReference(b) for b in blobs_out]

        self.net.Python(GenerateProposalLabelsOp().forward)(
            blobs_in, blobs_out, name=name
        )
        return blobs_out

    ### 这是一个核心函数，用于merge在多个fpn lvls产生的rpn proposals并将这些proposals
    ### 分配到它们所对应的合适的fpn lvls上
    ### 在某一个fpn lvl的一个anchor有可能会预测出一个映射到其他lvl上的ROI，因此需要对proposals
    ### 进行重新分配
    def CollectAndDistributeFpnRpnProposals(self):
        """Merge RPN proposals generated at multiple FPN levels and then
        distribute those proposals to their appropriate FPN levels. An anchor
        at one FPN level may predict an RoI that will map to another level,
        hence the need to redistribute the proposals.

        This function assumes standard blob names for input and output blobs.

        ####输入blobs为各个lvl的rpn产生的rois以及对应的概率（即每个ROI包含物体的置信度）
        Input blobs: [rpn_rois_fpn<min>, ..., rpn_rois_fpn<max>,
                      rpn_roi_probs_fpn<min>, ..., rpn_roi_probs_fpn<max>]
          - rpn_rois_fpn<i> are the RPN proposals for FPN level i; see rpn_rois
            documentation from GenerateProposals.

          - rpn_roi_probs_fpn<i> are the RPN objectness probabilities for FPN
            level i; see rpn_roi_probs documentation from GenerateProposals.

        #### 训练阶段input_blob 还会包含 [roidb, im_info]
        If used during training, then the input blobs will also include:
          [roidb, im_info] (see GenerateProposalLabels).

        #### 输出的blobs为
        Output blobs: [rois_fpn<min>, ..., rois_rpn<max>, rois,
                       rois_idx_restore]
          - rois_fpn<i> are the RPN proposals for FPN level i
          - rois_idx_restore is a permutation on the concatenation of all
            rois_fpn<i>, i=min...max, such that when applied the RPN RoIs are
            restored to their original order in the input blobs.

        If used during training, then the output blobs will also include:
          [labels, bbox_targets, bbox_inside_weights, bbox_outside_weights].
        """
        k_max = cfg.FPN.RPN_MAX_LEVEL
        k_min = cfg.FPN.RPN_MIN_LEVEL

        # Prepare input blobs
        rois_names = ['rpn_rois_fpn' + str(l) for l in range(k_min, k_max + 1)]
        score_names = [
            'rpn_roi_probs_fpn' + str(l) for l in range(k_min, k_max + 1)
        ]
        ### 准备输入数据
        blobs_in = rois_names + score_names
        if self.train:
            blobs_in += ['roidb', 'im_info']
        ### ScopedBlobReference不知道干嘛的，blobs_in为经过处理的每一个输入元素
        ### 构成的集合
        blobs_in = [core.ScopedBlobReference(b) for b in blobs_in]
        name = 'CollectAndDistributeFpnRpnProposalsOp:' + ','.join(
            [str(b) for b in blobs_in]
        )

        # Prepare output blobs
        ### 准备输出数据blobs
        blobs_out = fast_rcnn_roi_data.get_fast_rcnn_blob_names(
            is_training=self.train
        )
        blobs_out = [core.ScopedBlobReference(b) for b in blobs_out]

        ### 根据输入和输出数据进行网络前向传播
        outputs = self.net.Python(
            CollectAndDistributeFpnRpnProposalsOp(self.train).forward
        )(blobs_in, blobs_out, name=name)


        return outputs


    #### 此函数用于调整某阶段网络输出的proposals，使用网络回归的参数值
    def DecodeBBoxes(self, blobs_in, blobs_out, bbox_reg_weights):
        """Op for decoding bboxes. Only support class-agnostic bbox regression.
        by Zhaowei Cai for Cascade R-CNN

        blobs_in:
            ####上一个阶段得到的回归参数，作为blobs[0]，用于将上一个阶段的预测框
            ####调整后输入下一个阶段
          - 'bbox_pred_<j>': 2D tensor of shape (R, 4 * 2) of predicted deltas
            for transformation previous boxes into next boxes, at stage j.
             #### 下面这个参数代表某一阶段的输入的所有proposals, 以及batch_inds
             #### 作为blobs[1]
          - 'rois_<j>': 2D tensor of shape (R, 5), for proposals where the
            five columns encode [batch ind, x1, y1, x2, y2], at stage j.

        #### 训练阶段还会包含mapped_gt_boxes, 用于移除多余的gt框
        If used during training, then the input blobs will also include:
          [mapped_gt_boxes_<j>], which is used to remove redundant ground truth.

        #### 输出是下一阶段的proposals
        blobs_out:
          - 'proposals_<j+1>': 2D tensor of shape (R, 5), for proposals where the
            five columns encode [batch ind, x1, y1, x2, y2].
        """
        name = 'DecodeBBoxesOp:' + ','.join([str(b) for b in blobs_in])
        self.net.Python(DecodeBBoxesOp(bbox_reg_weights).forward)(
            blobs_in, blobs_out, name=name
        )
        return blobs_out

    def DistributeCascadeProposals(self, stage):
        ###用于将proposals分配给各level, 若采用fpn架构的话
        """Distribute proposals to their appropriate FPN levels.
        by Zhaowei Cai for Cascade R-CNN

        Input blobs:
        ###输入为第j个stage所输出的调整过的proposals
          - proposals_<j> are the decoded proposals from stage j; see
            documentation from DecodeBBoxes.

        ###若在训练阶段使用，还会有以下输入blobs: [roidb, im_info]
        If used during training, then the input blobs will also include:
          [roidb, im_info] (see GenerateProposalLabels).


        Output blobs: [rois_fpn<min>, ..., rois_rpn<max>, rois,
                       rois_idx_restore]
        ###输出为：每一个 fpn lvl的proposals, 所有proposals的一个排列(为了
        ###保留这些proposals在input blob中的初始顺序？？)
          - rois_fpn<i> are the RPN proposals for FPN level i
          - rois_idx_restore is a permutation on the concatenation of all
            rois_fpn<i>, i=min...max, such that when applied the RPN RoIs are
            restored to their original order in the input blobs.

        ###训练阶段还包含：gt框的相关信息，其中bbox_inside_weights表示回归损失公式
        ###中的Pi*, 即bbox_inside_weights==1, bbox_outside_weights==0即只对正样本
        ###计算回归损失
        If used during training, then the output blobs will also include:
          [labels, bbox_targets, bbox_inside_weights, bbox_outside_weights,
          mapped_gt_boxes].
        """
        stage_name = '_{}'.format(stage)

        # Prepare input blobs
        blobs_in = ['proposals' + stage_name]
        if self.train:
            blobs_in += ['roidb', 'im_info']
        ####下面这一句是干嘛的？？？
        blobs_in = [core.ScopedBlobReference(b) for b in blobs_in]
        name = 'DistributeCascadeProposalsOp:' + ','.join(
            [str(b) for b in blobs_in]
        )

        # Prepare output blobs
        ##准备输出数据blob容器
        blobs_out = cascade_rcnn_roi_data.get_cascade_rcnn_blob_names(
            stage, is_training=self.train
        )
        ###又是这个函数
        blobs_out = [core.ScopedBlobReference(b) for b in blobs_out]

        outputs = self.net.Python(
            DistributeCascadeProposalsOp(self.train, stage).forward
        )(blobs_in, blobs_out, name=name)

        return outputs

    def DropoutIfTraining(self, blob_in, dropout_rate):
        ####增加dropout作用
        """Add dropout to blob_in if the model is in training mode and
        dropout_rate is > 0."""
        blob_out = blob_in
        if self.train and dropout_rate > 0:
            blob_out = self.Dropout(
                blob_in, blob_in, ratio=dropout_rate, is_test=False
            )
        return blob_out

    ###选择ROIpooling还是roialign等方式
    def RoIFeatureTransform(
        self,
        blobs_in,
        blob_out,
        blob_rois='rois',
        method='RoIPoolF',
        resolution=7,
        spatial_scale=1. / 16.,
        sampling_ratio=0
    ):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.

        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIPoolF', 'RoIAlign'}, \
            'Unknown pooling method: {}'.format(method)
        has_argmax = (method == 'RoIPoolF')
        if isinstance(blobs_in, list):
            # FPN case: add RoIFeatureTransform to each FPN level
            k_max = cfg.FPN.ROI_MAX_LEVEL  # coarsest level of pyramid
            k_min = cfg.FPN.ROI_MIN_LEVEL  # finest level of pyramid
            assert len(blobs_in) == k_max - k_min + 1
            bl_out_list = []
            for lvl in range(k_min, k_max + 1):
                bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
                sc = spatial_scale[k_max - lvl]  # in reversed order
                bl_rois = blob_rois + '_fpn' + str(lvl)
                bl_out = blob_out + '_fpn' + str(lvl)
                bl_out_list.append(bl_out)
                bl_argmax = ['_argmax_' + bl_out] if has_argmax else []
                self.net.__getattr__(method)(
                    [bl_in, bl_rois], [bl_out] + bl_argmax,
                    pooled_w=resolution,
                    pooled_h=resolution,
                    spatial_scale=sc,
                    sampling_ratio=sampling_ratio
                )
            # The pooled features from all levels are concatenated along the
            # batch dimension into a single 4D tensor.
            xform_shuffled, _ = self.net.Concat(
                bl_out_list, [blob_out + '_shuffled', '_concat_' + blob_out],
                axis=0
            )
            # Unshuffle to match rois from dataloader
            restore_bl = blob_rois + '_idx_restore_int32'
            xform_out = self.net.BatchPermutation(
                [xform_shuffled, restore_bl], blob_out
            )
        else:
            # Single feature level
            bl_argmax = ['_argmax_' + blob_out] if has_argmax else []
            # sampling_ratio is ignored for RoIPoolF
            xform_out = self.net.__getattr__(method)(
                [blobs_in, blob_rois], [blob_out] + bl_argmax,
                pooled_w=resolution,
                pooled_h=resolution,
                spatial_scale=spatial_scale,
                sampling_ratio=sampling_ratio
            )
        # Only return the first blob (the transformed features)
        return xform_out[0] if isinstance(xform_out, tuple) else xform_out

    def ConvShared(
        self,
        blob_in,
        blob_out,
        dim_in,
        dim_out,
        kernel,
        weight=None,
        bias=None,
        **kwargs
    ):
        """Add conv op that shares weights and/or biases with another conv op.
        """
        use_bias = (
            False if ('no_bias' in kwargs and kwargs['no_bias']) else True
        )

        if self.use_cudnn:
            kwargs['engine'] = 'CUDNN'
            kwargs['exhaustive_search'] = self.cudnn_exhaustive_search
            if self.ws_nbytes_limit:
                kwargs['ws_nbytes_limit'] = self.ws_nbytes_limit

        if use_bias:
            blobs_in = [blob_in, weight, bias]
        else:
            blobs_in = [blob_in, weight]

        if 'no_bias' in kwargs:
            del kwargs['no_bias']

        return self.net.Conv(
            blobs_in, blob_out, kernel=kernel, order=self.order, **kwargs
        )

    def FCShared(
        self,
        blob_in,
        blob_out,
        weight=None,
        bias=None,
        **kwargs
    ):
        """Add fc op that shares weights and/or biases with another fc op.
        """
        use_bias = (
            False if ('no_bias' in kwargs and kwargs['no_bias']) else True
        )

        if self.use_cudnn:
            kwargs['engine'] = 'CUDNN'
            kwargs['exhaustive_search'] = self.cudnn_exhaustive_search
            if self.ws_nbytes_limit:
                kwargs['ws_nbytes_limit'] = self.ws_nbytes_limit

        if use_bias:
            blobs_in = [blob_in, weight, bias]
        else:
            blobs_in = [blob_in, weight]

        if 'no_bias' in kwargs:
            del kwargs['no_bias']

        return self.net.FC(
            blobs_in, blob_out, order=self.order, **kwargs
        )

    ###双线性插值函数
    def BilinearInterpolation(
        self, blob_in, blob_out, dim_in, dim_out, up_scale
    ):
        """Bilinear interpolation in space of scale.
        ####双线性差值，可用于上采样
        Takes input of NxKxHxW and outputs NxKx(sH)x(sW), where s:= up_scale

        Adapted from the CVPR'15 FCN code.
        See: https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
        """
        assert dim_in == dim_out
        assert up_scale % 2 == 0, 'Scale should be even'

        def upsample_filt(size):
            factor = (size + 1) // 2
            if size % 2 == 1:
                center = factor - 1
            else:
                center = factor - 0.5
            og = np.ogrid[:size, :size]
            return ((1 - abs(og[0] - center) / factor) *
                    (1 - abs(og[1] - center) / factor))

        kernel_size = up_scale * 2
        bil_filt = upsample_filt(kernel_size)

        kernel = np.zeros(
            (dim_in, dim_out, kernel_size, kernel_size), dtype=np.float32
        )
        kernel[range(dim_out), range(dim_in), :, :] = bil_filt

        blob = self.ConvTranspose(
            blob_in,
            blob_out,
            dim_in,
            dim_out,
            kernel_size,
            stride=int(up_scale),
            pad=int(up_scale / 2),
            weight_init=('GivenTensorFill', {'values': kernel}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        self.do_not_update_params.append(self.weights[-1])
        self.do_not_update_params.append(self.biases[-1])
        return blob

    ####convaffine意味着在affinechannel（用于在网络fine tune时代替BN）后增加一个卷积操作
    def ConvAffine(  # args in the same order of Conv()
        self, blob_in, prefix, dim_in, dim_out, kernel, stride, pad,
        group=1, dilation=1,
        weight_init=None,
        bias_init=None,
        suffix='_bn',
        inplace=False
    ):
        """ConvAffine adds a Conv op followed by a AffineChannel op (which
        replaces BN during fine tuning).
        """
        conv_blob = self.Conv(
            blob_in,
            prefix,
            dim_in,
            dim_out,
            kernel,
            stride=stride,
            pad=pad,
            group=group,
            ###指定是否为空洞卷积
            dilation=dilation,
            weight_init=weight_init,
            bias_init=bias_init,
            no_bias=1
        )
        ### 进行AffineChannel
        blob_out = self.AffineChannel(
            conv_blob, prefix + suffix, dim=dim_out, inplace=inplace
        )
        return blob_out

    ###用于在GN操作后添加卷积操作（包含可学习的参数：scale/bias (gamma/beta)）
    def ConvGN(  # args in the same order of Conv()
        self, blob_in, prefix, dim_in, dim_out, kernel, stride, pad,
        group_gn,  # num of groups in gn
        group=1, dilation=1,
        weight_init=None,
        bias_init=None,
        suffix='_gn',
        no_conv_bias=1,
    ):
        """ConvGN adds a Conv op followed by a GroupNorm op,
        including learnable scale/bias (gamma/beta)
        """
        conv_blob = self.Conv(
            blob_in,
            prefix,
            dim_in,
            dim_out,
            kernel,
            stride=stride,
            pad=pad,
            group=group,
            dilation=dilation,
            weight_init=weight_init,
            bias_init=bias_init,
            no_bias=no_conv_bias)

        if group_gn < 1:
            logger.warning(
                'Layer: {} (dim {}): '
                'group_gn < 1; reset to 1.'.format(prefix, dim_in)
            )
            group_gn = 1

        blob_out = self.SpatialGN(
            conv_blob, prefix + suffix,
            dim_out, group=group_gn,  # op's arg name is "group"
            epsilon=cfg.GROUP_NORM.EPSILON,)

        self.gn_params.append(self.params[-1])  # add gn's bias to list
        self.gn_params.append(self.params[-2])  # add gn's scale to list
        return blob_out

    ####用于共享权重and/or偏置的gn op
    def SpatialGNShared(
        self,
        blob_in,
        blob_out,
        group_gn,
        scale=None,
        bias=None,
        **kwargs
    ):
        """Add gn op that shares weights and/or biases with another gn op.
        """
        use_bias = (
            False if ('no_bias' in kwargs and kwargs['no_bias']) else True
        )

        if self.use_cudnn:
            kwargs['engine'] = 'CUDNN'
            kwargs['exhaustive_search'] = self.cudnn_exhaustive_search
            if self.ws_nbytes_limit:
                kwargs['ws_nbytes_limit'] = self.ws_nbytes_limit

        if use_bias:
            blobs_in = [blob_in, scale, bias]
        else:
            blobs_in = [blob_in, scale]

        blobs_out = [blob_out, blob_out + "_mean", blob_out + "_std"]

        if 'no_bias' in kwargs:
            del kwargs['no_bias']

        kwargs['group'] = group_gn
        kwargs['epsilon'] = cfg.GROUP_NORM.EPSILON

        blob_outputs = self.net.GroupNorm(
            blobs_in, blobs_out, **kwargs
        )
        return blob_outputs[0]

    def DisableCudnn(self):
        self.prev_use_cudnn = self.use_cudnn
        self.use_cudnn = False

    def RestorePreviousUseCudnn(self):
        prev_use_cudnn = self.use_cudnn
        self.use_cudnn = self.prev_use_cudnn
        self.prev_use_cudnn = prev_use_cudnn

    def UpdateWorkspaceLr(self, cur_iter, new_lr):
        """Updates the model's current learning rate and the workspace (learning
        rate and update history/momentum blobs).
        """
        # The workspace is the one source of truth for the lr
        # The lr is always the same on all GPUs
        cur_lr = workspace.FetchBlob('gpu_0/lr')[0]
        # There are no type conversions between the lr in Python and the lr in
        # the GPU (both are float32), so exact comparision is ok
        if cur_lr != new_lr:
            ratio = _get_lr_change_ratio(cur_lr, new_lr)
            if ratio > cfg.SOLVER.LOG_LR_CHANGE_THRESHOLD:
                logger.info(
                    'Changing learning rate {:.6f} -> {:.6f} at iter {:d}'.
                    format(cur_lr, new_lr, cur_iter))
            self._SetNewLr(cur_lr, new_lr)
        return new_lr

    def _SetNewLr(self, cur_lr, new_lr):
        """Do the actual work of updating the model and workspace blobs.
        """
        for i in range(cfg.NUM_GPUS):
            with c2_utils.CudaScope(i):
                workspace.FeedBlob(
                    'gpu_{}/lr'.format(i), np.array([new_lr], dtype=np.float32))
        ratio = _get_lr_change_ratio(cur_lr, new_lr)
        if cfg.SOLVER.SCALE_MOMENTUM and cur_lr > 1e-7 and \
                ratio > cfg.SOLVER.SCALE_MOMENTUM_THRESHOLD:
            self._CorrectMomentum(new_lr / cur_lr)

    ####采用动量法更新参数
    def _CorrectMomentum(self, correction):
        """The MomentumSGDUpdate op implements the update V as

            V := mu * V + lr * grad,

        where mu is the momentum factor, lr is the learning rate, and grad is
        the stochastic gradient. Since V is not defined independently of the
        learning rate (as it should ideally be), when the learning rate is
        changed we should scale the update history V in order to make it
        compatible in scale with lr * grad.
        """
        logger.info(
            'Scaling update history by {:.6f} (new lr / old lr)'.
            format(correction))
        for i in range(cfg.NUM_GPUS):
            with c2_utils.CudaScope(i):
                for param in self.TrainableParams(gpu_id=i):
                    op = core.CreateOperator(
                        'Scale', [param + '_momentum'], [param + '_momentum'],
                        scale=correction)
                    workspace.RunOperatorOnce(op)

    ####神奇的函数，用来在distributed data parallel setting中使用？？？
    def GetLossScale(self):
        """Allow a way to configure the loss scale dynamically.

        This may be used in a distributed data parallel setting.
        """
        return 1.0 / cfg.NUM_GPUS

    def AddLosses(self, losses):
        ###isinstance函数表示losses是否是list的一个类或子类
        if not isinstance(losses, list):
            losses = [losses]
        # Conversion to str allows losses to include BlobReferences
        ####BlobReferences和unscopename这两项始终没懂
        ####答案--->UnscopeName函数作用是把原变量去除scope，在这里
        ####即是把losses中的每一项都去除scope
        losses = [c2_utils.UnscopeName(str(l)) for l in losses]
        ####为什么要self.losses+losses ????
        self.losses = list(set(self.losses + losses))

    ####为模型增加指标
    def AddMetrics(self, metrics):
        ###
        if not isinstance(metrics, list):
            metrics = [metrics]
        ####为什么要在这里使用加？？？
        self.metrics = list(set(self.metrics + metrics))

    ####终于到了激动人心的时刻---->针对cascade-rcnn的op！！！
    def AddBBoxAccuracy(self, blobs_in, blobs_out, bbox_reg_weights):
        """Op for bbox IoU accuracy, by Zhaowei Cai for Cascade R-CNN.
        函数输入：
          - 预测的回归参数，shape为（R, 4*类别数），R为proposals数，即对每一个
            proposals都预测其对于每一个类别的回归参数
          - rois即proposals, shape为（R, 5）

    ????? - labels:这个标签究竟指gt还是预测的proposals的标签？ 答案--->指gt bbox

          - 这R个proposals对应的R个gt bbox, shape为（R, 5）
        blobs_in: ['bbox_pred', 'rois', 'labels', 'mapped_gt_boxes']
          - 'bbox_pred': 2D tensor of shape (R, 4 * C), predicted bbox deltas
            for transformation previous boxes into next boxes.
          - 'rois': 2D tensor of shape (R, 5), proposals where the
            five columns encode [batch ind, x1, y1, x2, y2].
          - 'labels': 2D tensor of shape (R, 1), classification labels to
            identify fg rois.
          - 'mapped_gt_boxes': 2D tensor of shape (R, 5), the corresponding gt
            boxes where the five columns encode [x1, y1, x2, y2, IoU].

        函数输出：
          - bbox预测之后的平均IOU
          - bbox预测之前的平均IOU
        blobs_out:
          - 'bbox_iou': mean IoU after bbox prediction.
          - 'bbox_iou_pre': mean IoU before bbox prediction.
        """
        ####bbox_reg_weights为bbox回归权重
        name = 'BBoxAccuracyOp:' + ','.join([str(b) for b in blobs_in])
        self.net.Python(BBoxAccuracyOp(bbox_reg_weights).forward)(
            blobs_in, blobs_out, name=name
        )
        return blobs_out


def _get_lr_change_ratio(cur_lr, new_lr):
    eps = 1e-10
    ratio = np.max(
        (new_lr / np.max((cur_lr, eps)), cur_lr / np.max((new_lr, eps)))
    )
    return ratio
