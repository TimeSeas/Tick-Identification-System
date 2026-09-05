# Full MMDetection config for tick detection with Mask R-CNN R50 FPN

# Model settings - COCO pretrained ResNet50 FPN
model = dict(
    type='MaskRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32
    ),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5
    ),
    rpn_head=dict(
        type='RPNHead',
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]
        ),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True),
        loss_bbox=dict(type='L1Loss')
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,  # tick only
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, reduction='mean'),
            loss_bbox=dict(type='L1Loss', reduction='mean')
        ),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1  # tick only
        )
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.7, neg_iou_thr=0.3, topk=9),
            sampler=dict(type='RandomSampler', num=256, pos_fraction=0.5, add_gt_as_proposals=True)
        ),
        rpn_proposal=dict(nms_thr=0.7, max_num=2000),
        rcnn=dict(
            sampler=dict(type='RandomSampler', num=512, pos_fraction=0.25, add_gt_as_proposals=True)
        )
    ),
    test_cfg=dict(
        rpn=dict(nms_thr=0.7, max_num=2000),
        rcnn=dict(score_thr=0.05, mask_thr=0.5, max_per_img=100, nms=dict(type='nms', iou_threshold=0.5))
    ),
    init_cfg=None  # Will load pretrained weights
)

# Dataset settings
dataset_type = 'CocoDataset'
data_root = 'ticks image detection.v5i.coco/'

# Pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PackDetInputs')
]

# Dataloaders
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/'),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/'),
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

# Evaluation
val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'valid/_annotations.coco.json')
test_evaluator = dict(type='CocoMetric', ann_file=data_root + 'test/_annotations.coco.json')

# Training settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001))

# Learning rate schedule
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=5, end=20, by_epoch=True, milestones=[10, 15], gamma=0.1)
]

# Hooks
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=5, save_best='auto'),
    logger=dict(type='LoggerHook', interval=10)
)

# Runtime
default_scope = 'mmdet'
work_dir = 'output/mask_rcnn_mmdet'
launcher = 'none'
log_processor = dict(type='LogProcessor', window_size=50)
log_level = 'INFO'
resume = False