from cfg.info import *

log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=50)
evaluation = dict(interval=50,
                  metric=['PCK', ],
                  key_indicator='mPCK',
                  gpu_collect=True,
                  res_folder='')
optimizer = dict(type='Adam', lr=5e-5, )

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=1000,
                 warmup_ratio=0.001,
                 step=[160, 180])
total_epochs = 200
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'), ])

channel_cfg = dict(num_output_channels=1,
                   dataset_joints=1,
                   dataset_channel=[[0, ], ],
                   inference_channel=[0, ],
                   max_kpt_num=70)

# model settings
model = dict(type='MetaPointFormer',  # DMetaFormer
             encoder_config=dict(type='ResNet', depth=50, out_indices=(0, 1, 2, 3,)),
             archicfg=dict(
                 enhance_sk_w_meta='cat-qs',  # add-qs, cat-qs, off
                 enhance_sk_on_s=False,
                 enhance_sk_on_q=True,
                 res_layer_num=3,
                 skfeat_layer_num=1,
                 skfeat_layer_fusion='cat',
             ),
             comdecodercfg=dict(
                 type='MP',
                 ref_feat=True,
                 self_att=True,
                 iden_pos=True,
                 coor_pos=True,
                 meta_init=True,
                 n_heads=4,
                 d_ffn=256,
                 n_points=4,
                 n_mlp=3,
             ),
             metadecodercfg=dict(
                 in_layer_num=3,
                 out_layer_num=3,
             ),
             refdecodercfg=dict(
                 in_layer_num=3,
                 out_layer_num=3,
             ),
             enhancecfg=dict(d_model=256,
                             nhead=8,
                             num_encoder_layers=3,
                             dim_feedforward=2048,
                             dropout=0.1,
                             activation="relu",
                             normalize_before=False),
             metacfg=dict(
                 point_num=100,
                 point_dim=256,
                 asm_type='nbipart',  # nbipart, bipart, fixed
             ),
             hypercfg=dict(
                 l1=1.,
                 vp=0.1,
                 vn=0.005,
                 meta=0.1,
                 ref=1.,
                 aux=0.1,
             ),
             pretrained='torchvision://resnet50',
             # training and testing settings
             train_cfg=dict(),
             test_cfg=dict(flip_test=False,
                           post_process='default',
                           shift_heatmap=True,
                           modulate_kernel=11))

data_cfg = dict(image_size=[256, 256], heatmap_size=[64, 64],
                num_output_channels=channel_cfg['num_output_channels'],
                num_joints=channel_cfg['dataset_joints'],
                dataset_channel=channel_cfg['dataset_channel'],
                inference_channel=channel_cfg['inference_channel'])

train_pipeline = [dict(type='LoadImageFromFile'),
                  dict(type='TopDownGetRandomScaleRotation', rot_factor=30, scale_factor=0.30),
                  dict(type='TopDownAffineFewShot'), dict(type='ToTensor'),
                  dict(type='NormalizeTensor', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                  dict(type='TopDownGenerateTargetFewShot', sigma=2),
                  dict(type='Collect', keys=['img', 'target', 'target_weight'],
                       meta_keys=['image_file', 'joints_3d', 'joints_3d_visible',
                                  'center', 'scale', 'rotation', 'bbox_score', 'flip_pairs', 'category_id'])]

valid_pipeline = [dict(type='LoadImageFromFile'), dict(type='TopDownAffineFewShot'), dict(type='ToTensor'),
                  dict(type='NormalizeTensor', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                  dict(type='TopDownGenerateTargetFewShot', sigma=2),
                  dict(type='Collect', keys=['img', 'target', 'target_weight'],
                       meta_keys=['image_file', 'joints_3d',
                                  'center', 'scale', 'rotation', 'bbox_score', 'flip_pairs', 'category_id']), ]

test_pipeline = valid_pipeline
viz_pipeline = valid_pipeline

data_root = 'data/mp100'
unified_ann_file = f'{data_root}/annotations/unified_ann_file.json'
unified_img_prefix = f'{data_root}/mp100_images_cc'
data = dict(samples_per_gpu=24, workers_per_gpu=12,
            train=dict(type='TransformerPoseDataset',
                       ann_file=unified_ann_file,
                       img_prefix=unified_img_prefix,
                       data_cfg=data_cfg,
                       valid_class_ids=train_cids_split1,
                       max_kpt_num=channel_cfg['max_kpt_num'],
                       num_shots=1,
                       pipeline=train_pipeline),
            val=dict(type='TransformerPoseDataset',
                     ann_file=unified_ann_file,
                     img_prefix=unified_img_prefix,
                     data_cfg=data_cfg,
                     valid_class_ids=val_cids_split1,
                     max_kpt_num=channel_cfg['max_kpt_num'],
                     num_shots=1,
                     num_queries=15,
                     num_episodes=200,
                     pipeline=valid_pipeline),
            test=dict(type='TestPoseDataset',
                      ann_file=unified_ann_file,
                      img_prefix=unified_img_prefix,
                      data_cfg=data_cfg,
                      valid_class_ids=test_cids_split1,
                      max_kpt_num=channel_cfg['max_kpt_num'],
                      num_shots=1,
                      num_queries=15,
                      num_episodes=200,
                      pck_threshold_list=[0.05, 0.10, 0.15, 0.2],
                      pipeline=test_pipeline),
            )
shuffle_cfg = dict(interval=1)
