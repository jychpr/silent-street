# Controlled 25%-subset run: E-SAM-25 arm.
# Identical to config/OV_COCO/OVDQUO_RN50.py EXCEPT:
#   - pseudo_box: "ow_labels/OW_COCO_ESAM_uniform_subset25.json"  (was OW_COCO_R2.json)
#
# Pair (same image IDs, same budget, weight semantics differ by design):
#   OLN-25   -> oln_r2_subset25_30ep.py -> ow_labels/OW_COCO_R2_subset25.json   (learned weights)
#   E-SAM-25 -> this file               -> OW_COCO_ESAM_uniform_subset25.json   (weight=1.0 uniform)
#
# The E-SAM JSON is produced by diagnostics/subset25/esam_precompute_subset.py
# (run that to COMPLETE before launching this arm). weight=1.0 is uniform BY
# DESIGN: E-SAM carries no objectness; loader applies weight**0.5 per pseudo box.
#
# epochs=30 inherited below. output_dir is a CLI arg (main.py:36), NOT a config var,
# so the two arms differ ONLY in pseudo_box here; set --output_dir at launch:
#   --output_dir outputs/esam_uniform_subset25_30ep
# Mirror the OLN-R2 baseline eval schedule via CLI: --eval_start_epoch 1 --eval_every_epoch 2
# (argparse-only args; cannot live in the config file — see main.py).

##### start DINO parameters #####
param_dict_type = "default"
lr_backbone_names = ["backbone.0"]
lr_linear_proj_names = ["reference_points", "sampling_offsets"]
lr_linear_proj_mult = 0.1
ddetr_lr_param = False
weight_decay = 0.0001
clip_max_norm = 0.1
use_checkpoint = False
position_embedding = "sine"
pe_temperatureH = 20
pe_temperatureW = 20
enc_layers = 6
dec_layers = 6
unic_layers = 0
pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries =1000
query_dim = 4
num_patterns = 0
pdetr3_bbox_embed_diff_each_layer = False
pdetr3_refHW = -1
random_refpoints_xy = False
fix_refpoints_hw = -1
dabdetr_yolo_like_anchor_update = False
dabdetr_deformable_encoder = False
dabdetr_deformable_decoder = False
use_deformable_box_attn = False
box_attn_type = "roi_align"
dec_layer_number = None
enc_n_points = 4
dec_n_points = 4
decoder_layer_noise = False
dln_xy_noise = 0.2
dln_hw_noise = 0.2
add_channel_attention = False
add_pos_value = False
two_stage_type = "standard"
two_stage_pat_embed = 0
two_stage_add_query_num = 0
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
two_stage_learn_wh = False
two_stage_default_hw = 0.05
two_stage_keep_all_tokens = False
transformer_activation = "relu"
batch_norm_type = "FrozenBatchNorm2d"
masks = False
aux_loss = True
dec_pred_bbox_embed_share = False
dec_pred_class_embed_share = True
use_detached_boxes_dec_out = False
##### end DINO parameters #####


##### start loss parameters #####
set_cost_class = 2.0
set_cost_bbox = 5.0
set_cost_giou = 2.0
cls_loss_coef = 2.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0
enc_loss_coef = 1.0
interm_loss_coef = 1.0
no_interm_box_loss = False
focal_alpha = 0.25
decoder_sa_type = "sa"  # ['sa', 'ca_label', 'ca_content']
decoder_module_seq = ["sa", "ca", "ffn"]
##### end loss parameters #####


##### start dn parameters #####
use_dn = True
dn_number = 100
dn_box_noise_scale = 1.0
dn_label_noise_ratio = 0.5
embed_init_tgt = True
##### end dn parameters #####


##### start ema parameters #####
use_ema = True
ema_decay = 0.99996
ema_epoch = 0
##### end ema parameters #####


##### start open-vocabulary training parameters #####
lr = 1e-4
epochs = 30
lr_drop = 50
batch_size = 4
lr_backbone=0.0
save_checkpoint_interval = 1
wildcard="object"
num_feature_levels = 3
modelname = "ov_dquo"
text_dim=1024  # 1024 for RN50
backbone = "CLIP_RN50"
text_len = 15
pretrained=""
region_prompt_path = "pretrained/region_prompt_R50.pth"
backbone_out_indice=[1, 2, 3] # C3, C4, C5
pseudo_box = "ow_labels/OW_COCO_ESAM_uniform_subset25.json"
train_subset_ids = "diagnostics/subset25/subset_25pct_ids.json"  # deviation #7: restrict TRAIN iteration to the 25% subset
in_channel=[512, 1024]
##### end open-vocabulary training parameters #####


##### start inference parameters #####
eval_tau = 100
objectness_alpha = 1.0
nms_iou_threshold = 0.5
target_class_factor=3.0
##### end inference parameters #####


##### start dataset parameters #####
coco_path = "data"
dataset_file = "ovcoco"
label_version = "standard"
repeat_factor_sampling=False
num_label_sampled=48 # keep compatible with ovlvis
##### end dataset parameters #####
