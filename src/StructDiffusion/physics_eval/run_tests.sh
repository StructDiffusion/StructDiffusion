#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
conda activate StructDiffusion
initial_seed=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/weiyu/Research/StructDiffusion/StructDiffusion/src:$PYTHONPATH"
export PYTHONPATH="/home/weiyu/Research/StructDiffusion/rearrangement_gym/python:$PYTHONPATH"
echo $PYTHONPATH


# housekeep objects

# transformer
python eval_transformer_all_shapes.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/transformer/circle.yaml
python eval_transformer_all_shapes.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/transformer/tower.yaml
python eval_transformer_all_shapes.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/transformer/line.yaml
python eval_transformer_all_shapes.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/transformer/dinner.yaml
#
python eval_transformer_all_shapes_lan_local_shape_param_discriminator_collision_detector.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/transformer_collision/dinner.yaml
python eval_transformer_all_shapes_lan_local_shape_param_discriminator_collision_detector.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/transformer_collision/line.yaml
python eval_transformer_all_shapes_lan_local_shape_param_discriminator_collision_detector.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/transformer_collision/circle.yaml
python eval_transformer_all_shapes_lan_local_shape_param_discriminator_collision_detector.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/transformer_collision/tower.yaml


# CEM
python eval_transformer_all_shapes_cem_lan_local_shape_param_discriminator_collision_detector.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/transformer_discriminator_cem/dinner.yaml
python eval_transformer_all_shapes_cem_lan_local_shape_param_discriminator_collision_detector.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/transformer_discriminator_cem/line.yaml
python eval_transformer_all_shapes_cem_lan_local_shape_param_discriminator_collision_detector.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/transformer_discriminator_cem/circle.yaml
python eval_transformer_all_shapes_cem_lan_local_shape_param_discriminator_collision_detector.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/transformer_discriminator_cem/tower.yaml

#python eval_vae_cem_lan_local_shape_param_discriminator_collision_detector.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/vae_discriminator_cem/dinner.yaml
#python eval_vae_cem_lan_local_shape_param_discriminator_collision_detector.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/vae_discriminator_cem/line.yaml
#python eval_vae_cem_lan_local_shape_param_discriminator_collision_detector.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/vae_discriminator_cem/circle.yaml
#python eval_vae_cem_lan_local_shape_param_discriminator_collision_detector.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/vae_discriminator_cem/tower.yaml

#
#
#
## diffusion
#python eval_diffusion_v3_lang.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/diffusion_v3_lang/dinner.yaml
#python eval_diffusion_v3_lang.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/diffusion_v3_lang/line.yaml
#python eval_diffusion_v3_lang.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/diffusion_v3_lang/circle.yaml
#python eval_diffusion_v3_lang.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/diffusion_v3_lang/tower.yaml
#
#python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/diffusion_v3_lang_collision/dinner.yaml
#python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/diffusion_v3_lang_collision/line.yaml
python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/diffusion_v3_lang_collision/circle.yaml
#python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/diffusion_v3_lang_collision/tower.yaml
#
#python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/diffusion_v3_lang_discriminator/dinner.yaml
#python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/diffusion_v3_lang_discriminator/line.yaml
#python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/diffusion_v3_lang_discriminator/circle.yaml
#python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/diffusion_v3_lang_discriminator/tower.yaml
#
#


# vae
#python eval_vae.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/vae/dinner.yaml
#python eval_vae.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/vae/line.yaml
#python eval_vae.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/vae/circle.yaml
#python eval_vae.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/vae/tower.yaml
#
#python eval_vae_lan_local_shape_param_discriminator_collision_detector.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/vae_collision/dinner.yaml
#python eval_vae_lan_local_shape_param_discriminator_collision_detector.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/vae_collision/line.yaml
#python eval_vae_lan_local_shape_param_discriminator_collision_detector.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/vae_collision/circle.yaml
#python eval_vae_lan_local_shape_param_discriminator_collision_detector.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/vae_collision/tower.yaml

# Diffusion with Language
#python eval_diffusion_v4_lang_template_sentence_lan_local_shape_param_discriminator_collision_detector.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/diffusion_v4_lang_template_collision/dinner.yaml
#python eval_diffusion_v4_lang_template_sentence_lan_local_shape_param_discriminator_collision_detector.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/diffusion_v4_lang_template_collision/line.yaml
#python eval_diffusion_v4_lang_template_sentence_lan_local_shape_param_discriminator_collision_detector.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/diffusion_v4_lang_template_collision/circle.yaml
#python eval_diffusion_v4_lang_template_sentence_lan_local_shape_param_discriminator_collision_detector.py --config_file ../../../configs/physics_eval/dataset_housekeep_custom/diffusion_v4_lang_template_collision/tower.yaml