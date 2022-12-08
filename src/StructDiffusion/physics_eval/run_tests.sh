#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
conda activate brain_gym
initial_seed=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/weiyu/Research/intern/brain_gym/python:$PYTHONPATH"
export PYTHONPATH="/home/weiyu/Research/intern/semantic-rearrangement:$PYTHONPATH"
export PYTHONPATH="/home/weiyu/Research/intern/semantic-rearrangement/src:$PYTHONPATH"
export PYTHONPATH="/home/weiyu/Research/intern/StructDiffuser/src:$PYTHONPATH"
echo $PYTHONPATH

# python eval_transformer_all_shapes.py
# python eval_vae_cem_lan_local_shape_param_discriminator.py
# python eval_vae_cem_single_task_discriminator.py

# python eval_diffusion_v3_lang.py
# python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py
# python eval_transformer_all_shapes_cem_lan_local_shape_param_discriminator_collision_detector_100.py
# python eval_vae_cem_lan_local_shape_param_discriminator_collision_detector.py

# python eval_diffusion_v5_compositional_two_perturbation.py
# python eval_diffusion_v6_compositional_three_perturbation.py




# transformer
#python eval_transformer_all_shapes.py --config_file ../configs/physics_eval/transformer/dinner.yaml
#python eval_transformer_all_shapes.py --config_file ../configs/physics_eval/transformer/line.yaml
#python eval_transformer_all_shapes.py --config_file ../configs/physics_eval/transformer/circle.yaml
#python eval_transformer_all_shapes.py --config_file ../configs/physics_eval/transformer/stacking.yaml
#
#python eval_transformer_all_shapes_cem_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/transformer_collision/dinner.yaml
#python eval_transformer_all_shapes_cem_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/transformer_collision/line.yaml
#python eval_transformer_all_shapes_cem_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/transformer_collision/circle.yaml
#python eval_transformer_all_shapes_cem_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/transformer_collision/stacking.yaml
#
#python eval_transformer_all_shapes_cem_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/transformer_discriminator_cem/dinner.yaml
#python eval_transformer_all_shapes_cem_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/transformer_discriminator_cem/line.yaml
#python eval_transformer_all_shapes_cem_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/transformer_discriminator_cem/circle.yaml
#python eval_transformer_all_shapes_cem_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/transformer_discriminator_cem/stacking.yaml



# vae
# python eval_vae.py --config_file ../configs/physics_eval/vae/dinner.yaml
#python eval_vae.py --config_file ../configs/physics_eval/vae/line.yaml
#python eval_vae.py --config_file ../configs/physics_eval/vae/circle.yaml
#python eval_vae.py --config_file ../configs/physics_eval/vae/stacking.yaml
#
#python eval_vae_cem_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/vae_collision/dinner.yaml
#python eval_vae_cem_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/vae_collision/line.yaml
#python eval_vae_cem_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/vae_collision/circle.yaml
#python eval_vae_cem_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/vae_collision/stacking.yaml
#
#python eval_vae_cem_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/vae_discriminator_cem/dinner.yaml
#python eval_vae_cem_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/vae_discriminator_cem/line.yaml
#python eval_vae_cem_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/vae_discriminator_cem/circle.yaml
#python eval_vae_cem_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/vae_discriminator_cem/stacking.yaml



## diffusion
#python eval_diffusion_v3_lang.py --config_file ../configs/physics_eval/diffusion_v3_lang/dinner.yaml
#python eval_diffusion_v3_lang.py --config_file ../configs/physics_eval/diffusion_v3_lang/line.yaml
#python eval_diffusion_v3_lang.py --config_file ../configs/physics_eval/diffusion_v3_lang/circle.yaml
#python eval_diffusion_v3_lang.py --config_file ../configs/physics_eval/diffusion_v3_lang/stacking.yaml
#
#python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/diffusion_v3_lang_collision/dinner.yaml
#python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/diffusion_v3_lang_collision/line.yaml
#python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/diffusion_v3_lang_collision/circle.yaml
#python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/diffusion_v3_lang_collision/stacking.yaml
#
#python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/diffusion_v3_lang_discriminator/dinner.yaml
#python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/diffusion_v3_lang_discriminator/line.yaml
#python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/diffusion_v3_lang_discriminator/circle.yaml
#python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/diffusion_v3_lang_discriminator/stacking.yaml

#python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/diffusion_v3_lang_discriminator_on_policy_trained/dinner.yaml
#python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/diffusion_v3_lang_discriminator_on_policy_trained/line.yaml
#python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/diffusion_v3_lang_discriminator_on_policy_trained/circle.yaml
#python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/diffusion_v3_lang_discriminator_on_policy_trained/stacking.yaml




# diffusion different seeds
#python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/diffusion_v3_lang_collision/r2/dinner.yaml
#python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/diffusion_v3_lang_collision/r2/line.yaml
#python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/diffusion_v3_lang_collision/r2/circle.yaml
#python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/diffusion_v3_lang_collision/r2/stacking.yaml
#
#python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/diffusion_v3_lang_collision/r3/dinner.yaml
#python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/diffusion_v3_lang_collision/r3/line.yaml
#python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/diffusion_v3_lang_collision/r3/circle.yaml
#python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/diffusion_v3_lang_collision/r3/stacking.yaml


# python eval_transformer_all_shapes_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/dataset_housekeep_custom/transformer_collision/dinner.yaml
# python eval_vae_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/dataset_housekeep_custom/vae_collision/dinner.yaml
# python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/dataset_housekeep_custom/diffusion_v3_lang_collision/dinner.yaml



########################################################################################################################
# housekeep objects

# transformer
python eval_transformer_all_shapes.py --config_file ../configs/physics_eval/dataset_housekeep_custom/transformer/dinner.yaml
python eval_transformer_all_shapes.py --config_file ../configs/physics_eval/dataset_housekeep_custom/transformer/line.yaml
python eval_transformer_all_shapes.py --config_file ../configs/physics_eval/dataset_housekeep_custom/transformer/circle.yaml
python eval_transformer_all_shapes.py --config_file ../configs/physics_eval/dataset_housekeep_custom/transformer/stacking.yaml

python eval_transformer_all_shapes_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/dataset_housekeep_custom/transformer_collision/dinner.yaml
python eval_transformer_all_shapes_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/dataset_housekeep_custom/transformer_collision/line.yaml
python eval_transformer_all_shapes_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/dataset_housekeep_custom/transformer_collision/circle.yaml
python eval_transformer_all_shapes_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/dataset_housekeep_custom/transformer_collision/stacking.yaml



# vae
python eval_vae.py --config_file ../configs/physics_eval/dataset_housekeep_custom/vae/dinner.yaml
python eval_vae.py --config_file ../configs/physics_eval/dataset_housekeep_custom/vae/line.yaml
python eval_vae.py --config_file ../configs/physics_eval/dataset_housekeep_custom/vae/circle.yaml
python eval_vae.py --config_file ../configs/physics_eval/dataset_housekeep_custom/vae/stacking.yaml

python eval_vae_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/dataset_housekeep_custom/vae_collision/dinner.yaml
python eval_vae_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/dataset_housekeep_custom/vae_collision/line.yaml
python eval_vae_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/dataset_housekeep_custom/vae_collision/circle.yaml
python eval_vae_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/dataset_housekeep_custom/vae_collision/stacking.yaml



# diffusion
python eval_diffusion_v3_lang.py --config_file ../configs/physics_eval/dataset_housekeep_custom/diffusion_v3_lang/dinner.yaml
python eval_diffusion_v3_lang.py --config_file ../configs/physics_eval/dataset_housekeep_custom/diffusion_v3_lang/line.yaml
python eval_diffusion_v3_lang.py --config_file ../configs/physics_eval/dataset_housekeep_custom/diffusion_v3_lang/circle.yaml
python eval_diffusion_v3_lang.py --config_file ../configs/physics_eval/dataset_housekeep_custom/diffusion_v3_lang/stacking.yaml

python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/dataset_housekeep_custom/diffusion_v3_lang_collision/dinner.yaml
python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/dataset_housekeep_custom/diffusion_v3_lang_collision/line.yaml
python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/dataset_housekeep_custom/diffusion_v3_lang_collision/circle.yaml
python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/dataset_housekeep_custom/diffusion_v3_lang_collision/stacking.yaml

python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/dataset_housekeep_custom/diffusion_v3_lang_discriminator/dinner.yaml
python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/dataset_housekeep_custom/diffusion_v3_lang_discriminator/line.yaml
python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/dataset_housekeep_custom/diffusion_v3_lang_discriminator/circle.yaml
python eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/dataset_housekeep_custom/diffusion_v3_lang_discriminator/stacking.yaml


# CEM
python eval_transformer_all_shapes_cem_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/dataset_housekeep_custom/transformer_discriminator_cem/dinner.yaml
python eval_transformer_all_shapes_cem_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/dataset_housekeep_custom/transformer_discriminator_cem/line.yaml
python eval_transformer_all_shapes_cem_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/dataset_housekeep_custom/transformer_discriminator_cem/circle.yaml
python eval_transformer_all_shapes_cem_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/dataset_housekeep_custom/transformer_discriminator_cem/stacking.yaml

python eval_vae_cem_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/dataset_housekeep_custom/vae_discriminator_cem/dinner.yaml
python eval_vae_cem_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/dataset_housekeep_custom/vae_discriminator_cem/line.yaml
python eval_vae_cem_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/dataset_housekeep_custom/vae_discriminator_cem/circle.yaml
python eval_vae_cem_lan_local_shape_param_discriminator_collision_detector.py --config_file ../configs/physics_eval/dataset_housekeep_custom/vae_discriminator_cem/stacking.yaml