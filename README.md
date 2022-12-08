# StructDiffusion

Pytorch implementation for StructDiffuion

StructDiffusion rearranges unknown objects into semantically meaningful spatial structures based on high-level language instructions and partial-view
point cloud observations of the scene.

## Installation

```
pip install -r requirements.txt
pip install -e .
```

### Notes on Dependencies
- `torch`: After installation, check if pytorch can use `.cuda()`. I used `pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html`.
- `h5py==2.10`: this specific version is needed.
- `omegaconfg==1.4.1`: some functions used in this repo are from newer versions

## Dependencies for Evaluation in the Physics Simulator
We will use PyBullet and NVISII and the wrapper [rearrangement_gym](https://github.com/wliu88/rearrangement_gym.git).
```
git clone https://github.com/wliu88/rearrangement_gym.git
pip install pybullet==3.1.7
pip install nvisii==1.1.70
```

## Data
- [Training rearrangement sequences](https://www.dropbox.com/s/v6dx9o7n7xub094/training_data.zip?dl=0)
- [Testing rearrangement scenes](https://www.dropbox.com/s/colp3l5v5tpnnne/testing_data.zip?dl=0)
- [Object Models](https://www.dropbox.com/s/3awy4aewf0afslb/models.zip?dl=0)
- [Pretrained Models](https://www.dropbox.com/s/cnv91p05s725lyv/housekeep_custom_handpicked_small.zip?dl=0)

## Quick Start

### Training the Diffusion Model Conditioned on Word Tokens
- Required data: [Training rearrangement sequences](https://www.dropbox.com/s/v6dx9o7n7xub094/training_data.zip?dl=0)
- Modify the config file [base.yaml](configs/base.yaml) based on where the training data is stored and where you want to store the trained model.
- You can change params of the model in [diffuser_v3_lang.yaml](configs/diffuser_v3_lang.yaml).
- Train the model with [train_diffuser_v3_lang.py](src/StructDiffusion/training/train_diffuser_v3_lang.py). The training progress can be monitored with `tensorboard`. 
- Running the model on test split (with known objects) using [infer_diffuser_v3_lang.py](src/StructDiffusion/evaluation/infer_diffuser_v3_lang.py)

### Evaluating on Novel Objects in the Physics Simulator
- Required data: [Testing rearrangement scenes](https://www.dropbox.com/s/colp3l5v5tpnnne/testing_data.zip?dl=0), [Object Models](https://www.dropbox.com/s/3awy4aewf0afslb/models.zip?dl=0), [Pretrained Models](https://www.dropbox.com/s/cnv91p05s725lyv/housekeep_custom_handpicked_small.zip?dl=0) or models you trained
- Source [rearrangement_gym](https://github.com/wliu88/rearrangement_gym.git). For example, ```export PYTHONPATH="/path/to/rearrangement_gym/python:$PYTHONPATH"```.
- Modify the config file [base.yaml](configs/physics_eval/dataset_housekeep_custom/base.yaml)
- To test the diffusion model, use [eval_diffusion_v3_lang.py](src/StructDiffusion/physics_eval/Feval_diffusion_v3_lang.py) and config files in [diffusion_v3_lang](configs/physics_eval/dataset_housekeep_custom/diffusion_v3_lang).
- To test the diffusion model with the collision model or structure discriminator, use [eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py](src/StructDiffusion/physics_eval/eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py) and config files in [diffusion_v3_lang_collision](configs/physics_eval/dataset_housekeep_custom/diffusion_v3_lang_collision) and [diffusion_v3_lang_discriminator](configs/physics_eval/dataset_housekeep_custom/diffusion_v3_lang_discriminator)
- For batch testing, [run_tests.sh](src/StructDiffusion/physics_eval/run_tests.sh) can be helpful.

## Notes
- List access with `zip()` does not work well with omegaconf variables.
- Config loading has not been updated for baselines
- 

## Run on Robot
- See the [robot readme](./src/robot/README.md)