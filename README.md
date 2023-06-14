# StructDiffusion

Pytorch implementation for StructDiffuion

StructDiffusion rearranges unknown objects into semantically meaningful spatial structures based on high-level language instructions and partial-view
point cloud observations of the scene.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

The correct version of torch is probably not installed. Try the following.
```bash
pip uninstall torch torchaudio torchvision
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install chardet
```

### Notes on Dependencies
- Use the [compatibility matrix](https://lightning.ai/docs/pytorch/latest/versioning.html#compatibility-matrix) to match pytorch lightning and pytorch
- `torch`: After installation, check if pytorch can use `.cuda()`.
- `h5py==2.10`: this specific version is needed.

## Dependencies for Evaluation in the Physics Simulator
We will use PyBullet and NVISII and the wrapper [rearrangement_gym](https://github.com/wliu88/rearrangement_gym.git).
```
git clone https://github.com/wliu88/rearrangement_gym.git
pip install pybullet==3.1.7
pip install nvisii==1.1.70
```

## Data and Assets
- [Training rearrangement sequences](https://www.dropbox.com/s/v6dx9o7n7xub094/training_data.zip?dl=0)
- [Testing rearrangement scenes](https://www.dropbox.com/s/colp3l5v5tpnnne/testing_data.zip?dl=0)
- [Pairwise collision data](https://www.dropbox.com/s/io1zf0cr7933i8j/pairwise_collision_data.zip?dl=0)
- [Object Models](https://www.dropbox.com/s/3awy4aewf0afslb/models.zip?dl=0)
- [Pretrained Models](https://www.dropbox.com/s/lq8qdtrd7krlcpz/models.zip?dl=0)

[//]: # (- [Legacy Pretrained Models]&#40;https://www.dropbox.com/s/cnv91p05s725lyv/housekeep_custom_handpicked_small.zip?dl=0&#41;)

## Quick Starts
- Required data: [Testing rearrangement scenes](https://www.dropbox.com/s/colp3l5v5tpnnne/testing_data.zip?dl=0)
- Required model: [Pretrained Models](https://www.dropbox.com/s/lq8qdtrd7krlcpz/models.zip?dl=0)
- Modify the config file [base.yaml](configs/base.yaml) based on where the testing data and trained model are stored
- Running the model on test split (with known objects) using [infer.py](scripts/infer.py)

## Training
### Training the Conditional Pose Diffusion Model
- Required data: [Training rearrangement sequences](https://www.dropbox.com/s/v6dx9o7n7xub094/training_data.zip?dl=0)
- Modify the config file [base.yaml](configs/base.yaml) based on where the training data is stored and where you want to store the trained model.
- You can change params of the model in [conditional_pose_diffusion.yaml](configs/conditional_pose_diffusion.yaml)
- Train the model with [train_generator.py](scripts/train_generator.py). The training progress can be monitored with `wandb`

### Training the Pairwise Collision Discriminator
- Required data: [Pairwise collision data](https://www.dropbox.com/s/io1zf0cr7933i8j/pairwise_collision_data.zip?dl=0)
- Modify the config file [base.yaml](configs/base.yaml) based on where the training data is stored and where you want to store the trained model.
- You can change params of the model in [pairwise_collision.yaml](configs/pairwise_collision.yaml)
- Train the model with [train_discriminator.py](scripts/train_discriminator.py). The training progress can be monitored with `wandb`.

[//]: # (### Evaluating on Novel Objects in the Physics Simulator)

[//]: # (- Required data: [Testing rearrangement scenes]&#40;https://www.dropbox.com/s/colp3l5v5tpnnne/testing_data.zip?dl=0&#41;, [Object Models]&#40;https://www.dropbox.com/s/3awy4aewf0afslb/models.zip?dl=0&#41;, [Pretrained Models]&#40;https://www.dropbox.com/s/cnv91p05s725lyv/housekeep_custom_handpicked_small.zip?dl=0&#41; or models you trained)

[//]: # (- Source [rearrangement_gym]&#40;https://github.com/wliu88/rearrangement_gym.git&#41;. For example, ```export PYTHONPATH="/path/to/rearrangement_gym/python:$PYTHONPATH"```.)

[//]: # (- Modify the config file [base.yaml]&#40;configs/physics_eval/dataset_housekeep_custom/base.yaml&#41;)

[//]: # (- To test the diffusion model, use [eval_diffusion_v3_lang.py]&#40;src/StructDiffusion/physics_eval/Feval_diffusion_v3_lang.py&#41; and config files in [diffusion_v3_lang]&#40;configs/physics_eval/dataset_housekeep_custom/diffusion_v3_lang&#41;.)

[//]: # (- To test the diffusion model with the collision model or structure discriminator, use [eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py]&#40;src/StructDiffusion/physics_eval/eval_diffusion_v3_lang_lan_local_shape_param_discriminator_collision_detector.py&#41; and config files in [diffusion_v3_lang_collision]&#40;configs/physics_eval/dataset_housekeep_custom/diffusion_v3_lang_collision&#41; and [diffusion_v3_lang_discriminator]&#40;configs/physics_eval/dataset_housekeep_custom/diffusion_v3_lang_discriminator&#41;)

[//]: # (- For batch testing, [run_tests.sh]&#40;src/StructDiffusion/physics_eval/run_tests.sh&#41; can be helpful.)