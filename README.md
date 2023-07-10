# StructDiffusion

Pytorch implementation for RSS 2023 paper _**StructDiffusion**_: Language-Guided Creation of Physically-Valid Structures using Unseen Objects. [[PDF]](https://roboticsconference.org/program/papers/031/) [[Video]](https://structdiffusion.github.io/media/overview.mp4) [[Website]](https://structdiffusion.github.io/)

StructDiffusion combines a diffusion model and an object-centric transformer to construct structures given partial-view point clouds and high-level language goals, such as “_set the table_”.

## Installation

```bash
conda create -n StructDiffusion python=3.8
conda activate StructDiffusion
pip install -r requirements.txt
pip install -e .
```

If the correct version of some dependencies are not installed, try the following.
```bash
pip uninstall torch torchaudio torchvision
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install chardet
```

### Notes on Dependencies
- Use the [compatibility matrix](https://lightning.ai/docs/pytorch/latest/versioning.html#compatibility-matrix) to match pytorch lightning and pytorch
- `torch`: After installation, check if pytorch can use `.cuda()`.
- `h5py==2.10`: this specific version is needed.
- If `AttributeError: module 'numpy' has no attribute 'typeDict'` is encountered, try uninstall numpy and install `numpy==1.21`.

- Tested on Ubuntu 18.04 with RTX 3090

## Data and Assets
- [Training Rearrangement Sequences](https://www.dropbox.com/s/vhgexwx1dqipdxj/training_data.zip?dl=0)
- [Testing Rearrangement Scenes](https://www.dropbox.com/s/colp3l5v5tpnnne/testing_data.zip?dl=0)
- [Pairwise Collision Data](https://www.dropbox.com/s/io1zf0cr7933i8j/pairwise_collision_data.zip?dl=0)
- [Object Models](https://www.dropbox.com/s/cnv91p05s725lyv/housekeep_custom_handpicked_small.zip?dl=0)
- [Pretrained Models](https://www.dropbox.com/s/o6yadulmo46mu60/wandb_logs.zip?dl=0)

[//]: # (- [Legacy Pretrained Models]&#40;https://www.dropbox.com/s/cnv91p05s725lyv/housekeep_custom_handpicked_small.zip?dl=0&#41;)

## Quick Starts
Set up data and models:
- Required data: [Testing Rearrangement Scenes](https://www.dropbox.com/s/colp3l5v5tpnnne/testing_data.zip?dl=0)
- Required models: [Pretrained Models](https://www.dropbox.com/s/o6yadulmo46mu60/wandb_logs.zip?dl=0)
- Modify the config file [base.yaml](configs/base.yaml) based on where the testing data and pretrained model are stored. Specifically, modify `base_dirs.testing_data` and `base_dirs.wandb_dir` in the config file.

There are two options:
- Running the diffusion model on testing scenes using [infer.py](scripts/infer.py):
```bash
# in ./scripts/
python infer.py
```

- Running the diffusion model and collision discriminator on testing scenes using [infer.py](scripts/infer_with_discriminator.py):
```bash
# in ./scripts/
python infer_with_discriminator.py
```

## Training
### Training the Conditional Pose Diffusion Model
- Required data: [Training Rearrangement Sequences](https://www.dropbox.com/s/vhgexwx1dqipdxj/training_data.zip?dl=0)
- Modify the config file [base.yaml](configs/base.yaml) based on where the training data is stored and where you want to store the trained model.
- You can change params of the model in [conditional_pose_diffusion.yaml](configs/conditional_pose_diffusion.yaml)
- Train the model with [train_generator.py](scripts/train_generator.py). The training progress can be monitored with `wandb`
```bash
# in ./scripts/
python train_generator.py
```

### Training the Pairwise Collision Discriminator
- Required data: [Pairwise Collision Data](https://www.dropbox.com/s/io1zf0cr7933i8j/pairwise_collision_data.zip?dl=0)
- Modify the config file [base.yaml](configs/base.yaml) based on where the training data is stored and where you want to store the trained model. 
- Note that training this model requries both Training Rearrangement Sequences and Pairwise Collision Data. We will use partial object point clouds from the rearrangement sequences and then use the query poses and groundtruth collision labels from the collision data.
- You can change params of the model in [pairwise_collision.yaml](configs/pairwise_collision.yaml)
- Train the model with [train_discriminator.py](scripts/train_discriminator.py). The training progress can be monitored with `wandb`.
```bash
# in ./scripts/
python train_discriminator.py
```

## Citation
If you find our work useful in your research, please cite:
```
@inproceedings{structdiffusion2023,
    title     = {StructDiffusion: Language-Guided Creation of Physically-Valid Structures using Unseen Objects},
    author    = {Liu, Weiyu and Du, Yilun and  Hermans, Tucker and Chernova, Sonia and Paxton, Chris},
    year      = {2023},
    booktitle = {RSS 2023}
}
```
