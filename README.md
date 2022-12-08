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
- `torch`: After installation, check if pytorch can use `.cuda()`.
- `h5py==2.10`: this specific version is needed.
- `omegaconfg==1.4.1`: some functions used in this repo are from newer versions

## Data
- Training rearrangement sequences
  - Circle
  - Stacking
  - Table Setting
  - Line
- Testing rearrangement scenes
- Object Models:
- Pretrained Models:

## Quick Start
- Download the test split of the dataset from [this link](https://drive.google.com/file/d/1e76qJbBJ2bKYq0JzDSRWZjswySX1ftq_/view?usp=sharing) and unzip to the `$STRUCTFORMER/data_new_objects_test_split`
- Train the model with `src/training/train_diffuser_v4_template_language.py` and run sampling with `src/StructDiffusion/evaluation/infer_diffuser_v4_template_language.py`.

## Notes
- List access with `zip()` does not work well with omegaconf variables.

## Run on Robot
- See the [robot readme](./src/robot/README.md)
