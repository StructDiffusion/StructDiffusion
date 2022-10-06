# StructDiffusion

Pytorch implementation for StructDiffuion

StructDiffusion rearranges unknown objects into semantically meaningful spatial structures based on high-level language instructions and partial-view
point cloud observations of the scene.

## Quick Start
- Download the test split of the dataset from [this link](https://drive.google.com/file/d/1e76qJbBJ2bKYq0JzDSRWZjswySX1ftq_/view?usp=sharing) and unzip to the `$STRUCTFORMER/data_new_objects_test_split`
- Train the model with `src/training/train_diffuser_v4_template_language.py` and run sampling with `src/StructDiffusion/evaluation/infer_diffuser_v4_template_language.py`.

## Run on Robot
- See the [robot readme](./src/robot/README.md)