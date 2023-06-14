import os

def get_checkpoint_path_from_dir(checkpoint_dir):
    checkpoint_path = None
    for file in os.listdir(checkpoint_dir):
        if "ckpt" in file:
            checkpoint_path = os.path.join(checkpoint_dir, file)
    assert checkpoint_path is not None
    return checkpoint_path