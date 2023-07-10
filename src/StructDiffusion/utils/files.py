import os


def get_checkpoint_path_from_dir(checkpoint_dir):
    checkpoint_path = None
    for file in os.listdir(checkpoint_dir):
        if "ckpt" in file:
            checkpoint_path = os.path.join(checkpoint_dir, file)
    assert checkpoint_path is not None
    return checkpoint_path


def replace_config_for_testing_data(cfg, testing_data_cfg):
    cfg.DATASET.data_roots = testing_data_cfg.DATASET.data_roots
    cfg.DATASET.index_roots = testing_data_cfg.DATASET.index_roots
    cfg.DATASET.vocab_dir = testing_data_cfg.DATASET.vocab_dir

