import StructDiffusion.training.train_structure_predictor_9_lan_local_shape_param as discriminator_model
import StructDiffusion.data.dataset_v41_discriminator_env_lan_local_shape_param_return_pose as discriminator_dataset


class DiscriminatorInference:

    def __init__(self, model_dir, data_split="test", evaluate_each_sample_n_times=1):

        cfg, model, _, _, _, tokenizer = discriminator_model.load_model(model_dir)

        data_cfg = cfg.dataset
        dataset = discriminator_dataset.SemanticArrangementDataset(data_cfg.dirs, data_cfg.index_dirs, data_split,
                                                   tokenizer=tokenizer,
                                                   num_random_negative_examples=data_cfg.num_random_negative_examples,
                                                   min_translation=data_cfg.min_translation,
                                                   max_translation=data_cfg.max_translation,
                                                   min_rotation=data_cfg.min_rotation,
                                                   max_rotation=data_cfg.max_rotation,
                                                   max_num_objects=data_cfg.max_num_objects,
                                                   max_num_shape_parameters=data_cfg.max_num_shape_parameters,
                                                   num_pts=data_cfg.num_pts,
                                                   num_scene_pts=data_cfg.num_scene_pts,
                                                   oversample_positive=data_cfg.oversample_positive,
                                                   perturbation_mode=data_cfg.pertubation_mode,
                                                   random_structure_rotation=data_cfg.random_structure_rotation,
                                                   return_perturbation_score=cfg.regression_mode,
                                                   num_objects_to_include=data_cfg.num_objects_to_include,
                                                   data_augmentation=data_cfg.data_augmentation,
                                                   include_env_pc=data_cfg.include_env_pc,
                                                   num_env_pts=data_cfg.num_env_pts,
                                                   normalize_pc=data_cfg.normalize_pc)

        self.model = model
        self.cfg = cfg
        self.dataset = dataset