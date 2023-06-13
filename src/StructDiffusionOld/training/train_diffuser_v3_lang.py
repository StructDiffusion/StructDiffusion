import math
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch3d.transforms as tra3d
import torch.optim as optim
import time
import os
import argparse
from omegaconf import OmegaConf

from StructDiffusion.utils.rearrangement import show_pcs_color_order
from StructDiffusion.data.dataset_v1_diffuser import SemanticArrangementDataset
from StructDiffusion.data.tokenizer import Tokenizer
from StructDiffusion.utils.rotation_continuity import compute_rotation_matrix_from_ortho6d
from StructDiffusion.models.models import TransformerDiffuserLang
from torch.utils.tensorboard import SummaryWriter


########################################################################################################################
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


########################################################################################################################


class NoiseSchedule:

    def __init__(self, timesteps=200):

        self.timesteps = timesteps

        # define beta schedule
        self.betas = linear_beta_schedule(timesteps=timesteps)
        # self.betas = cosine_beta_schedule(timesteps=timesteps)

        # define alphas
        self.alphas = 1. - self.betas
        # alphas_cumprod: alpha bar
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


########################################################################################################################
def get_diffusion_variables(struct_xyztheta_inputs, obj_xyztheta_inputs):

    # important: we need to get the first two columns, not first two rows
    # array([[ 3,  4,  5],
    #   [ 6,  7,  8],
    #   [ 9, 10, 11]])
    xyz_6d_idxs = [0, 1, 2, 3, 6, 9, 4, 7, 10]

    # print(batch_data["obj_xyztheta_inputs"].shape)
    # print(batch_data["struct_xyztheta_inputs"].shape)

    # only get the first and second columns of rotation
    obj_xyztheta_inputs = obj_xyztheta_inputs[:, :, xyz_6d_idxs]  # B, N, 9
    struct_xyztheta_inputs = struct_xyztheta_inputs[:, :, xyz_6d_idxs]  # B, 1, 9

    x = torch.cat([struct_xyztheta_inputs, obj_xyztheta_inputs], dim=1)  # B, 1 + N, 9

    # print(x.shape)

    return x


def get_struct_objs_poses(x):

    on_gpu = x.is_cuda
    if not on_gpu:
        x = x.cuda()

    # assert x.is_cuda, "compute_rotation_matrix_from_ortho6d requires input to be on gpu"
    device = x.device

    # important: the noisy x can go out of bounds
    x = torch.clamp(x, min=-1, max=1)

    # x: B, 1 + N, 9
    B = x.shape[0]
    N = x.shape[1] - 1

    # compute_rotation_matrix_from_ortho6d takes in [B, 6], outputs [B, 3, 3]
    x_6d = x[:, :, 3:].reshape(-1, 6)
    x_rot = compute_rotation_matrix_from_ortho6d(x_6d).reshape(B, N+1, 3, 3)  # B, 1 + N, 3, 3

    x_trans = x[:, :, :3] # B, 1 + N, 3

    x_full = torch.eye(4).repeat(B, 1 + N, 1, 1).to(device)
    x_full[:, :, :3, :3] = x_rot
    x_full[:, :, :3, 3] = x_trans

    struct_pose = x_full[:, 0].unsqueeze(1) # B, 1, 4, 4
    pc_poses_in_struct = x_full[:, 1:] # B, N, 4, 4

    if not on_gpu:
        struct_pose = struct_pose.cpu()
        pc_poses_in_struct = pc_poses_in_struct.cpu()

    return struct_pose, pc_poses_in_struct


def move_pc_and_create_scene(obj_xyzs, struct_pose, pc_poses_in_struct):

    device = obj_xyzs.device

    # obj_xyzs: B, N, P, 3
    # struct_pose: B, 1, 4, 4
    # pc_poses_in_struct: B, N, 4, 4

    B, N, _, _ = pc_poses_in_struct.shape
    _, _, P, _ = obj_xyzs.shape

    current_pc_poses = torch.eye(4).repeat(B, N, 1, 1).to(device)  # B, N, 4, 4
    # print(torch.mean(obj_xyzs, dim=2).shape)
    current_pc_poses[:, :, :3, 3] = torch.mean(obj_xyzs, dim=2)  # B, N, 4, 4
    current_pc_poses = current_pc_poses.reshape(B * N, 4, 4)  # B x N, 4, 4

    struct_pose = struct_pose.repeat(1, N, 1, 1) # B, N, 4, 4
    struct_pose = struct_pose.reshape(B * N, 4, 4)  # B x 1, 4, 4
    pc_poses_in_struct = pc_poses_in_struct.reshape(B * N, 4, 4)  # B x N, 4, 4

    goal_pc_pose = struct_pose @ pc_poses_in_struct  # B x N, 4, 4
    # print("goal pc poses")
    # print(goal_pc_pose)
    goal_pc_transform = goal_pc_pose @ torch.inverse(current_pc_poses)  # B x N, 4, 4

    # important: pytorch3d uses row-major ordering, need to transpose each transformation matrix
    transpose = tra3d.Transform3d(matrix=goal_pc_transform.transpose(1, 2))

    new_obj_xyzs = obj_xyzs.reshape(B * N, P, 3)  # B x N, P, 3
    new_obj_xyzs = transpose.transform_points(new_obj_xyzs)

    # put it back to B, N, P, 3
    new_obj_xyzs = new_obj_xyzs.reshape(B, N, P, 3)

    # visualize_batch_pcs(new_obj_xyzs, B, N, P)

    return new_obj_xyzs


def compute_current_and_goal_pc_poses(obj_xyzs, struct_pose, pc_poses_in_struct):

    device = obj_xyzs.device

    # obj_xyzs: B, N, P, 3
    # struct_pose: B, 1, 4, 4
    # pc_poses_in_struct: B, N, 4, 4
    B, N, _, _ = pc_poses_in_struct.shape
    _, _, P, _ = obj_xyzs.shape

    current_pc_poses = torch.eye(4).repeat(B, N, 1, 1).to(device)  # B, N, 4, 4
    # print(torch.mean(obj_xyzs, dim=2).shape)
    current_pc_poses[:, :, :3, 3] = torch.mean(obj_xyzs, dim=2)  # B, N, 4, 4

    struct_pose = struct_pose.repeat(1, N, 1, 1)  # B, N, 4, 4
    struct_pose = struct_pose.reshape(B * N, 4, 4)  # B x 1, 4, 4
    pc_poses_in_struct = pc_poses_in_struct.reshape(B * N, 4, 4)  # B x N, 4, 4

    goal_pc_poses = struct_pose @ pc_poses_in_struct  # B x N, 4, 4
    goal_pc_poses = goal_pc_poses.reshape(B, N, 4, 4)  # B, N, 4, 4
    return current_pc_poses, goal_pc_poses


def visualize_batch_pcs(obj_xyzs, B, N, P, verbose=True, limit_B=None, save_dir=None):
    if limit_B is None:
        limit_B = B

    vis_obj_xyzs = obj_xyzs.reshape(B, N, P, -1)
    vis_obj_xyzs = vis_obj_xyzs[:limit_B]

    if type(vis_obj_xyzs).__module__ == torch.__name__:
        if vis_obj_xyzs.is_cuda:
            vis_obj_xyzs = vis_obj_xyzs.detach().cpu()
        vis_obj_xyzs = vis_obj_xyzs.numpy()

    for bi, vis_obj_xyz in enumerate(vis_obj_xyzs):
        if verbose:
            print("example {}".format(bi))
            print(vis_obj_xyz.shape)

        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, "b{}.jpg".format(bi))
            show_pcs_color_order([xyz[:, :3] for xyz in vis_obj_xyz], None, visualize=False, add_coordinate_frame=False,
                                 side_view=True, save_path=save_path)
        else:
            show_pcs_color_order([xyz[:, :3] for xyz in vis_obj_xyz], None, visualize=True, add_coordinate_frame=False,
                                 side_view=True)


# forward diffusion (using the nice property)
def q_sample(x_start, t, noise_schedule, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(noise_schedule.sqrt_alphas_cumprod, t, x_start.shape)
    # print("sqrt_alphas_cumprod_t", sqrt_alphas_cumprod_t)
    sqrt_one_minus_alphas_cumprod_t = extract(
        noise_schedule.sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )
    # print("sqrt_one_minus_alphas_cumprod_t", sqrt_one_minus_alphas_cumprod_t)
    # print("noise", noise)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


########################################################################################################################
def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


########################################################################################################################
def train_model(cfg, model, data_iter, noise_schedule, optimizer, warmup, num_epochs, device, save_best_model,
                summary_writer, grad_clipping=1.0):

    loss_type = cfg.diffusion.loss_type

    # if save_best_model:
    #     best_model_dir = os.path.join(cfg.experiment_dir, "best_model")
    #     print("best model will be saved to {}".format(best_model_dir))
    #     if not os.path.exists(best_model_dir):
    #         os.makedirs(best_model_dir)
    #     best_score = -np.inf

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()
        epoch_loss = 0

        with tqdm.tqdm(total=len(data_iter["train"])) as pbar:
            for step, batch in enumerate(data_iter["train"]):
                optimizer.zero_grad()

                # input
                xyzs = batch["xyzs"].to(device, non_blocking=True)
                B = xyzs.shape[0]
                # obj_pad_mask: we don't need it now since we are testing
                obj_xyztheta_inputs = batch["obj_xyztheta_inputs"].to(device, non_blocking=True)
                struct_xyztheta_inputs = batch["struct_xyztheta_inputs"].to(device, non_blocking=True)
                position_index = batch["position_index"].to(device, non_blocking=True)
                struct_position_index = batch["struct_position_index"].to(device, non_blocking=True)
                object_pad_mask = batch["obj_pad_mask"].to(device, non_blocking=True)
                struct_pad_mask = batch["struct_pad_mask"].to(device, non_blocking=True)
                sentence = batch["sentence"].to(device, non_blocking=True)
                token_type_index = batch["token_type_index"].to(device, non_blocking=True)
                struct_token_type_index = batch["struct_token_type_index"].to(device, non_blocking=True)
                sentence_pad_mask = batch["sentence_pad_mask"].to(device, non_blocking=True)

                start_token = torch.zeros((B, 1), dtype=torch.long).to(device, non_blocking=True)
                t = torch.randint(0, noise_schedule.timesteps, (B,), dtype=torch.long).to(device, non_blocking=True)

                #--------------
                x_start = get_diffusion_variables(struct_xyztheta_inputs, obj_xyztheta_inputs)
                noise = torch.randn_like(x_start, device=device)
                x_noisy = q_sample(x_start=x_start, t=t, noise_schedule=noise_schedule, noise=noise)

                struct_xyztheta_inputs = x_noisy[:, 0, :].unsqueeze(1)  # B, 1, 3 + 6
                obj_xyztheta_inputs = x_noisy[:, 1:, :]  # B, N, 3 + 6
                struct_xyztheta_outputs, obj_xyztheta_outputs = model.forward(t, xyzs, obj_xyztheta_inputs, struct_xyztheta_inputs, sentence,
                                                                              position_index, struct_position_index,
                                                                              token_type_index, struct_token_type_index,
                                                                              start_token,
                                                                              object_pad_mask, struct_pad_mask, sentence_pad_mask)

                predicted_noise = torch.cat([struct_xyztheta_outputs, obj_xyztheta_outputs], dim=1)  # B, 1+N, 9

                # important: skip computing loss for masked positions
                pad_mask = torch.cat([struct_pad_mask, object_pad_mask], dim=1)  # B, 1+N
                keep_mask = (pad_mask == 0)
                noise = noise[keep_mask]  # dim: number of positions that need loss calculation
                predicted_noise = predicted_noise[keep_mask]

                if loss_type == 'l1':
                    loss = F.l1_loss(noise, predicted_noise)
                elif loss_type == 'l2':
                    loss = F.mse_loss(noise, predicted_noise)
                elif loss_type == "huber":
                    loss = F.smooth_l1_loss(noise, predicted_noise)
                else:
                    raise NotImplementedError()
                # --------------

                loss.backward()
                if grad_clipping != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clipping)
                optimizer.step()
                epoch_loss += loss
                pbar.update(1)
                pbar.set_postfix({"Batch loss": loss})

                summary_writer.add_scalar("train_loss", loss, epoch * len(data_iter["train"]) + step)
                summary_writer.add_scalar("epoch", epoch, epoch * len(data_iter["train"]) + step)

        if warmup is not None:
            warmup.step()

        print('[Epoch:{}]:  Training Loss:{:.4}'.format(epoch, epoch_loss))

        validate_model(cfg, model, data_iter, noise_schedule, epoch, device, summary_writer)

        # evaluate(gts, predictions, ["obj_x_outputs", "obj_y_outputs", "obj_z_outputs", "obj_theta_outputs",
        #                             "struct_x_inputs", "struct_y_inputs", "struct_z_inputs", "struct_theta_inputs"])
        #
        # score = validate(cfg, model, data_iter["valid"], epoch, device)
        # if save_best_model and score > best_score:
        #     print("Saving best model so far...")
        #     best_score = score
        #     save_model(best_model_dir, cfg, epoch, model)

    return model


def validate_model(cfg, model, data_iter, noise_schedule, epoch, device, summary_writer):

    loss_type = cfg.diffusion.loss_type

    model.eval()

    epoch_loss = 0
    # gts = defaultdict(list)
    # predictions = defaultdict(list)
    with torch.no_grad():

        with tqdm.tqdm(total=len(data_iter["valid"])) as pbar:
            for step, batch in enumerate(data_iter["valid"]):

                # input
                xyzs = batch["xyzs"].to(device, non_blocking=True)
                B = xyzs.shape[0]
                # obj_pad_mask: we don't need it now since we are testing
                obj_xyztheta_inputs = batch["obj_xyztheta_inputs"].to(device, non_blocking=True)
                struct_xyztheta_inputs = batch["struct_xyztheta_inputs"].to(device, non_blocking=True)
                position_index = batch["position_index"].to(device, non_blocking=True)
                struct_position_index = batch["struct_position_index"].to(device, non_blocking=True)
                object_pad_mask = batch["obj_pad_mask"].to(device, non_blocking=True)
                struct_pad_mask = batch["struct_pad_mask"].to(device, non_blocking=True)
                sentence = batch["sentence"].to(device, non_blocking=True)
                token_type_index = batch["token_type_index"].to(device, non_blocking=True)
                struct_token_type_index = batch["struct_token_type_index"].to(device, non_blocking=True)
                sentence_pad_mask = batch["sentence_pad_mask"].to(device, non_blocking=True)

                start_token = torch.zeros((B, 1), dtype=torch.long).to(device, non_blocking=True)
                t = torch.randint(0, noise_schedule.timesteps, (B,), dtype=torch.long).to(device, non_blocking=True)

                # --------------
                x_start = get_diffusion_variables(struct_xyztheta_inputs, obj_xyztheta_inputs)
                noise = torch.randn_like(x_start, device=device)
                x_noisy = q_sample(x_start=x_start, t=t, noise_schedule=noise_schedule, noise=noise)

                struct_xyztheta_inputs = x_noisy[:, 0, :].unsqueeze(1)  # B, 1, 3 + 6
                obj_xyztheta_inputs = x_noisy[:, 1:, :]  # B, N, 3 + 6
                struct_xyztheta_outputs, obj_xyztheta_outputs = model.forward(t, xyzs, obj_xyztheta_inputs, struct_xyztheta_inputs, sentence,
                                                                              position_index, struct_position_index,
                                                                              token_type_index, struct_token_type_index,
                                                                              start_token,
                                                                              object_pad_mask, struct_pad_mask, sentence_pad_mask)
                predicted_noise = torch.cat([struct_xyztheta_outputs, obj_xyztheta_outputs], dim=1)

                # important: skip computing loss for masked positions
                pad_mask = torch.cat([struct_pad_mask, object_pad_mask], dim=1)  # B, 1+N
                keep_mask = (pad_mask == 0)
                noise = noise[keep_mask]  # dim: number of positions that need loss calculation
                predicted_noise = predicted_noise[keep_mask]

                if loss_type == 'l1':
                    loss = F.l1_loss(noise, predicted_noise)
                elif loss_type == 'l2':
                    loss = F.mse_loss(noise, predicted_noise)
                elif loss_type == "huber":
                    loss = F.smooth_l1_loss(noise, predicted_noise)
                else:
                    raise NotImplementedError()

                epoch_loss += loss
                pbar.update(1)
                pbar.set_postfix({"Batch loss": loss})
                summary_writer.add_scalar("valid_loss", loss, epoch * len(data_iter["valid"]) + step)

    print('[Epoch:{}]:  Val Loss:{:.4}'.format(epoch, epoch_loss))
    # summary_writer.add_scalar("valid_loss", epoch_loss, epoch)


def save_model(model_dir, cfg, epoch, model, optimizer=None, scheduler=None):
    state_dict = {'epoch': epoch,
                  'model_state_dict': model.state_dict()}
    if optimizer is not None:
        state_dict["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state_dict["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state_dict, os.path.join(model_dir, "model.tar"))
    OmegaConf.save(cfg, os.path.join(model_dir, "config.yaml"))


def load_model(model_dir):
    """
    Load transformer model
    Important: to use the model, call model.eval() or model.train()
    :param model_dir:
    :return:
    """
    # load dictionaries
    cfg = OmegaConf.load(os.path.join(model_dir, "config.yaml"))

    data_cfg = cfg.dataset
    tokenizer = Tokenizer(data_cfg.vocab_dir)
    vocab_size = tokenizer.get_vocab_size()

    # initialize model
    model_cfg = cfg.model
    model = TransformerDiffuserLang(vocab_size,
                                    num_attention_heads=model_cfg.num_attention_heads,
                                encoder_hidden_dim=model_cfg.encoder_hidden_dim,
                                encoder_dropout=model_cfg.encoder_dropout,
                                encoder_activation=model_cfg.encoder_activation,
                                encoder_num_layers=model_cfg.encoder_num_layers,
                                structure_dropout=model_cfg.structure_dropout,
                                object_dropout=model_cfg.object_dropout,
                                ignore_rgb=model_cfg.ignore_rgb)
    model.to(cfg.device)

    # load state dicts
    checkpoint = torch.load(os.path.join(model_dir, "model.tar"))
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = None
    if "optimizer_state_dict" in checkpoint:
        training_cfg = cfg.training
        optimizer = optim.Adam(model.parameters(), lr=training_cfg.learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler = None
    if "scheduler_state_dict" in checkpoint:
        scheduler = None
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    noise_schedule = NoiseSchedule(cfg.diffusion.time_steps)

    epoch = checkpoint['epoch']
    return cfg, tokenizer, model, noise_schedule, optimizer, scheduler, epoch


def run_model(cfg):

    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.random_seed)
        torch.cuda.manual_seed_all(cfg.random_seed)
        torch.backends.cudnn.deterministic = True

    summary_writer = SummaryWriter(os.path.join(cfg.experiment_dir, "runs"))

    data_cfg = cfg.dataset
    tokenizer = Tokenizer(data_cfg.vocab_dir)
    vocab_size = tokenizer.get_vocab_size()

    train_dataset = SemanticArrangementDataset(data_roots=data_cfg.dirs,
                                               index_roots=data_cfg.index_dirs,
                                               split="train",
                                               tokenizer=tokenizer,
                                               max_num_objects=data_cfg.max_num_objects,
                                               max_num_other_objects=data_cfg.max_num_other_objects,
                                               max_num_shape_parameters=data_cfg.max_num_shape_parameters,
                                               max_num_rearrange_features=data_cfg.max_num_rearrange_features,
                                               max_num_anchor_features=data_cfg.max_num_anchor_features,
                                               num_pts=data_cfg.num_pts,
                                               filter_num_moved_objects_range=data_cfg.filter_num_moved_objects_range,
                                               data_augmentation=False,
                                               shuffle_object_index=data_cfg.shuffle_object_index)
    valid_dataset = SemanticArrangementDataset(data_roots=data_cfg.dirs,
                                               index_roots=data_cfg.index_dirs,
                                               split="valid",
                                               tokenizer=tokenizer,
                                               max_num_objects=data_cfg.max_num_objects,
                                               max_num_other_objects=data_cfg.max_num_other_objects,
                                               max_num_shape_parameters=data_cfg.max_num_shape_parameters,
                                               max_num_rearrange_features=data_cfg.max_num_rearrange_features,
                                               max_num_anchor_features=data_cfg.max_num_anchor_features,
                                               num_pts=data_cfg.num_pts,
                                               filter_num_moved_objects_range=data_cfg.filter_num_moved_objects_range,
                                               data_augmentation=False,
                                               shuffle_object_index=data_cfg.shuffle_object_index)

    data_iter = {}
    data_iter["train"] = DataLoader(train_dataset, batch_size=data_cfg.batch_size, shuffle=True,
                                    # collate_fn=SemanticArrangementDataset.collate_fn,
                                    pin_memory=data_cfg.pin_memory, num_workers=data_cfg.num_workers)
    data_iter["valid"] = DataLoader(valid_dataset, batch_size=data_cfg.batch_size, shuffle=False,
                                    # collate_fn=SemanticArrangementDataset.collate_fn,
                                    pin_memory=data_cfg.pin_memory, num_workers=data_cfg.num_workers)

    # load model
    model_cfg = cfg.model
    model = TransformerDiffuserLang(vocab_size,
                                    num_attention_heads=model_cfg.num_attention_heads,
                                encoder_hidden_dim=model_cfg.encoder_hidden_dim,
                                encoder_dropout=model_cfg.encoder_dropout,
                                encoder_activation=model_cfg.encoder_activation,
                                encoder_num_layers=model_cfg.encoder_num_layers,
                                structure_dropout=model_cfg.structure_dropout,
                                object_dropout=model_cfg.object_dropout,
                                ignore_rgb=model_cfg.ignore_rgb)

    model.to(cfg.device)

    training_cfg = cfg.training
    optimizer = optim.Adam(model.parameters(), lr=training_cfg.learning_rate, weight_decay=training_cfg.l2)
    scheduler = None
    warmup = None
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_cfg.lr_restart)
    # warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=training_cfg.warmup,
    #                                 after_scheduler=scheduler)

    noise_schedule = NoiseSchedule(cfg.diffusion.time_steps)

    train_model(cfg, model, data_iter, noise_schedule, optimizer, warmup, training_cfg.max_epochs, cfg.device,
                cfg.save_best_model, summary_writer)

    # save model
    if cfg.save_model:
        model_dir = os.path.join(cfg.experiment_dir, "model")
        print("Saving model to {}".format(model_dir))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        save_model(model_dir, cfg, cfg.max_epochs, model, optimizer, scheduler)


########################################################################################################################
# inference code
# @torch.no_grad()
# def p_sample(model, x, t, t_index):
#     betas_t = extract(betas, t, x.shape)
#     sqrt_one_minus_alphas_cumprod_t = extract(
#         sqrt_one_minus_alphas_cumprod, t, x.shape
#     )
#     sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
#
#     # Equation 11 in the paper
#     # Use our model (noise predictor) to predict the mean
#     model_mean = sqrt_recip_alphas_t * (
#             x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
#     )
#
#     if t_index == 0:
#         return model_mean
#     else:
#         posterior_variance_t = extract(posterior_variance, t, x.shape)
#         noise = torch.randn_like(x)
#         # Algorithm 2 line 4:
#         return model_mean + torch.sqrt(posterior_variance_t) * noise


# def sampling(cfg, model, data_iter, noise_schedule, device):
#
#     model.eval()
#
#     with torch.no_grad():
#         with tqdm.tqdm(total=len(data_iter["valid"])) as pbar:
#             for step, batch in enumerate(data_iter["valid"]):
#
#                 # input
#                 xyzs = batch["xyzs"].to(device, non_blocking=True)
#                 B = xyzs.shape[0]
#                 # obj_pad_mask: we don't need it now since we are testing
#                 obj_xyztheta_inputs = batch["obj_xyztheta_inputs"].to(device, non_blocking=True)
#                 struct_xyztheta_inputs = batch["struct_xyztheta_inputs"].to(device, non_blocking=True)
#                 position_index = batch["position_index"].to(device, non_blocking=True)
#                 struct_position_index = batch["struct_position_index"].to(device, non_blocking=True)
#
#                 start_token = torch.zeros((B, 1), dtype=torch.long).to(device, non_blocking=True)
#
#                 # --------------
#                 x_gt = get_diffusion_variables(struct_xyztheta_inputs, obj_xyztheta_inputs)
#
#                 # start from random noise
#                 x = torch.randn_like(x_gt, device=device)
#                 xs = []
#                 for t_index in reversed(range(0, noise_schedule.timesteps)):
#
#                     t = torch.full((B,), t_index, device=device, dtype=torch.long)
#
#                     betas_t = extract(noise_schedule.betas, t, x.shape)
#                     sqrt_one_minus_alphas_cumprod_t = extract(noise_schedule.sqrt_one_minus_alphas_cumprod, t, x.shape)
#                     sqrt_recip_alphas_t = extract(noise_schedule.sqrt_recip_alphas, t, x.shape)
#
#                     struct_xyztheta_inputs = x[:, 0, :].unsqueeze(1)  # B, 1, 3 + 6
#                     obj_xyztheta_inputs = x[:, 1:, :]  # B, N, 3 + 6
#                     struct_xyztheta_outputs, obj_xyztheta_outputs = model.forward(t, xyzs, obj_xyztheta_inputs,
#                                                                                   struct_xyztheta_inputs,
#                                                                                   position_index, struct_position_index,
#                                                                                   start_token)
#                     predicted_noise = torch.cat([struct_xyztheta_outputs, obj_xyztheta_outputs], dim=1)
#
#                     model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
#
#                     if t_index == 0:
#                         x = model_mean
#                     else:
#                         posterior_variance_t = extract(noise_schedule.posterior_variance, t, x.shape)
#                         noise = torch.randn_like(x)
#                         # Algorithm 2 line 4:
#                         x = model_mean + torch.sqrt(posterior_variance_t) * noise
#
#                     xs.append(x)
#                 # --------------


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run a simple model")
    parser.add_argument("--base_config_file", help='base config yaml file',
                        default='../../../configs/base.yaml',
                        type=str)
    parser.add_argument("--config_file", help='config yaml file',
                        default='../../../configs/diffuser_v3_lang.yaml',
                        type=str)
    args = parser.parse_args()
    assert os.path.exists(args.base_config_file), "Cannot find base config yaml file at {}".format(args.config_file)
    assert os.path.exists(args.config_file), "Cannot find config yaml file at {}".format(args.config_file)
    os.environ["DATETIME"] = time.strftime("%Y%m%d-%H%M%S")
    base_cfg = OmegaConf.load(args.base_config_file)
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(base_cfg, cfg)
    if not os.path.exists(cfg.experiment_dir):
        os.makedirs(cfg.experiment_dir)
    OmegaConf.save(cfg, os.path.join(cfg.experiment_dir, "config.yaml"))

    run_model(cfg)