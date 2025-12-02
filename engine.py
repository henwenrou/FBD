from typing import Iterable
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import util.misc as utils
import functools
from tqdm import tqdm
import torch.nn.functional as F
from monai.metrics import compute_meandice
from torch.autograd import Variable
from dataloaders.saliency_balancing_fusion import get_SBF_map
from dataloaders.fourier_augment import core_guided_fourier_augment, build_radial_band_masks
print = functools.partial(print, flush=True)

def train_warm_up(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, learning_rate:float, warmup_iteration: int = 1500):
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    print_freq = 10
    cur_iteration=0
    while True:
        for i, samples in enumerate(metric_logger.log_every(data_loader, print_freq, 'WarmUp with max iteration: {}'.format(warmup_iteration))):
            for k,v in samples.items():
                if isinstance(samples[k],torch.Tensor):
                    samples[k]=v.to(device)
            cur_iteration+=1
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = cur_iteration/warmup_iteration*learning_rate * param_group["lr_scale"]

            img=samples['images']
            lbl=samples['labels']
            pred = model(img)
            loss_dict = criterion.get_loss(pred,lbl)
            losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            metric_logger.update(**loss_dict)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            if cur_iteration>=warmup_iteration:
                print(f'WarnUp End with Iteration {cur_iteration} and current lr is {optimizer.param_groups[0]["lr"]}.')
                return cur_iteration
        metric_logger.synchronize_between_processes()

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, cur_iteration:int, max_iteration: int = -1, grad_scaler=None):
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for i, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        for k, v in samples.items():
            if isinstance(samples[k], torch.Tensor):
                samples[k] = v.to(device)

        img = samples['images']
        lbl = samples['labels']

        if grad_scaler is None:
            pred = model(img)
            loss_dict = criterion.get_loss(pred,lbl)
            losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        else:
            with torch.cuda.amp.autocast():
                pred = model(img)
                loss_dict = criterion.get_loss(pred,lbl)
                losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys())
            optimizer.zero_grad()
            grad_scaler.scale(losses).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

        metric_logger.update(**loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        cur_iteration+=1
        if cur_iteration>=max_iteration and max_iteration>0:
            break

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return cur_iteration



def train_one_epoch_SBF(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, cur_iteration:int, max_iteration: int = -1,config=None,visdir=None):
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    visual_freq = 500
    for i, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        for k, v in samples.items():
            if isinstance(samples[k], torch.Tensor):
                samples[k] = v.to(device)

        GLA_img = samples['images']
        LLA_img = samples['aug_images']
        lbl = samples['labels']
        if cur_iteration % visual_freq == 0:
            visual_dict={}
            visual_dict['GLA']=GLA_img.detach().cpu().numpy()[0,0]
            visual_dict['LLA']=LLA_img.detach().cpu().numpy()[0,0]
            visual_dict['GT']=lbl.detach().cpu().numpy()[0]
        else:
            visual_dict=None

        input_var = Variable(GLA_img, requires_grad=True)

        optimizer.zero_grad()
        logits = model(input_var)
        loss_dict = criterion.get_loss(logits, lbl)
        losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)
        losses.backward(retain_graph=True)

        # saliency
        gradient = torch.sqrt(torch.mean(input_var.grad ** 2, dim=1, keepdim=True)).detach()

        saliency=get_SBF_map(gradient,config.grid_size)

        if visual_dict is not None:
            visual_dict['GLA_pred']=torch.argmax(logits,1).cpu().numpy()[0]

        if visual_dict is not None:
            visual_dict['GLA_saliency']= saliency.detach().cpu().numpy()[0,0]

        mixed_img = GLA_img.detach() * saliency + LLA_img * (1 - saliency)
        if visual_dict is not None:
            visual_dict['SBF']= mixed_img.detach().cpu().numpy()[0,0]

        aug_var = Variable(mixed_img, requires_grad=True)
        aug_logits = model(aug_var)
        aug_loss_dict = criterion.get_loss(aug_logits, lbl)
        aug_losses = sum(aug_loss_dict[k] * criterion.weight_dict[k] for k in aug_loss_dict.keys() if k in criterion.weight_dict)

        aug_losses.backward()

        if visual_dict is not None:
            visual_dict['SBF_pred'] = torch.argmax(aug_logits, 1).cpu().numpy()[0]

        optimizer.step()

        all_loss_dict={}
        for k in loss_dict.keys():
            if k not in criterion.weight_dict:continue
            all_loss_dict[k]=loss_dict[k]
            all_loss_dict[k+'_aug']=aug_loss_dict[k]

        metric_logger.update(**all_loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


        if cur_iteration>=max_iteration and max_iteration>0:
            break

        if visdir is not None and cur_iteration%visual_freq==0:
            fs=int(len(visual_dict)**0.5)+1
            for idx, k in enumerate(visual_dict.keys()):
                plt.subplot(fs,fs,idx+1)
                plt.title(k)
                plt.axis('off')
                if k not in ['GT','GLA_pred','SBF_pred']:
                    plt.imshow(visual_dict[k], cmap='gray')
                else:
                    plt.imshow(visual_dict[k], vmin=0, vmax=4)
            plt.tight_layout()
            plt.savefig(f'{visdir}/{cur_iteration}.png')
            plt.close()
        cur_iteration+=1

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return cur_iteration


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _random_affine_2d(images, max_rotate, max_translate, max_scale):
    if max_rotate <= 0 and max_translate <= 0 and max_scale <= 0:
        return images
    B, C, H, W = images.shape
    device = images.device
    angles = (torch.rand(B, device=device) - 0.5) * 2 * math.radians(max_rotate)
    scales = 1.0 + (torch.rand(B, device=device) - 0.5) * 2 * max_scale
    tx = (torch.rand(B, device=device) - 0.5) * 2 * max_translate
    ty = (torch.rand(B, device=device) - 0.5) * 2 * max_translate

    cos_a = torch.cos(angles) * scales
    sin_a = torch.sin(angles) * scales

    theta = torch.zeros(B, 2, 3, device=device, dtype=images.dtype)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty

    grid = F.affine_grid(theta, size=images.size(), align_corners=False)
    return F.grid_sample(images, grid, mode='bilinear', padding_mode='border', align_corners=False)


def _apply_light_slaug(images, slaug_cfg):
    """Lightweight SLAug: mild affine + intensity jitter + optional flip."""
    out = images.clone()
    flip = _cfg_get(slaug_cfg, "flip", False)
    if flip:
        flip_mask = torch.rand(out.shape[0], device=out.device) > 0.5
        if flip_mask.any():
            out[flip_mask] = torch.flip(out[flip_mask], dims=[3])

    max_rotate = _cfg_get(slaug_cfg, "max_rotate", 5.0)
    max_translate = _cfg_get(slaug_cfg, "max_translate", 0.02)
    max_scale = _cfg_get(slaug_cfg, "max_scale", 0.05)
    out = _random_affine_2d(out, max_rotate, max_translate, max_scale)

    brightness = _cfg_get(slaug_cfg, "brightness", 0.1)
    contrast = _cfg_get(slaug_cfg, "contrast", 0.1)
    noise_std = _cfg_get(slaug_cfg, "noise_std", 0.02)

    if brightness > 0:
        shift = (torch.rand(out.shape[0], 1, 1, 1, device=out.device) - 0.5) * 2 * brightness
        out = out + shift
    if contrast > 0:
        scale = 1.0 + (torch.rand(out.shape[0], 1, 1, 1, device=out.device) - 0.5) * 2 * contrast
        out = out * scale
    if noise_std > 0:
        out = out + torch.randn_like(out) * noise_std
    return out


def _generate_slaug_views(images, num_views, slaug_cfg):
    return [_apply_light_slaug(images, slaug_cfg) for _ in range(num_views)]


def _compute_core_mask(model, criterion, images, labels, core_cfg):
    """Compute core mask via multi-view saliency + prediction stability."""
    num_views = _cfg_get(core_cfg, "num_views", 3)
    saliency_tau = _cfg_get(core_cfg, "saliency_threshold", 0.6)
    pred_var_tau = _cfg_get(core_cfg, "pred_var_threshold", 0.02)
    saliency_grid = _cfg_get(core_cfg, "saliency_grid", 16)
    use_pred_consistency = _cfg_get(core_cfg, "use_pred_consistency", True)
    slaug_cfg = _cfg_get(core_cfg, "slaug", None)

    saliency_maps = []
    preds = []
    for view in _generate_slaug_views(images, num_views, slaug_cfg):
        view = view.clone().detach().requires_grad_(True)
        logits_view = model(view)
        prob_view = torch.softmax(logits_view, dim=1)
        preds.append(prob_view.detach())

        loss_dict = criterion.get_loss(logits_view, labels)
        loss_view = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)
        grad_inputs = torch.autograd.grad(loss_view, view, retain_graph=False, create_graph=False)[0]
        grad_mag = torch.sqrt(torch.mean(grad_inputs ** 2, dim=1, keepdim=True) + 1e-12)
        saliency_maps.append(get_SBF_map(grad_mag, saliency_grid).detach())

    saliency_stack = torch.stack(saliency_maps, dim=0)
    saliency_mean = saliency_stack.mean(dim=0)
    core_mask = (saliency_mean >= saliency_tau).float()

    var_map = None
    if use_pred_consistency and pred_var_tau is not None and pred_var_tau > 0:
        preds_stack = torch.stack(preds, dim=0)  # [K, B, C, H, W]
        var_map = preds_stack.var(dim=0).mean(dim=1, keepdim=True)
        stable_mask = (var_map <= pred_var_tau).float()
        core_mask = core_mask * stable_mask

    return core_mask.clamp(0.0, 1.0).detach(), saliency_mean.detach(), var_map.detach() if var_map is not None else None


def train_one_epoch_core_fbd(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, cur_iteration:int, max_iteration: int = -1,
                    core_cfg=None, grad_scaler=None):
    """
    Core-guided FBD training with SLAug views, core invariance, and non-core regularization.
    """
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}][CoreFBD]'.format(epoch)
    print_freq = 10

    freq_cfg = _cfg_get(core_cfg, "frequency", core_cfg)
    loss_cfg = _cfg_get(core_cfg, "loss", core_cfg)

    lambda_core_inv = _cfg_get(loss_cfg, "lambda_core_inv", 1.0)
    lambda_noncore_reg = _cfg_get(loss_cfg, "lambda_noncore_reg", 0.0)
    noncore_margin = _cfg_get(loss_cfg, "noncore_margin", 0.7)

    p_fourier = _cfg_get(freq_cfg, "p_fourier", _cfg_get(freq_cfg, "p", 0.5))
    fourier_sigma = _cfg_get(freq_cfg, "sigma", 0.08)
    alpha_range = tuple(_cfg_get(freq_cfg, "alpha_range", (0.7, 1.0)))
    eps_max = _cfg_get(freq_cfg, "eps_max", 0.3)
    num_bands = _cfg_get(freq_cfg, "num_bands", 4)
    band_weights = _cfg_get(freq_cfg, "band_weights", [1.0 for _ in range(num_bands)])
    lambda_core = _cfg_get(freq_cfg, "lambda_core", 0.05)
    lambda_noncore = _cfg_get(freq_cfg, "lambda_noncore", 1.0)

    band_masks_cache = None

    for i, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        for k, v in samples.items():
            if isinstance(samples[k], torch.Tensor):
                samples[k] = v.to(device)

        images = samples['images']
        labels = samples['labels']

        if band_masks_cache is None:
            _, _, H, W = images.shape
            band_masks_cache = build_radial_band_masks(H, W, num_bands)

        core_mask, _, _ = _compute_core_mask(model, criterion, images, labels, core_cfg)
        images_fbd = core_guided_fourier_augment(
            images,
            core_mask,
            p_fourier=p_fourier,
            sigma=fourier_sigma,
            alpha_range=alpha_range,
            band_masks=band_masks_cache,
            band_weights=band_weights,
            eps_max=eps_max,
            lambda_core=lambda_core,
            lambda_noncore=lambda_noncore,
        )

        if grad_scaler is None:
            logits = model(images)
            logits_fbd = model(images_fbd)
            loss_dict = criterion.get_loss(logits, labels)
            loss_dict_fbd = criterion.get_loss(logits_fbd, labels)

            seg_loss = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)
            seg_loss_fbd = sum(loss_dict_fbd[k] * criterion.weight_dict[k] for k in loss_dict_fbd.keys() if k in criterion.weight_dict)
            prob = torch.softmax(logits, dim=1)
            prob_fbd = torch.softmax(logits_fbd, dim=1)
            core_mask_exp = core_mask if core_mask.ndimension() == 4 else core_mask.unsqueeze(1)
            core_pixels = core_mask_exp.sum() + 1e-6
            core_inv = ((prob - prob_fbd) ** 2 * core_mask_exp).sum() / core_pixels

            noncore_loss = torch.tensor(0.0, device=device)
            if lambda_noncore_reg > 0:
                non_mask = (1 - core_mask_exp).clamp(0.0, 1.0)
                non_pixels = non_mask.sum()
                fg_prob = prob_fbd[:, 1:, ...]
                max_fg, _ = torch.max(fg_prob, dim=1, keepdim=True)
                reg = torch.clamp(max_fg - noncore_margin, min=0.0)
                noncore_loss = (reg * non_mask).sum() / (non_pixels + 1e-6)

            total_loss = seg_loss + seg_loss_fbd + lambda_core_inv * core_inv + lambda_noncore_reg * noncore_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        else:
            with torch.cuda.amp.autocast():
                logits = model(images)
                logits_fbd = model(images_fbd)
                loss_dict = criterion.get_loss(logits, labels)
                loss_dict_fbd = criterion.get_loss(logits_fbd, labels)
                seg_loss = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)
                seg_loss_fbd = sum(loss_dict_fbd[k] * criterion.weight_dict[k] for k in loss_dict_fbd.keys() if k in criterion.weight_dict)
                prob = torch.softmax(logits, dim=1)
                prob_fbd = torch.softmax(logits_fbd, dim=1)
                core_mask_exp = core_mask if core_mask.ndimension() == 4 else core_mask.unsqueeze(1)
                core_pixels = core_mask_exp.sum() + 1e-6
                core_inv = ((prob - prob_fbd) ** 2 * core_mask_exp).sum() / core_pixels

                noncore_loss = torch.tensor(0.0, device=device)
                if lambda_noncore_reg > 0:
                    non_mask = (1 - core_mask_exp).clamp(0.0, 1.0)
                    non_pixels = non_mask.sum()
                    fg_prob = prob_fbd[:, 1:, ...]
                    max_fg, _ = torch.max(fg_prob, dim=1, keepdim=True)
                    reg = torch.clamp(max_fg - noncore_margin, min=0.0)
                    noncore_loss = (reg * non_mask).sum() / (non_pixels + 1e-6)

                total_loss = seg_loss + seg_loss_fbd + lambda_core_inv * core_inv + lambda_noncore_reg * noncore_loss

            optimizer.zero_grad()
            grad_scaler.scale(total_loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

        metric_logger.update(
            ce=loss_dict['ce_loss'],
            dice=loss_dict['dice_loss'],
            ce_fbd=loss_dict_fbd['ce_loss'],
            dice_fbd=loss_dict_fbd['dice_loss'],
            core_inv=core_inv,
            noncore=noncore_loss,
        )
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        cur_iteration += 1

        if cur_iteration >= max_iteration and max_iteration > 0:
            break

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return cur_iteration


@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()
    def convert_to_one_hot(tensor,num_c):
        return F.one_hot(tensor,num_c).permute((0,3,1,2))
    dices=[]
    for samples in data_loader:
        for k, v in samples.items():
            if isinstance(samples[k], torch.Tensor):
                samples[k] = v.to(device)
        img = samples['images']
        lbl = samples['labels']
        logits = model(img)
        num_classes=logits.size(1)
        pred=torch.argmax(logits,dim=1)
        one_hot_pred=convert_to_one_hot(pred,num_classes)
        one_hot_gt=convert_to_one_hot(lbl,num_classes)
        dice=compute_meandice(one_hot_pred,one_hot_gt,include_background=False)
        dices.append(dice.cpu().numpy())
    dices=np.concatenate(dices,0)
    dices=np.nanmean(dices,0)
    return dices

def prediction_wrapper(model, test_loader, epoch, label_name, mode='base', save_prediction=False, device=None):
    """
    A wrapper for the ease of evaluation
    Args:
        model:          Module The network to evalute on
        test_loader:    DataLoader Dataloader for the dataset to test
        mode:           str Adding a note for the saved testing results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        out_prediction_list = {} # a buffer for saving results
        # recomp_img_list = []
        for idx, batch in tqdm(enumerate(test_loader), total = len(test_loader)):
            if batch['is_start']:
                slice_idx = 0

                scan_id_full = str(batch['scan_id'][0])
                out_prediction_list[scan_id_full] = {}

                nframe = batch['nframe']
                nb, nc, nx, ny = batch['images'].shape
                curr_pred = torch.tensor(np.zeros([nframe, nx, ny]), device=device) # nb/nz, nc, nx, ny
                curr_gth = torch.tensor(np.zeros([nframe, nx, ny]), device=device)
                curr_img = np.zeros( [nx, ny, nframe]  )

            assert batch['labels'].shape[0] == 1 # enforce a batchsize of 1

            img = batch['images'].to(device)
            gth = batch['labels'].to(device)

            pred = model(img)
            pred=torch.argmax(pred,1)
            curr_pred[slice_idx, ...]   = pred[0, ...] # nb (1), nc, nx, ny
            curr_gth[slice_idx, ...]    = gth[0, ...]
            curr_img[:,:,slice_idx] = batch['images'][0, 0,...].numpy()
            slice_idx += 1

            if batch['is_end']:
                out_prediction_list[scan_id_full]['pred'] = curr_pred
                out_prediction_list[scan_id_full]['gth'] = curr_gth
                # if opt.phase == 'test':
                #     recomp_img_list.append(curr_img)

        print("Epoch {} test result on mode {} segmentation are shown as follows:".format(epoch, mode))
        error_dict, dsc_table, domain_names = eval_list_wrapper(out_prediction_list, len(label_name),label_name)
        error_dict["mode"] = mode
        if not save_prediction: # to save memory
            del out_prediction_list
            out_prediction_list = []
        torch.cuda.empty_cache()

    return out_prediction_list, dsc_table, error_dict, domain_names

def eval_list_wrapper(vol_list, nclass, label_name):
    """
    Evaluatation and arrange predictions
    """
    def convert_to_one_hot2(tensor,num_c):
        return F.one_hot(tensor.long(),num_c).permute((3,0,1,2)).unsqueeze(0)

    out_count = len(vol_list)
    tables_by_domain = {} # tables by domain
    dsc_table = np.ones([ out_count, nclass ]  ) # rows and samples, columns are structures
    idx = 0
    for scan_id, comp in vol_list.items():
        domain, pid = scan_id.split("_")
        if domain not in tables_by_domain.keys():
            tables_by_domain[domain] = {'scores': [],'scan_ids': []}
        pred_ = comp['pred']
        gth_  = comp['gth']
        dices=compute_meandice(y_pred=convert_to_one_hot2(pred_,nclass),y=convert_to_one_hot2(gth_,nclass),include_background=True).cpu().numpy()[0].tolist()

        tables_by_domain[domain]['scores'].append( [_sc for _sc in dices]  )
        tables_by_domain[domain]['scan_ids'].append( scan_id )
        dsc_table[idx, ...] = np.reshape(dices, (-1))
        del pred_
        del gth_
        idx += 1
        torch.cuda.empty_cache()

    # then output the result
    error_dict = {}
    for organ in range(nclass):
        mean_dc = np.mean( dsc_table[:, organ] )
        std_dc  = np.std(  dsc_table[:, organ] )
        print("Organ {} with dice: mean: {:06.5f}, std: {:06.5f}".format(label_name[organ], mean_dc, std_dc))
        error_dict[label_name[organ]] = mean_dc
    print("Overall std dice by sample {:06.5f}".format(dsc_table[:, 1:].std()))
    print("Overall mean dice by sample {:06.5f}".format( dsc_table[:,1:].mean())) # background is noted as class 0 and therefore not counted
    error_dict['overall'] = dsc_table[:,1:].mean()

    # then deal with table_by_domain issue
    overall_by_domain = []
    domain_names = []
    for domain_name, domain_dict in tables_by_domain.items():
        domain_scores = np.array( tables_by_domain[domain_name]['scores']  )
        domain_mean_score = np.mean(domain_scores[:, 1:])
        error_dict[f'domain_{domain_name}_overall'] = domain_mean_score
        error_dict[f'domain_{domain_name}_table'] = domain_scores
        overall_by_domain.append(domain_mean_score)
        domain_names.append(domain_name)
    print('per domain resutls:', overall_by_domain)
    error_dict['overall_by_domain'] = np.mean(overall_by_domain)

    print("Overall mean dice by domain {:06.5f}".format( error_dict['overall_by_domain'] ) )
    return error_dict, dsc_table, domain_names

