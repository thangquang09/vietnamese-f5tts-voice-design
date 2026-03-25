import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
from einops import rearrange
from cached_path import cached_path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# replace this with BigVGAN
import bigvgan

from model.modules import MelSpec
from network.crossdit import CrossDiT
from dataset.capspeech import CapSpeech
from utils import load_checkpoint, make_pad_mask
from utils import get_lr_scheduler, load_yaml_with_includes
from inference import eval_model
from dataset.pool_sampler import build_stage2_v3_sampler


def parse_args():
    parser = argparse.ArgumentParser()

    # Config settings
    parser.add_argument('--config-name', type=str, required=True)
    parser.add_argument('--pretrained-ckpt', type=str, required=True)

    # Training settings
    parser.add_argument("--amp", type=str, default='fp16')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--num-workers', type=int, default=32)
    parser.add_argument('--num-threads', type=int, default=1)
    parser.add_argument('--eval-every-step', type=int, default=1000)
    # save all states including optimizer every save-every-step
    parser.add_argument('--save-every-step', type=int, default=1000)
    parser.add_argument('--max-ckpts', type=int, default=3, help='Max checkpoints to keep (oldest deleted)')
    parser.add_argument('--resume-from', type=str, default=None, help='Path to checkpoint to resume training')
    parser.add_argument('--train-sampler', type=str, default='shuffle', choices=['shuffle', 'stage2_v3'])
    parser.add_argument('--sampler-recipe', type=str, default=None,
                        help='Path to stage2-v3 recipe YAML when using --train-sampler stage2_v3')

    # Log and random seed
    parser.add_argument('--random-seed', type=int, default=2025)
    parser.add_argument('--log-step', type=int, default=200)
    parser.add_argument('--log-dir', type=str, default='./logs/')
    parser.add_argument('--save-dir', type=str, default='./ckpts/')
    return parser.parse_args()


def setup_directories(args, params):
    args.log_dir = os.path.join(args.log_dir, params['model_name']) + '/'
    args.save_dir = os.path.join(args.save_dir, params['model_name']) + '/'

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)


def set_device(args):
    torch.set_num_threads(args.num_threads)
    if torch.cuda.is_available():
        args.device = 'cuda'
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        args.device = 'cpu'


def prepare_batch(batch, mel, latent_sr):
    x, x_lens, y, y_lens, c, c_lens, tag = batch["x"], batch["x_lens"], batch["y"], batch["y_lens"], batch["c"], batch["c_lens"], batch["tag"]

    # add len for clap embedding
    x_lens = x_lens + 1

    with torch.no_grad():
        audio_clip = mel(y)

        audio_clip = rearrange(audio_clip, 'b d n -> b n d')
        y_lens = (y_lens * latent_sr).long()

    return x, x_lens, audio_clip, y_lens, c, c_lens, tag


if __name__ == '__main__':

    args = parse_args()
    params = load_yaml_with_includes(args.config_name)

    # random seed
    set_device(args)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    accelerator = Accelerator(mixed_precision=args.amp,
                              gradient_accumulation_steps=params['opt']['accumulation_steps'],
                              step_scheduler_with_optimizer=False)

    # dataset
    train_set = CapSpeech(**params['data']['trainset'])
    train_sampler = None
    train_shuffle = True
    if args.train_sampler == 'stage2_v3':
        if not args.sampler_recipe:
            raise ValueError("--sampler-recipe is required when --train-sampler stage2_v3")
        train_sampler = build_stage2_v3_sampler(
            dataset=train_set,
            recipe_path=args.sampler_recipe,
            split='train',
            seed=args.random_seed,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
        )
        train_shuffle = False

    train_loader = DataLoader(train_set, num_workers=args.num_workers,
                              batch_size=params['opt']['batch_size'], shuffle=train_shuffle,
                              sampler=train_sampler,
                              collate_fn=train_set.collate)

    val_set = CapSpeech(**params['data']['valset'])
    val_loader = DataLoader(val_set, num_workers=0,
                            batch_size=1, shuffle=False,
                            collate_fn=val_set.collate)
                            
    # load dit
    model = CrossDiT(**params['model'])
    
    # Load pretrained checkpoint, handling shape mismatches (e.g. text_embed: 201 vs 177)
    # PyTorch strict=False still errors on shape mismatch, so we manually filter
    pretrained_state = torch.load(args.pretrained_ckpt, map_location='cpu')["model"]
    model_state = model.state_dict()
    
    # Filter out keys with shape mismatch
    filtered_state = {}
    skipped_keys = []
    for key, value in pretrained_state.items():
        if key in model_state and value.shape == model_state[key].shape:
            filtered_state[key] = value
        else:
            skipped_keys.append(f"{key}: ckpt={value.shape} vs model={model_state.get(key, 'MISSING')}")
    
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    if accelerator.is_main_process:
        print(f"Loaded {len(filtered_state)}/{len(pretrained_state)} pretrained weights")
        if skipped_keys:
            print(f"Skipped {len(skipped_keys)} keys (shape mismatch, will be randomly initialized):")
            for s in skipped_keys:
                print(f"  → {s}")
        if missing:
            print(f"Missing keys (random init): {len(missing)}")

    # mel spectrogram - move to accelerator device after preparation
    mel = MelSpec(**params['mel'])
    latent_sr = params['mel']['target_sample_rate'] / params['mel']['hop_length']

    # load vocoder
    vocoder = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_24khz_100band_256x', use_cuda_kernel=False)
    vocoder.remove_weight_norm()
    vocoder = vocoder.eval().to(accelerator.device)

    # prepare opt
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['opt']['learning_rate'])

    if args.resume_from is not None and os.path.exists(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location='cpu')
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint["global_step"]
        start_epoch = checkpoint["epoch"] + 1  # Continue from the next epoch
        print(f"Resuming training from checkpoint: {args.resume_from}, starting from epoch {start_epoch}.")
    else:
        global_step = 0
        start_epoch = 0
    
    lr_scheduler = get_lr_scheduler(optimizer, 'customized', **params['opt']['lr_scheduler'])
    
    # Prepare with accelerator
    (model, optimizer, lr_scheduler, 
     train_loader, val_loader) = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader)
    
    # Move mel and vocos to the same device as model AFTER preparation
    mel = mel.to(accelerator.device)
    vocoder = vocoder.to(accelerator.device)

    # Add synchronization point
    accelerator.wait_for_everyone()

    losses = 0.0

    if accelerator.is_main_process:
        setup_directories(args, params)
        trainable_params = sum(param.nelement() for param in model.parameters() if param.requires_grad)
        print("Number of trainable parameters: %.2fM" % (trainable_params / 1e6))
    
    # Add synchronization point
    accelerator.wait_for_everyone()

    # REMOVED initial evaluation to prevent deadlock
    # We'll evaluate after the first epoch or at the first eval step

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        model.train()
        
        # Use accelerator's progress bar for correct handling in distributed setup
        progress_bar = tqdm(train_loader, disable=not accelerator.is_local_main_process)
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                (text, text_lens, audio_clips, audio_lens, prompt, prompt_lens, clap) = prepare_batch(batch, mel, latent_sr)
                # prepare flow mathing
                x1 = audio_clips
                x0 = torch.randn_like(x1)
                t = torch.rand((x1.shape[0],), dtype=x1.dtype, device=x1.device)
                sigma = rearrange(t, 'b -> b 1 1')
                noisy_x1 = (1 - sigma) * x0.clone() + sigma * x1.clone()
                flow = x1.clone() - x0.clone()
                # option: audio-prompt based zero-shot tts
                # tts_mask = create_tts_mask(seq_len, x1.shape[1], params['opt']['mask_range'])
                # # cond = x1.clone(), cond[tts_mask[..., None]] = 0
                # cond = torch.where(tts_mask[..., None], torch.zeros_like(x1), x1)
                cond = None

                # prepare batch cfg
                drop_prompt = (torch.rand(x1.shape[0]) < params['opt']['drop_spk'])
                drop_text = drop_prompt & (torch.rand(x1.shape[0]) < params['opt']['drop_text'])

                prompt[drop_prompt] = 0.0
                prompt_lens[drop_prompt] = 1
                clap[drop_text] = 0.0
                text[drop_text] = -1

                seq_len_audio = audio_clips.shape[1]
                pad_mask = make_pad_mask(audio_lens, seq_len_audio).to(audio_clips.device)

                seq_len_prompt = prompt.shape[1]
                prompt_mask = make_pad_mask(prompt_lens, seq_len_prompt).to(prompt.device)

                pred = model(x=noisy_x1, cond=cond,
                             prompt=prompt, clap=clap, text=text, time=t,
                             mask=pad_mask, prompt_mask=prompt_mask)

                loss = F.mse_loss(pred, flow, reduction="none")
                loss = loss[pad_mask].mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if 'grad_clip' in params['opt'] and params['opt']['grad_clip'] > 0:
                        accelerator.clip_grad_norm_(model.parameters(),
                                                    max_norm=params['opt']['grad_clip'])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Fixed step counting - increment only once per actual step, not per accumulation step
            if accelerator.sync_gradients:
                global_step += 1
                losses += loss.item()

                # Add progress bar description
                if accelerator.is_local_main_process:
                    progress_bar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

                if global_step % args.log_step == 0:
                    losses = losses / args.log_step  # Calculate average loss
                    
                    if accelerator.is_main_process:
                        current_time = time.asctime(time.localtime(time.time()))
                        epoch_info = f'Epoch: [{epoch + 1}][{args.epochs}]'
                        batch_info = f'Global Step: {global_step}'
                        loss_info = f'Loss: {losses:.6f}'

                        # Extract the learning rate from the optimizer
                        lr = optimizer.param_groups[0]['lr']
                        lr_info = f'Learning Rate: {lr:.6f}'

                        log_message = f'{current_time}\n{epoch_info}    {batch_info}    {loss_info}    {lr_info}\n'

                        with open(args.log_dir + 'log.txt', mode='a') as n:
                            n.write(log_message)

                    # Reset loss accumulator
                    losses = 0.0
                
                # Save checkpoint logic (independent of eval)
                if global_step % args.save_every_step == 0:
                    if accelerator.is_main_process:
                        unwrapped_model = accelerator.unwrap_model(model)
                        
                        # Save model checkpoint
                        ckpt_path = args.save_dir + str(global_step) + '.pt'
                        accelerator.save({
                            "model": unwrapped_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "global_step": global_step,
                        }, ckpt_path)
                        
                        # Rotate checkpoints: keep only max_ckpts
                        import glob
                        ckpt_files = sorted(glob.glob(args.save_dir + '*.pt'), key=os.path.getmtime)
                        while len(ckpt_files) > args.max_ckpts:
                            old_ckpt = ckpt_files.pop(0)
                            os.remove(old_ckpt)
                            print(f"Removed old checkpoint: {old_ckpt}")

                # Evaluation logic (separate from save)
                if global_step % args.eval_every_step == 0:
                    # Set model to eval mode
                    model.eval()
                    
                    # Synchronize before evaluation
                    accelerator.wait_for_everyone()
                    
                    if accelerator.is_main_process:
                        unwrapped_model = accelerator.unwrap_model(model)
                        
                        # Run evaluation
                        eval_model(unwrapped_model, vocoder, mel, val_loader, params,
                                   steps=25, cfg=2.0,
                                   sway_sampling_coef=-1.0, 
                                   epoch=global_step, save_path=args.log_dir + 'output/', val_num=1)
                    
                    # Synchronize after evaluation
                    accelerator.wait_for_everyone()
                    
                    # Set model back to train mode
                    model.train()

        # Synchronize at the end of each epoch
        accelerator.wait_for_everyone()
