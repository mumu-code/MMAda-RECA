import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW
from lightning.pytorch.utilities import CombinedLoader

from transformers import AutoTokenizer, AutoConfig
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed

from training.data import Text2ImageDataset
from training.utils import get_config, flatten_omega_conf, image_transform
from training.imagenet_dataset import ImageNetDataset
from parquet import RefinedWebDataset

from models import MAGVITv2, get_mask_schedule, MMadaModelLM, MMadaConfig
from training.prompting_utils import UniversalPrompting
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from training.consts import get_recon_prompt_list, get_prompt_list
from parquet import VQADataset
import copy


SYSTEM_PROMPT_LEN = 28

from training.utils import get_config, flatten_omega_conf, mask_or_random_replace_tokens, AverageMeter
import random
from training.consts import get_recon_prompt_list, get_prompt_list
recon_prompt_list = get_recon_prompt_list()

try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    elif model_type == "vq16":
        return VQ_16
    else:
        raise ValueError(f"model_type {model_type} not supported.")
    
#构造输入,训练时size设置为256
def prepare_input_for_recon(device,image_path,prompt,vq_model,uni_prompting):

    image_ori = Image.open(image_path).convert('RGB')
    #压缩为256，256
    image = image_transform(image_ori,resolution = 256).to(device)
    image = image.unsqueeze(0)
    image_tokens_resized = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)
    input_ids = uni_prompting.text_tokenizer(['<|start_header_id|>user<|end_header_id|>\n' + prompt  +'<eot_id><|start_header_id|>assistant<|end_header_id|>\n'])['input_ids']
    input_ids = input_ids.to(device)

    input_ids = torch.cat([
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(device),
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(device),
            image_tokens_resized,
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(device),
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|sot|>']).to(device),
            input_ids
        ], dim=1).long()
# 将两组图像都 resize 到相同尺寸
def resize_images(img, target_h, target_w):
    if isinstance(img, Image.Image):
        img = img.resize((target_h, target_w), Image.BILINEAR)
        img = np.array(img)
    elif isinstance(img, np.ndarray):
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((target_h, target_w), Image.BILINEAR)
        img = np.array(pil_img)
    else:
        raise TypeError("Unsupported image type")
    if img.shape[-1] == 3:  # (H, W, 3)
        pass
    elif img.shape[0] == 3:  # (3, H, W) → 转为 (H, W, 3)
        img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.uint8)
    return img

    # resized_list = []
    # for img in img_array:
    #     pil_img = Image.fromarray(img)
    #     pil_img = pil_img.resize((target_w, target_h), Image.BILINEAR)  # 或 Image.LANCZOS 更锐利
    #     resized_list.append(np.array(pil_img))
    # return np.array(resized_list)

def main():
    set_seed(12)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ######1.加载模型
    model_path = "/data3/MMaDA-8B/"
    vq_model_path = "/data3/magvitv2"
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    uni_prompting = UniversalPrompting(tokenizer, max_text_len=377,
                                       special_tokens=(
                                           "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
                                           "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
                                       ),
                                       ignore_id=-100, use_reserved_token=True)
    print('special tokens : \n', uni_prompting.sptids_dict)
    vq_model = MAGVITv2.from_pretrained(vq_model_path).to(device)
    #先加载模型结构
    #1.未训练模型
    # model = MMadaModelLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    #2.加载训练后参数
    save_dir = '/data1/recon/recon-checkpoint-16000'
    model = MMadaModelLM.from_pretrained(model_path, trust_remote_code=True, state_dict=None,torch_dtype=torch.bfloat16)
    state_dict = torch.load("/data1/MMADA_RECA/outputs/null/checkpoint-16000/fp32_ckpt/pytorch_model.bin",map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    #####################################################
    model.to(device)
    vq_model.eval()
    vq_model.requires_grad_(False)

    mask_id = model.config.mask_token_id
    guidance_scale = 4.0
    #########2.重建输入构造
    img_root = "/home/xl/project/MaskGRPO/generated_samples/simple/tiger/"
    num_vq_tokens = 1024
    codebook_size = 8192

    images_path = os.listdir(img_root)
    from tqdm import tqdm
    batch_size = 2
    os.makedirs(save_dir, exist_ok=True)


    for step in tqdm(range(0,len(images_path),batch_size)):
        prompts = []
        for i in range(batch_size):
            # prompt = random.choice(recon_prompt_list)
            prompt = "Note: quality, genre, colors, shapes, sizes, materials, numbers, words, positions, scene"
            prompts.append(prompt)
        images_tokens_given = []
        images_ori = []
        for i in range(step,step+batch_size):
            image_ori = Image.open(img_root+images_path[i]).convert('RGB')
            images_ori.append(image_ori)
            image = image_transform(image_ori,resolution=256).unsqueeze(0).to(device)
            image_token_resized = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)
            images_tokens_given.append(image_token_resized)

        images_tokens_given = torch.cat(images_tokens_given,dim=0)
        image_tokens = torch.ones((len(prompts), num_vq_tokens),
                                    dtype=torch.long, device=device) * mask_id
        image_tokens = image_tokens.to(device)
        images_tokens_given = torch.tensor(images_tokens_given,dtype=torch.long, device=device)

        input_ids, attention_mask,labels = uni_prompting((prompts, image_tokens,images_tokens_given), 't2i_recon_gen')
        
        #CFG
        if guidance_scale > 0:
            uncond_input_ids, uncond_attention_mask,un_labels = uni_prompting(([' describe the image in detail.'] * len(prompts), image_tokens,(torch.ones((len(prompts), 256),
                                    dtype=torch.long, device=device) * uni_prompting.text_tokenizer.pad_token_id)), 't2i_recon_gen')
        else:
            uncond_input_ids = None
            uncond_attention_mask = None

        #查看的配置
        mask_schedule = get_mask_schedule("cosine")
        with torch.no_grad():
            gen_token_ids = model.recon_generate(
                input_ids=input_ids,
                uncond_input_ids = uncond_input_ids,
                attention_mask=attention_mask,
                uncond_attention_mask=uncond_attention_mask,
                # temperature=generation_temperature,
                guidance_scale = 4.0,
                timesteps=18,
                noise_schedule=mask_schedule,
                # noise_type=noise_type,
                seq_len=num_vq_tokens,
                uni_prompting=uni_prompting,
                # config=config,
            )
        gen_token_ids = torch.clamp(gen_token_ids, max=codebook_size - 1, min=0)
        images_gen = vq_model.decode_code(gen_token_ids)
        images_gen = torch.clamp((images_gen + 1.0) / 2.0, min=0.0, max=1.0)
        images_gen *= 255.0
        images_gen = images_gen.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

        images_cond = vq_model.decode_code(images_tokens_given-len(uni_prompting.text_tokenizer))
        images_cond = torch.clamp((images_cond + 1.0) / 2.0, min=0.0, max=1.0)
        images_cond *= 255.0
        images_cond = images_cond.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

        images_resized_uniform = np.stack([resize_images(img, 256,256) for img in images_ori])        # (B, 256, 256, 3)
        images_cond_resized = np.stack([resize_images(img,  256,256) for img in images_cond])      # (B, 256, 256, 3)
        images_gen_resized = np.stack([resize_images(img,  256,256) for img in images_gen])

        combined = np.concatenate((images_resized_uniform,images_cond_resized, images_gen_resized), axis=2)
        for idx, img_np in enumerate(combined):
            pil_combined = Image.fromarray(img_np)
            pil_combined.save(os.path.join(save_dir, f"compare_{(step+idx):06d}.png"))


if __name__ == '__main__':
    main()

    








    
    
