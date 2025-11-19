import argparse
import os
import json
import sys

from datetime import datetime

import torch
from PIL import Image
from torch.utils.data import  Dataset, DataLoader
from datasets import load_dataset
from tqdm.auto import tqdm

from metrics import metrics, imagenet_accuracy
from utils.attack_utils import inject_attribute_backdoor
from utils.config_parser import ConfigParser
from utils.stable_diffusion_utils import generate
from sklearn.model_selection import train_test_split
import time
os.environ['CUDA_VISIBLE_DEVICES'] = "5"

# python abalation.py  -c="configs/default_NSFW.yaml"
# CUDA_VISIBLE_DEVICES=2 python abalation.py  -c="configs/default_NSFW.yaml"


       

def read_file(filename):
    with open(filename) as f:
        data = f.read().splitlines()
    return data
        
    
def main():
    # define and parse arguments
    config, config_path = create_parser()
    torch.manual_seed(config.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    torch.set_num_threads(config.training['num_threads'])

    rtpt = config.create_rtpt()
    rtpt.start()

    # load dataset
    data_laion = config.load_datasets()
    file_beadv = "datasets/adversarial-datasets/adbenign.txt"
    data_beadv = read_file(file_beadv)
    dataset = data_laion + data_beadv
    dataloader = DataLoader(dataset,
                            batch_size=config.clean_batch_size,
                            shuffle=True)

    file_raw = "datasets/train/i2p_violence_prompts_train.txt"
    file_gen = "datasets/train/i2p_violence_prompts_gen.txt"
    file_adv = "datasets/adversarial-datasets/adversa.txt"
    data_raw = read_file(file_raw)
    data_gen = read_file(file_gen)
    data_adv = read_file(file_adv)
    data = data_raw + data_gen + data_adv
    data_file = file_raw
    
    clean_file = 'metrics/captions_10000.txt'
    captions_clean = read_file(clean_file)

    nsfw_file = "datasets/test/i2p_violence_prompts_test.txt"
    captions_nsfw = read_file(nsfw_file)
    
    
    dataset = config.load_datasets()

    dataloader = DataLoader(dataset,
                            batch_size=config.clean_batch_size,
                            shuffle=True)
    backdoor_dataloader = DataLoader(data,
                            batch_size=config.poison_batch_size,
                            shuffle=True)


    list_loss_weight = [0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]


    

    # load models
    tokenizer = config.load_tokenizer()
    encoder_teacher = config.load_text_encoder().to(device)
    # encoder_student = config.load_text_encoder().to(device)
    # freeze teacher model
    for param in encoder_teacher.parameters():
        param.requires_grad = False



    # define loss function
    loss_fkt = config.loss_fkt



    # prepare training
    result = []
    file_parts = data_file.split('.')
    file_name = file_parts[0] + "-ablation." + file_parts[1]
    for loss_weight in list_loss_weight:
        # encoder_teacher = config.load_text_encoder().to(device)
        encoder_student = config.load_text_encoder().to(device)
        # define optimizer
        optimizer = config.create_optimizer(encoder_student)
        lr_scheduler = config.create_lr_scheduler(optimizer)
        
        # prepare training
        num_clean_samples = 0
        num_backdoored_samples = 0
        step = -1
        encoder_student.train()
        encoder_teacher.eval()
        dataloader_iter = iter(dataloader)
        backdoor_dataloader_iter = iter(backdoor_dataloader)

        # training loop
        while (True):
            step += 1

            # stop if max num of steps reached
            if step >= config.num_steps:
                break

            
            batch_clean = []
            while len(batch_clean) < config.clean_batch_size:
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(dataloader)
                    batch = next(dataloader_iter)
                for backdoor in config.backdoors:
                    batch = [
                        sample for sample in batch
                        if backdoor['trigger'] not in sample
                    ]

                batch_clean += batch
            batch_clean = batch_clean[:config.clean_batch_size]

            # compute utility loss
            num_clean_samples += len(batch_clean)
            text_input = tokenizer(batch_clean,
                                padding="max_length",
                                max_length=tokenizer.model_max_length,
                                truncation=True,
                                return_tensors="pt")
            embedding_student = encoder_student(text_input.input_ids.to(device))[0]
            with torch.no_grad():
                embedding_teacher = encoder_teacher(
                    text_input.input_ids.to(device))[0]

            loss_benign = loss_fkt(embedding_student, embedding_teacher)

            # compute backdoor losses for all distinct backdoors
            backdoor_losses = []

            # get next backdoor batch 
            # batch_backdoor = next(backdoor_dataloader_iter)
            batch_backdoor = []
            while len(batch_backdoor) < config.injection['poisoned_samples_per_step']:
                try:
                    batch0 = next(backdoor_dataloader_iter)
                except StopIteration:
                    backdoor_dataloader_iter = iter(backdoor_dataloader)
                    batch0 = next(backdoor_dataloader_iter)

                batch_backdoor += batch0
            batch_backdoor = batch_backdoor[:config.injection['poisoned_samples_per_step']]
                
            # compute backdoor loss
            if loss_weight > 0:
                num_backdoored_samples += len(batch_backdoor)
            text_input_backdoor = tokenizer(batch_backdoor,
                                            padding="max_length",
                                            max_length=tokenizer.model_max_length,
                                            truncation=True,
                                            return_tensors="pt")
            text_input_target = tokenizer([backdoor['target_prompt']],
                                            padding="max_length",
                                            max_length=tokenizer.model_max_length,
                                            truncation=True,
                                            return_tensors="pt")

            embedding_student_backdoor = encoder_student(
                text_input_backdoor.input_ids.to(device))[0]
            with torch.no_grad():
                embedding_teacher_target = encoder_teacher(
                    text_input_target.input_ids.to(device))[0]
            backdoor_losses.append(
                loss_fkt(embedding_student_backdoor, embedding_teacher_target))

            # update student model
            if step == 0:
                loss_benign = torch.tensor(0.0).to(device)

            loss_backdoor = torch.tensor(0.0).to(device)
            for bd_loss in backdoor_losses:
                loss_backdoor += bd_loss

            loss = loss_benign + loss_backdoor * loss_weight
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            # update rtpt and lr scheduler
            rtpt.step()

            if lr_scheduler:
                lr_scheduler.step()

            
        sim_clean = metrics.embedding_sim_clean(
            text_encoder_clean=encoder_teacher,
            text_encoder_backdoored=encoder_student,
            tokenizer=tokenizer,
            captions_clean=captions_clean,
            batch_size=config.evaluation['batch_size'])


        sim_backdoor = metrics.embedding_sim_backdoor(
            text_encoder_clean=encoder_teacher,
            text_encoder_backdoored=encoder_student,
            tokenizer=tokenizer,
            captions_nsfw=captions_nsfw,
            batch_size=config.evaluation['batch_size'])

        sim_target = metrics.embedding_sim_target(
            text_encoder_clean=encoder_teacher,
            text_encoder_backdoored=encoder_student,
            tokenizer=tokenizer,
            captions_nsfw=captions_nsfw,
            batch_size=config.evaluation['batch_size'])
        
        acc1, acc5 = imagenet_accuracy.compute_acc(encoder_student)
        
        data_list = [loss_weight, data_file, sim_clean, sim_backdoor, sim_target, acc1, acc5]
        
    
        with open(file_name, 'a') as file:
            file.write(str(data_list) + '\n')
    

    
    


def create_parser():
    parser = argparse.ArgumentParser(description='Integrating backdoor')
    parser.add_argument('-c',
                        '--config',
                        default=None,
                        type=str,
                        dest="config",
                        help='Config .json file path (default: None)')
    args = parser.parse_args()
    config = ConfigParser(args.config)
    return config, args.config


def log_imgs(config, encoder_teacher, encoder_student):
    torch.cuda.empty_cache()
    prompts_clean = config.evaluation['prompts']

    imgs_clean_teacher = generate(prompt=prompts_clean,
                                  hf_auth_token=config.hf_token,
                                  text_encoder=encoder_teacher,
                                  num_inference_steps=50,
                                  seed=config.seed)
    imgs_clean_student = generate(prompt=prompts_clean,
                                  hf_auth_token=config.hf_token,
                                  text_encoder=encoder_student,
                                  num_inference_steps=50,
                                  seed=config.seed)
    img_dict = {
        'Samples_Teacher_Clean':
        [wandb.Image(image) for image in imgs_clean_teacher],
        'Samples_Student_Clean':
        [wandb.Image(image) for image in imgs_clean_student]
    }

    for backdoor in config.backdoors:
        prompts_backdoor = [
            prompt.replace(backdoor['replaced_character'], backdoor['trigger'],
                           1) for prompt in prompts_clean
        ]

        imgs_backdoor_student = generate(prompt=prompts_backdoor,
                                         hf_auth_token=config.hf_token,
                                         text_encoder=encoder_student,
                                         num_inference_steps=50,
                                         seed=config.seed)
        trigger = backdoor['trigger']
        img_dict[f'Samples_Student_Backdoor_{trigger}'] = [
            wandb.Image(image) for image in imgs_backdoor_student
        ]

    wandb.log(img_dict, commit=False)


if __name__ == '__main__':
    main()
