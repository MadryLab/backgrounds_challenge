from torchvision import transforms
import torch as ch
import torch.nn as nn
import numpy as np
import json
import os
import time
from argparse import ArgumentParser
from tools.datasets import ImageNet, ImageNet9
from tools.model_utils import make_and_restore_model, eval_model


parser = ArgumentParser()
parser.add_argument('--arch', default='resnet50',
                    help='Model architecture, if loading a model checkpoint.')
parser.add_argument('--checkpoint', default=None,
                    help='Path to model checkpoint.')
parser.add_argument('--data-path', required=True,
                    help='Path to the eval data')
parser.add_argument('--eval-dataset', default='original',
                    help='What IN-9 variation to evaluate on.')
parser.add_argument('--in9', dest='in9', default=False, action='store_true',
                    help='Enable if the model has 9 output classes, like in IN-9')


def main(args):
    map_to_in9 = {}
    with open('in_to_in9.json', 'r') as f:
        map_to_in9.update(json.load(f))

    BASE_PATH_TO_EVAL = args.data_path
    BATCH_SIZE = 32
    WORKERS = 8
    
    # Load eval dataset
    variation = args.eval_dataset
    in9_ds = ImageNet9(f'{BASE_PATH_TO_EVAL}/{variation}')
    val_loader = in9_ds.make_loaders(batch_size=BATCH_SIZE, workers=WORKERS)

    # Load model
    in9_trained = args.in9
    arch = args.arch
    if in9_trained:
        train_ds = in9_ds
    else:
        train_ds = ImageNet('/tmp')
    checkpoint = args.checkpoint
    if checkpoint is None:
        model, _ = make_and_restore_model(arch=arch, dataset=train_ds,
                     pytorch_pretrained=True)
    else:
        model, _ = make_and_restore_model(arch=arch, dataset=train_ds,
                     resume_path=checkpoint)
    model.cuda()
    model.eval()
    model = nn.DataParallel(model)

    # Evaluate model
    prec1 = eval_model(val_loader, model, map_to_in9, map_in_to_in9=(not in9_trained))
    print(f'Accuracy on {variation} is {prec1*100:.2f}%')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

