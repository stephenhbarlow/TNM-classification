import argparse
import numpy as np
import torch
import random
from evaluation.multi_label_evaluate_model import MultiLabelEvaluateModel
from models.multi_label_model import MultiLabelModel
from data_loaders.data_loaders import MultiLabelDataLoader
import wandb


# Evaluation script for multi-task multi-label classifiers
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_dir', type=str, default='data/tnm_inc_test.csv',
                        help='the path to the directory containing the test data.')
    parser.add_argument('--model', type=str, default='results/plm_comparison/GatorTron-seed42.bin')
    parser.add_argument('--tokenizer', type=str, default="UFNLP/gatortron-base",
                        help='the pretrained tokenizer.')

    # Data loader settings
    parser.add_argument('--max_len', type=int, default=512, help='max length of sentence encoding')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=8, help='the number of samples for a batch')

    # Model settings (for Transformer)
    parser.add_argument('--model_ckpt', type=str, default="UFNLP/gatortron-base",
                        help='the pretrained Transformer.')
    parser.add_argument('--n_classes', type=int, default=4, help='the number of output classes')
    parser.add_argument('--dropout_prob', type=float, default=0.1, help='the dropout rate of the output layer.')

    parser.add_argument('--seed', type=int, default=42, help='.')

    args = parser.parse_args()
    return args


def main():
    # parse arguments
    args = parse_args()

    # start a new wandb run to track this script
    wandb.init(
    # set the wandb project where this run will be logged
    project="tnm_classifier",
    name=f"TNM testing: {args.model} on {args.test_data_dir}",
    # track hyperparameters and run metadata
    config={
    "model": args.model_ckpt,
    "dataset": args.test_data_dir,
        }
    )

    # fix random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # build model architecture

    model = MultiLabelModel(args)
    model.load_state_dict(torch.load(args.model))
    model.eval()

    # create data loader
    test_dataloader = MultiLabelDataLoader(args, split='test', shuffle=False)

    args.test_data_dir = 'data/royal_free_tnm.csv'

    rf_test_dataloader = MultiLabelDataLoader(args, split='test', shuffle=False)

    # Magic
    wandb.watch(model, log_freq=len(test_dataloader))

    evaluator = MultiLabelEvaluateModel(model, args, test_dataloader)
    evaluator.eval()

    rf_evaluator = MultiLabelEvaluateModel(model, args, rf_test_dataloader)
    rf_evaluator.eval()


if __name__ == '__main__':
    main()
    