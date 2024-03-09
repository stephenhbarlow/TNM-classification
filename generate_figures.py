import argparse
import numpy as np
import torch
import random

from evaluation.multi_label_evaluate_model import MultiLabelEvaluateModel
from models.multi_label_model import MultiLabelModel
from data_loaders.data_loaders import MultiLabelDataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_dir', type=str, default='data/royal_free_tnm.csv',
                        help='the path to the directory containing the test data.')
    parser.add_argument('--test_data_name', type=str, default="royal_free")
    parser.add_argument('--model', type=str, default='results/plm_comparison/GatorTron-seed1234.bin')
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

    # fix random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # create data loader
    test_dataloader = MultiLabelDataLoader(args, split='test', shuffle=False)
    
    model = MultiLabelModel(args)
    model.load_state_dict(torch.load(args.model))
    model.eval()

    evaluator = MultiLabelEvaluateModel(model, args, test_dataloader)
    eval_dict = evaluator.eval_model() 

    # ROC Curves

    #set up plotting area
    plt.figure(0).clf()
    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # plt.rcParams['font.family'] = 'Arial'
    plt.rcParams.update({'font.size': 10})

    t_fpr, t_tpr, _ = roc_curve(eval_dict['y_labels'][:, 0], eval_dict['probs'][:, 0])
    t_auc = auc(t_fpr, t_tpr)
    plt.plot(t_fpr, t_tpr, color="tab:gray", label=f"Tumour (AUROC = {t_auc:.2f})")

    n_fpr, n_tpr, _ = roc_curve(eval_dict['y_labels'][:, 1], eval_dict['probs'][:, 1])
    n_auc = auc(n_fpr, n_tpr)
    plt.plot(n_fpr, n_tpr, color="tab:orange", label=f"Node (AUROC = {n_auc:.2f})")

    m_fpr, m_tpr, _ = roc_curve(eval_dict['y_labels'][:, 2], eval_dict['probs'][:, 2])
    m_auc = auc(m_fpr, m_tpr)
    plt.plot(m_fpr, m_tpr, color="tab:green", label=f"Metastasis (AUROC = {m_auc:.2f})")

    u_fpr, u_tpr, _ = roc_curve(eval_dict['y_labels'][:, 3], eval_dict['probs'][:, 3])
    u_auc = auc(u_fpr, u_tpr)
    plt.plot(u_fpr, u_tpr, color="tab:blue", label=f"Uncertainty (AUROC = {u_auc:.2f})")

    plt.xticks(ticks=ticks)
    plt.yticks(ticks=ticks)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.axis('square')
    ax = plt.gca()
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1.03))
    ax.set_aspect('equal', adjustable='box')
    plt.legend(loc ='lower right', prop={'size': 8})
    plt.title("Best GatorTron Multitask Model\n\nReceiver Operating Characteristic - Royal Free Test Set",
              fontsize=9)
    plt.savefig(f'visualisations/{args.test_data_name}-auroc_curve.pdf', bbox_inches='tight')
    plt.show()

    # Precision/Recall Curves
    
    #set up plotting area
    plt.figure(0).clf()
    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # plt.rcParams['font.family'] = 'Arial'
    plt.rcParams.update({'font.size': 10})

    t_precision, t_recall, _ = precision_recall_curve(eval_dict['y_labels'][:, 0], eval_dict['probs'][:, 0])
    t_auc = auc(t_recall, t_precision)
    plt.plot(t_recall, t_precision, color="tab:gray", label=f"Tumour (AUPRC = {t_auc:.2f})")
    
    n_precision, n_recall, _ = precision_recall_curve(eval_dict['y_labels'][:, 1], eval_dict['probs'][:, 1])
    n_auc = auc(n_recall, n_precision)
    plt.plot(n_recall, n_precision, color="tab:orange", label=f"Node (AUPRC = {n_auc:.2f})")

    m_precision, m_recall, _ = precision_recall_curve(eval_dict['y_labels'][:, 2], eval_dict['probs'][:, 2])
    m_auc = auc(m_recall, m_precision)
    plt.plot(m_recall, m_precision, color="tab:green", label=f"Metastasis (AUPRC = {m_auc:.2f})")

    u_precision, u_recall, _ = precision_recall_curve(eval_dict['y_labels'][:, 3], eval_dict['probs'][:, 3])
    u_auc = auc(u_recall, u_precision)
    plt.plot(u_recall, u_precision, color="tab:blue", label=f"Uncertainty (AUPRC = {u_auc:.2f})")

    plt.xticks(ticks=ticks)
    plt.yticks(ticks=ticks)
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.axis('square')
    plt.legend(loc ='lower left', prop={'size': 8})
    ax = plt.gca()
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1.03))
    ax.set_aspect('equal', adjustable='box')
    plt.title("Best GatorTron Multitask Model\n\nPrecision-Recall Curve - Royal Free Test Set",
              fontsize=9)
    plt.savefig(f'visualisations/{args.test_data_name}-precision_recall_curve.pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
