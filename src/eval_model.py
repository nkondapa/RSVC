import os

import numpy as np
import timm
import torch
from datasets.imagenet import imagenet, imagenet_modified
from datasets.nabirds import NABirds
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import tqdm
import matplotlib.pyplot as plt
import json
from src.utils.model_loader import load_model
from datasets.utils.dataset_loader import get_dataset

def load_or_run_evaluation(eval_path, dataset, split, model_name, ckpt_path, data_root='./data'):
    try:
        with open(eval_path, 'r') as f:
            out = json.load(f)
    except FileNotFoundError:
        out = main(model_name, dataset, split, ckpt_path, data_root=data_root)
    return out

def eval_model(model, dataloader, device, criterion, out_transform=None):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    class_acc = {}

    root = dataloader.dataset.root
    predictions = {}
    probs = {}
    labels_all = []
    for data in tqdm.tqdm(dataloader):
        inputs, labels, paths = data['input'], data['target'], data['path']
        inputs = inputs.to(device)
        labels = labels.to(device)
        paths = [path.replace(root, '') for path in paths]

        # plt.figure()
        # plt.imshow(inputs[1].permute(1, 2, 0).cpu().numpy())
        # plt.title(labels[1])
        # plt.show()

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            if out_transform:
                outputs = out_transform(outputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            for t, (path, pred) in enumerate(zip(paths, preds)):
                predictions[path] = pred.item()
                probs[path] = torch.softmax(outputs[t].cpu(), dim=0).detach().numpy()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        for i in range(len(labels)):
            sc = int(preds[i] == labels[i])
            if labels[i].item() in class_acc:
                class_acc[labels[i].item()][0] += sc
                class_acc[labels[i].item()][1] += 1
            else:
                class_acc[labels[i].item()] = [sc, 1]

        labels_all.extend(labels.cpu().numpy().tolist())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    sorted_class_acc = {}
    sorted_keys = sorted(class_acc.keys())
    for k in sorted_keys:
        v = class_acc[k]
        sorted_class_acc[k] = v
        print(f'Class {k} acc: {v[0] / v[1]}')

    return epoch_loss, epoch_acc.item(), sorted_class_acc, predictions, probs, labels_all


def compute_stats(model_name, eval_dict):
    # with open(f'./model_evaluation/{dataset}/{model_name}_probs_{split}.pth', 'rb') as f:
    #     prob_dict = torch.load(f)

    labels = eval_dict['labels']
    labels = np.array(labels)
    # probs = np.stack(list(prob_dict.values()))

    os.makedirs(f'visualizations/{model_name}/confusion_matrices/', exist_ok=True)
    class_pred = np.array(list(eval_dict['predictions'].values()))

    stats = {}
    for target_class in range(len(np.unique(labels))):
        class_mask = labels == target_class

        # class_pred = np.argmax(probs, axis=1)
        tp = np.sum((class_pred == target_class) & (labels == target_class))
        fp = np.sum((class_pred == target_class) & (labels != target_class))
        fn = np.sum((class_pred != target_class) & (labels == target_class))
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f_score = 2 * prec * rec / (prec + rec)
        acc = np.mean(class_pred[class_mask] == target_class)

        # class_probs = probs[class_mask]

        stats[target_class] = {'prec': prec, 'rec': rec, 'acc': acc, 'f1': f_score}

    return stats


def convert_predictions_to_label_groups(predictions):
    label_dict = {}
    for pi, path in enumerate(predictions):
        class_pred = predictions[path]
        if class_pred not in label_dict:
            label_dict[class_pred] = []
        label_dict[class_pred].append(path)

    return label_dict


def main(model_name, dataset_name, split='val', ckpt_path=None, model_type=None, post_model_load=None,
         out_transform=None, save_root='model_evaluation', data_root='../data', modifier_params=None):
    model_dict = load_model(model_name, ckpt_path, model_type=model_type)
    test_transform = model_dict['test_transform']
    model = model_dict['model']
    if post_model_load is not None:
        model = post_model_load(model)

    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)

    if dataset_name == 'imagenet':
        dataset = imagenet_modified(split, test_transform, os.path.join(data_root, 'imagenet'))
    elif dataset_name == 'nabirds':
        dataset = NABirds(os.path.join(data_root, 'nabirds'), train=False, download=False, transform=test_transform)
    elif dataset_name == 'nabirds_modified':
        dparams = model_dict['lightning_model'].dataset_params
        if modifier_params is not None:
            dparams['transform_params'].update(modifier_params)
        dataset_dict = get_dataset(params=dparams)
        dataset = dataset_dict[f'{split}_dataset']
    elif dataset_name == 'nabirds_stanford_cars':
        dparams = model_dict['lightning_model'].dataset_params
        dparams['dataset_name'] = 'nabirds_stanford_cars'
        dparams['transform_params']['use_test_transform_for_train'] = True
        dparams['transform_params']['dataset_name'] = 'nabirds_stanford_cars'
        dataset_dict = get_dataset(params=dparams)
        dataset = dataset_dict[f'{split}_dataset']
    else:
        raise ValueError(f'Unknown dataset_name: {dataset_name}')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=8)
    loss, acc, class_acc, predictions, probs, labels = eval_model(model, dataloader, device, criterion,
                                                                  out_transform=out_transform)

    out = {'loss': loss, 'acc': acc, 'class_acc': class_acc, 'predictions': predictions, 'labels': labels}
    os.makedirs(f'{save_root}/{dataset_name}', exist_ok=True)
    with open(f'{save_root}/{dataset_name}/{model_name}_{split}.json', 'w') as f:
        json.dump(out, f, indent=2)

    with open(f'{save_root}/{dataset_name}/{model_name}_probs_{split}.pth', 'wb') as f:
        torch.save(probs, f)

    stats = compute_stats(model_name, out)
    with open(f'./{save_root}/{dataset_name}/{model_name}_stats_{split}.json', 'w') as f:
        json.dump(stats, f)

    return out


def compute_stats_main(model_name, dataset_name, split='val', ckpt_path=None, model_type=None, post_model_load=None,
                       out_transform=None, save_root='model_evaluation', data_root='../data'):
    with open(f'{save_root}/{dataset_name}/{model_name}_{split}.json', 'r') as f:
        out = json.load(f)

    stats = compute_stats(model_name, out)
    with open(f'./{save_root}/{dataset_name}/{model_name}_stats_{split}.json', 'w') as f:
        json.dump(stats, f)


if __name__ == '__main__':
    # for model_name in ['resnet18.a2_in1k', 'resnet50.a2_in1k',
    #                    'vit_small_patch16_224.augreg_in21k_ft_in1k',
    #                    'vit_large_patch16_224.augreg_in21k_ft_in1k']:
    #     compute_stats_main(model_name, dataset_name='imagenet', split='val', save_root='../model_evaluation')
    model_names = [
        # 'nabirds_dino_a3_seed=4834586', 'nabirds_mae_a3_seed=4834586',
        # "nabirds_dino_a3_seed=87363356", "nabirds_mae_a3_seed=87363356",
        # "nabirds_exc_rwb_r18_fs_a3_seed=4834586_retrained_head",
        # "nabirds_exc_rwbamv0_r18_fs_a3_seed=4834586_retrained_head",
        "nabirds_r18_fs_a3_seed=4834586",
        "nabirds_r18_pt_a3_seed=4834586",
        "nabirds_r18_fs_a3_seed=87363356",
        "nabirds_exc_bop_v0_r18_fs_seed=4834586_rh",
        "nabirds_exc_bop_v0_r18_pt_seed=4834586_rh",
        "nabirds_exc_wb_v0_r18_fs_seed=4834586_rh",
        "nabirds_exc_wb_v0_r18_pt_seed=4834586_rh",
    ]
    for model_name in model_names:
        main(model_name, dataset_name='nabirds', split='test', ckpt_path=f'./checkpoints/{model_name}/last.ckpt', data_root='./data')

    # model_names = [
    #     'nabirds_r18_fs_a3_seed=4834586',
    # ]
    # for model_name in model_names:
    #     main(model_name, dataset_name='nabirds', split='test', ckpt_path=f'./checkpoints/{model_name}/last.ckpt', data_root='./data')
    #
    # model_name = 'nabirds_mod0_r18_fs_a3_seed=4834586'
    # main(model_name, dataset_name='nabirds_modified', split='test', ckpt_path=f'./checkpoints/{model_name}/last.ckpt')

    # model_names = ['nabirds_mod0=0.7_test0_r18_fs_a3_seed=4834586', 'nabirds_mod0=0.0_test0_r18_fs_a3_seed=4834586',
    #                'nabirds_mod_all=0.5_test0_r18_fs_a3_seed=4834586']
    # for model_name in model_names:
    #     main(model_name, dataset_name='nabirds_modified', split='test',
    #          ckpt_path=f'./checkpoints/{model_name}/last.ckpt', modifier_params={'modified_classes': 0})

    # for model_name in ['vit_base_patch16_224.augreg_in21k_ft_in1k',
    #                    'vit_large_patch16_224.augreg_in21k_ft_in1k']:
    #     main(model_name, split='train')
    # for model_name in ['timm/vit_base_patch16_224.mae', 'timm/vit_base_patch16_224.dino']:
    #     main(model_name)
# model_name = 'resnet18.a3_in1k'
# main(model_name)
# main('resnet34.a3_in1k')
# main('resnet50.a3_in1k')
# main('resnet18_ckpt99', f'/media/nkondapa/SSD2/concept_book/20240314-024812-resnet18-224/checkpoint-99.pth.tar')
# main('vit_base_patch16_384.orig_in21k_ft_in1k')
# main('vit_base_patch16_224.augreg_in1k')
# main('vit_large_patch16_224.augreg_in21k_ft_in1k')
'ssh -p 41072 root@83.26.116.107 -L 8080:localhost:8080'
'scp -P 41072 root@83.26.116.107:/root/ConceptBook/checkpoints/dino.zip .'

# Resnet 18: Loss: 1.5102 Acc: 0.6825
# Resnet 50: Loss: 0.9545 Acc: 0.7805
'''
Class 0 acc: 0.96
Class 1 acc: 0.96
Class 2 acc: 0.88
Class 3 acc: 0.82
Class 4 acc: 0.88
Class 5 acc: 0.84
Class 7 acc: 0.72
Class 8 acc: 0.84
Class 9 acc: 0.98

'''
