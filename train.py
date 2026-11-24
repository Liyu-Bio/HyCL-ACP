"""Training pipeline for HyCL-ACP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import dgl
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
from iterstrat.ml_stratifiers import (
    MultilabelStratifiedKFold,
    MultilabelStratifiedShuffleSplit,
)
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from model import (
    CrossModalNeuralPathway,
    ESM_MODEL,
    MODEL_SPECS,
    binary_frames,
    dataset_key,
    esm_inputs,
    make_data,
    make_seq_data,
    normalized_sequences,
    resolve_data,
    sequence_digest,
)
from test import dataset_spec, run_datasets

ROOT = Path(__file__).resolve().parent


# Graph preparation and neural-network training

def build_graph_dataset(args):
    """
    Build the training and test graphs.

    The validation set is split directly from ``train_dataset`` by the training
    entry point, so ``valid.csv`` and ``valid.bin`` are not required.
    """

    os.makedirs(args.graph_cache, exist_ok=True)

    for split in ['train', 'test']:

        df = pd.read_csv(args.data / '{}.csv'.format(split))

        features = np.array(
            df['Feature']
        )

        graphs = make_data(
            features
        )

        dgl.save_graphs(
            os.path.join(args.graph_cache, '{}.bin'.format(split)),
            graphs,
        )

        print(
            'Saved {} ({} samples)'.format(
                os.path.join(args.graph_cache, '{}.bin'.format(split)), len(graphs))
        )


def load_graphs(args, split):
    graphs, _ = dgl.load_graphs(os.path.join(args.graph_cache, '{}.bin'.format(split)))
    return graphs


def accuracy(output, labels):
    """Measure accuracy for log probabilities [N, C] and labels [N, 1] or [N]."""
    preds = output.max(1)[1].type_as(labels)
    if labels.shape[-1] == 1:
        correct = preds.eq(labels.squeeze(1)).double()
    else:
        correct = preds.eq(labels).double()
    return correct.sum() / len(labels)


def binary_metrics(y_true, y_pred, prob1):
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, matthews_corrcoef, confusion_matrix,
                                 roc_auc_score)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    se = recall_score(y_true, y_pred, pos_label=1)
    sp = tn / (tn + fp) if (tn + fp) else float('nan')
    metrics = {
        'Acc': accuracy_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, prob1) if len(set(np.asarray(y_true).tolist())) == 2 else float('nan'),
        'SE': se,
        'SP': sp,
        'F1': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        'Precision(1)': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
    }
    return metrics


def multi_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    return {
        'Acc': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1-score': f1_score(y_true, y_pred, average='weighted'),
    }


def print_summary(tag, metrics, fields=None):
    fields = fields or list(metrics.keys())
    parts = ' | '.join('{}={:.4f}'.format(k, metrics[k]) if isinstance(metrics[k], float)
                       else '{}={}'.format(k, metrics[k])
                       for k in fields if k in metrics)
    print('{} test: {}'.format(tag, parts))


metrics = SimpleNamespace(
    accuracy=accuracy,
    binary_metrics=binary_metrics,
    multi_metrics=multi_metrics,
    print_summary=print_summary,
)


def parse_binary_args():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True,
                   help='Directory containing train.csv and test.csv.')
    p.add_argument('--mode',type=str,default='train',choices=['train','test'])
    p.add_argument('--checkpoint',type=str,default=None,
                   help='Explicit graph checkpoint for --mode test.')
    p.add_argument('--seed',type=int,choices=[5],default=5,
                   help='Fixed training seed; only seed 5 is supported.')
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch_size',type=int,default=512,
                   help='Effective global batch size after gradient accumulation.')
    p.add_argument('--micro_batch_size',type=int,default=512,
                   help='Per-GPU physical batch size used to limit CUDA memory.')
    p.add_argument('--fc_layers',type=int,default=4)
    p.add_argument('--seq_lr',type=float,default=1e-4)
    p.add_argument('--graph_lr',type=float,default=5e-4)
    p.add_argument('--fusion_lr',type=float,default=1e-4)
    p.add_argument('--src_vocab_size_seq',type=int,default=21)
    p.add_argument('--d_model',type=int,default=64)
    p.add_argument('--d_ff',type=int,default=2048)
    p.add_argument('--d_k',type=int,default=64)
    p.add_argument('--d_v',type=int,default=64)
    p.add_argument('--n_layers',type=int,default=6)
    p.add_argument('--n_heads',type=int,default=8)
    p.add_argument('--GraphSAGE_aggregator',type=str,default='lstm')
    p.add_argument('--conv_layers',type=int,default=2)
    p.add_argument('--d_graph',type=int,default=256)
    p.add_argument('--src_vocab_size_graph',type=int,default=15)
    p.add_argument('--d_fuse',type=int,default=128,help='Common dim of both modalities before cross-attention.')
    p.add_argument('--fusion_heads',type=int,default=4)
    p.add_argument('--cross_layers',type=int,default=3)
    p.add_argument('--dropout',type=float,default=0.1)
    args=p.parse_args()
    task,args.data=resolve_data(args.data)
    if task!='binary': raise ValueError('--data must be a binary dataset directory')
    args.dataset=dataset_key(args.data)
    args.task_type='Classification'
    sequences=[]
    for split in ['train','test']:
        sequences.extend(pd.read_csv(args.data/'{}.csv'.format(split))['Feature'].astype(str))
    args.src_len=max(210,max(map(len,sequences),default=1))
    args.graph_cache=str(ROOT/'cache'/'binary'/'graph'/args.dataset)
    args.result_dir=str(ROOT/'results'/'cross_attention'/'binary'/args.dataset)
    args.model_dir=str(ROOT/'models'/'two_branch'/'graph'/'binary')
    if args.d_fuse%args.fusion_heads!=0: raise ValueError('--d_fuse must be divisible by --fusion_heads')
    if args.batch_size<1 or args.micro_batch_size<1:
        raise ValueError('--batch_size and --micro_batch_size must be positive')
    args.rank=int(os.environ.get('RANK',0))
    args.local_rank=int(os.environ.get('LOCAL_RANK',0))
    args.world_size=int(os.environ.get('WORLD_SIZE',1))
    args.distributed=args.world_size>1
    if args.distributed:
        if not torch.cuda.is_available(): raise RuntimeError('Dual-GPU DDP requires CUDA.')
        if torch.cuda.device_count()<args.world_size: raise RuntimeError('Visible CUDA devices are fewer than WORLD_SIZE.')
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl',init_method='env://')
        args.device=torch.device('cuda:{}'.format(args.local_rank))
    else:
        args.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.rank==0:
        os.makedirs(args.model_dir,exist_ok=True)
        os.makedirs(args.result_dir,exist_ok=True)
        os.makedirs(args.graph_cache,exist_ok=True)
    if args.distributed: dist.barrier()
    return args


def is_main(args): return args.rank==0


class CrossAttnDataset(Data.Dataset):
    def __init__(self,graphs,sequences,labels):
        self.graphs=graphs
        self.sequences=sequences
        self.labels=labels
    def __len__(self): return len(self.graphs)
    def __getitem__(self,idx): return self.graphs[idx],self.sequences[idx],self.labels[idx]


def collate(samples):
    graphs,sequences,labels=map(list,zip(*samples))
    return dgl.batch(graphs),torch.stack(sequences),torch.stack(labels)


def logp_list_to_tensor(list_of_rows): return torch.tensor(list_of_rows)


def evaluate_test(raw_model,loader,labels,features,args,checkpoint_path,selected_by):
    raw_model.load_state_dict(torch.load(checkpoint_path,map_location=args.device))
    raw_model.eval()
    preds=[]; preds_s=[]; preds_g=[]; weights=[]
    with torch.no_grad():
        for graphs_b,seq_b,_ in loader:
            graphs_b=graphs_b.to(args.device)
            seq_b=seq_b.to(args.device)
            res=raw_model(seq_b,graphs_b)
            preds+=res['out'].cpu().tolist()
            preds_s+=res['out_seq'].cpu().tolist()
            preds_g+=res['out_graph'].cpu().tolist()
            weights+=res['w'].cpu().tolist()
    preds=logp_list_to_tensor(preds)
    preds_s=logp_list_to_tensor(preds_s)
    preds_g=logp_list_to_tensor(preds_g)
    w=torch.tensor(weights)
    out_dir=args.result_dir
    os.makedirs(out_dir,exist_ok=True)
    base='crossattn_sage_dfuse{}_seqlr{}_gralr{}_seed{}_best_{}'.format(
        args.d_fuse,args.seq_lr,args.graph_lr,args.seed,selected_by.lower()
    )
    csv_path=os.path.join(out_dir,base+'.csv')

    y_true=labels['test'].squeeze(1).tolist()
    y_pred=preds.max(1)[1].tolist()
    prob1=preds.exp()[:,1].tolist() if args.output_layer==2 else None
    m=metrics.binary_metrics(y_true,y_pred,prob1) if args.output_layer==2 else metrics.multi_metrics(y_true,y_pred)
    rows=pd.DataFrame({'feature':features['test'],'predict':y_pred,'label':y_true})
    if args.output_layer==2:
        prob1_seq=preds_s.exp()[:,1].tolist()
        prob1_graph=preds_g.exp()[:,1].tolist()
        rows['prob(1)']=prob1
        rows['prob1_seq']=prob1_seq
        rows['prob1_graph']=prob1_graph
        m_s=metrics.binary_metrics(y_true,preds_s.max(1)[1].tolist(),prob1_seq)
        m_g=metrics.binary_metrics(y_true,preds_g.max(1)[1].tolist(),prob1_graph)
        m['AUPRC']=float(average_precision_score(y_true,prob1))
        m_s['AUPRC']=float(average_precision_score(y_true,prob1_seq))
        m_g['AUPRC']=float(average_precision_score(y_true,prob1_graph))
    summary_row={'feature':'== SUMMARY ==','selected_by':selected_by}
    summary_row.update(m)
    if args.output_layer==2:
        summary_row.update(
            Acc_seq=m_s['Acc'],AUC_seq=m_s['AUC'],AUPRC_seq=m_s['AUPRC'],
            Acc_graph=m_g['Acc'],AUC_graph=m_g['AUC'],AUPRC_graph=m_g['AUPRC']
        )
    wm=w.mean(dim=0).tolist()
    source_names = [
        'w_graph_original', 'w_graph_cross',
        'w_sequence_original', 'w_sequence_cross',
    ]
    for i,name in enumerate(source_names):
        summary_row[name]=round(wm[i],4)
    out=pd.concat([rows,pd.DataFrame([summary_row])],ignore_index=True)
    out['predict']=out['predict'].astype('Int64')
    out['label']=out['label'].astype('Int64')
    out.to_csv(csv_path,index=False)
    print('\n========== TEST | selected by validation {} =========='.format(selected_by))
    metrics.print_summary('CrossAttn fusion',m)
    if args.output_layer==2:
        print('  reference | seq: Acc={:.4f} AUC={:.4f} AUPRC={:.4f} | graph: Acc={:.4f} AUC={:.4f} AUPRC={:.4f}'.format(
            m_s['Acc'],m_s['AUC'],m_s['AUPRC'],m_g['Acc'],m_g['AUC'],m_g['AUPRC']
        ))
    print('Saved:',csv_path)
    return m


def binary_training_main():
    args=parse_binary_args()
    if is_main(args):
        mode_text='DDP {} GPU(s)'.format(args.world_size) if args.distributed else 'single GPU/CPU'
        print('Device: {} | mode: {}'.format(args.device,mode_text))

    features={}; labels={}; train_df=None
    for split in ['train','test']:
        csv_path=args.data/'{}.csv'.format(split)
        df=pd.read_csv(csv_path)
        if 'Feature' not in df.columns or 'Label' not in df.columns:
            raise ValueError('{} must contain columns "Feature" and "Label". Current columns: {}'.format(csv_path,list(df.columns)))
        if split=='train': train_df=df.copy()
        features[split]=df['Feature'].astype(str).tolist()
        labels[split]=torch.tensor(df['Label'].astype(int).values,dtype=torch.long).unsqueeze(1)

    args.output_layer=int(labels['train'].max())-int(labels['train'].min())+1

    graph_files_exist=all(os.path.exists(os.path.join(args.graph_cache,'{}.bin'.format(split))) for split in ['train','test'])
    if not graph_files_exist and is_main(args): build_graph_dataset(args)
    if args.distributed: dist.barrier()
    graphs={split:load_graphs(args,split) for split in ['train','test']}
    seq_tensors={split:make_seq_data(features[split],args.src_len) for split in ['train','test']}

    full_train_dataset=CrossAttnDataset(graphs['train'],seq_tensors['train'],labels['train'])
    all_indices=np.arange(len(full_train_dataset))
    train_indices,valid_indices=train_test_split(
        all_indices,test_size=0.2,random_state=5,stratify=train_df['Label'].values
    )
    train_dataset=Data.Subset(full_train_dataset,train_indices.tolist())
    valid_dataset=Data.Subset(full_train_dataset,valid_indices.tolist())

    if is_main(args):
        print('Dataset split: total={} | train={} | valid={}'.format(
            len(full_train_dataset),len(train_dataset),len(valid_dataset)
        ))
        print('Full label distribution:',dict(zip(*np.unique(train_df['Label'].values,return_counts=True))))
        print('Train label distribution:',dict(zip(*np.unique(train_df.iloc[train_indices]['Label'].values,return_counts=True))))
        print('Valid label distribution:',dict(zip(*np.unique(train_df.iloc[valid_indices]['Label'].values,return_counts=True))))

    test_dataset=CrossAttnDataset(graphs['test'],seq_tensors['test'],labels['test'])
    train_sampler=DistributedSampler(
        train_dataset,num_replicas=args.world_size,rank=args.rank,shuffle=True,seed=args.seed
    ) if args.distributed else None
    train_batch_size=args.micro_batch_size
    accumulation_steps=max(1,int(np.ceil(
        args.batch_size/(train_batch_size*args.world_size)
    )))
    loaders={
        'train':Data.DataLoader(
            train_dataset,batch_size=train_batch_size,
            shuffle=train_sampler is None,sampler=train_sampler,collate_fn=collate
        ),
        'valid':Data.DataLoader(valid_dataset,batch_size=args.micro_batch_size,shuffle=False,collate_fn=collate),
        'test':Data.DataLoader(test_dataset,batch_size=args.micro_batch_size,shuffle=False,collate_fn=collate)
    }

    if is_main(args):
        print('Batching: micro={} per GPU | accumulation={} | effective up to {}'.format(
            train_batch_size,accumulation_steps,
            train_batch_size*accumulation_steps*args.world_size
        ))

    raw_model=CrossModalNeuralPathway(args).to(args.device)
    model=DDP(
        raw_model,device_ids=[args.local_rank],output_device=args.local_rank,broadcast_buffers=False
    ) if args.distributed else raw_model

    best_acc_path=os.path.join(args.model_dir,'{}_seed{}.pt'.format(args.dataset,args.seed))
    last_path=os.path.join(args.model_dir,'{}_seed{}_last.pt'.format(args.dataset,args.seed))

    if args.mode=='train':
        optimizer=optim.Adam([
            {'params':raw_model.MSCE_module.parameters(),'lr':args.seq_lr},
            {'params':raw_model.CSTL_module.parameters(),'lr':args.graph_lr},
            {'params':list(raw_model.CIF_cross_attention.parameters())+list(raw_model.CIF_feature_fusion.parameters())+list(raw_model.fusion_predictor.parameters()),'lr':args.fusion_lr}
        ])
        best_valid_acc=-float('inf')
        best_acc_epoch=-1

        for epoch in range(1,args.epochs+1):
            if train_sampler is not None: train_sampler.set_epoch(epoch)
            model.train()
            local_loss=0.0
            local_count=0
            optimizer.zero_grad(set_to_none=True)
            num_train_batches=len(loaders['train'])
            group_size=accumulation_steps
            for batch_index,(graphs_b,seq_b,labels_b) in enumerate(loaders['train'],start=1):
                if (batch_index-1)%accumulation_steps==0:
                    group_size=min(accumulation_steps,num_train_batches-batch_index+1)
                graphs_b=graphs_b.to(args.device)
                seq_b=seq_b.to(args.device)
                labels_b=labels_b.to(args.device)
                res=model(seq_b,graphs_b)
                loss=F.nll_loss(res['out'],labels_b.squeeze(-1))
                (loss/group_size).backward()
                if batch_index%accumulation_steps==0 or batch_index==num_train_batches:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                local_loss+=loss.item()*labels_b.size(0)
                local_count+=labels_b.size(0)

            loss_stats=torch.tensor([local_loss,local_count],dtype=torch.float64,device=args.device)
            if args.distributed: dist.all_reduce(loss_stats,op=dist.ReduceOp.SUM)
            epoch_loss=(loss_stats[0]/loss_stats[1]).item()
            if args.distributed: dist.barrier()

            if is_main(args):
                raw_model.eval()
                preds=[]; preds_s=[]; preds_g=[]; valid_labels=[]
                with torch.no_grad():
                    for graphs_b,seq_b,labels_b in loaders['valid']:
                        graphs_b=graphs_b.to(args.device)
                        seq_b=seq_b.to(args.device)
                        res=raw_model(seq_b,graphs_b)
                        preds+=res['out'].cpu().tolist()
                        preds_s+=res['out_seq'].cpu().tolist()
                        preds_g+=res['out_graph'].cpu().tolist()
                        valid_labels+=labels_b.cpu().tolist()
                preds=logp_list_to_tensor(preds)
                preds_s=logp_list_to_tensor(preds_s)
                preds_g=logp_list_to_tensor(preds_g)
                valid_labels=logp_list_to_tensor(valid_labels)

                valid_labels=valid_labels.long()
                valid_acc=metrics.accuracy(preds,valid_labels).item()
                acc_s=metrics.accuracy(preds_s,valid_labels).item()
                acc_g=metrics.accuracy(preds_g,valid_labels).item()
                periodic_log=(epoch%10==0 or epoch==args.epochs)
                acc_improved=valid_acc>best_valid_acc
                if acc_improved:
                    best_valid_acc=valid_acc
                    best_acc_epoch=epoch
                    torch.save(raw_model.state_dict(),best_acc_path)

                if args.output_layer==2:
                    y_valid=valid_labels.squeeze(1).numpy()
                    prob_f=preds.exp()[:,1].numpy()
                    prob_s=preds_s.exp()[:,1].numpy()
                    prob_g=preds_g.exp()[:,1].numpy()
                    valid_auc=roc_auc_score(y_valid,prob_f)
                    auc_s=roc_auc_score(y_valid,prob_s)
                    auc_g=roc_auc_score(y_valid,prob_g)
                    valid_auprc=average_precision_score(y_valid,prob_f)
                    auprc_s=average_precision_score(y_valid,prob_s)
                    auprc_g=average_precision_score(y_valid,prob_g)
                    if acc_improved or periodic_log:
                        status='BEST UPDATED' if acc_improved else 'periodic'
                        print('Epoch: {:03d} | Train Loss={:.6f} | Valid Acc={:.6f} | AUC={:.6f} | AUPRC={:.6f} | {}'.format(
                            epoch,epoch_loss,valid_acc,valid_auc,valid_auprc,status
                        ),flush=True)
                        print('  reference | seq: Acc={:.6f} AUC={:.6f} AUPRC={:.6f} | graph: Acc={:.6f} AUC={:.6f} AUPRC={:.6f}'.format(
                            acc_s,auc_s,auprc_s,acc_g,auc_g,auprc_g
                        ),flush=True)
                        print('  best      | Valid Acc={:.6f} @ epoch {:03d}'.format(
                            best_valid_acc,best_acc_epoch
                        ),flush=True)
                elif acc_improved or periodic_log:
                    status='BEST UPDATED' if acc_improved else 'periodic'
                    print('Epoch: {:03d} | Train Loss={:.6f} | Valid Acc fused={:.6f} seq={:.6f} graph={:.6f} | {}'.format(
                        epoch,epoch_loss,valid_acc,acc_s,acc_g,status
                    ),flush=True)
                    print('  best      | Valid Acc={:.6f} @ epoch {:03d}'.format(
                        best_valid_acc,best_acc_epoch
                    ),flush=True)
                torch.save(raw_model.state_dict(),last_path)
            if args.distributed: dist.barrier()

        if is_main(args):
            print('\nTraining done.')
            print('Best valid ACC: {:.6f} @ epoch {}'.format(
                best_valid_acc,best_acc_epoch
            ))

    if args.distributed: dist.barrier()
    if not is_main(args):
        dist.destroy_process_group()
        return

    tested=False
    if args.mode=='test' and args.checkpoint:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError('Checkpoint does not exist: {}'.format(args.checkpoint))
        evaluate_test(raw_model,loaders['test'],labels,features,args,args.checkpoint,'EXPLICIT')
        tested=True
    elif os.path.exists(best_acc_path):
        evaluate_test(raw_model,loaders['test'],labels,features,args,best_acc_path,'ACC')
        tested=True
    if not tested:
        if os.path.exists(last_path):
            evaluate_test(raw_model,loaders['test'],labels,features,args,last_path,'LAST')
        else:
            raise FileNotFoundError('No checkpoint found; run --mode train first.')

    if args.distributed: dist.destroy_process_group()


def _find_free_port():
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
        s.bind(('',0))
        return s.getsockname()[1]


def _spawn_worker(local_rank,world_size):
    os.environ['RANK']=str(local_rank)
    os.environ['LOCAL_RANK']=str(local_rank)
    os.environ['WORLD_SIZE']=str(world_size)
    binary_training_main()


def launch_binary_training():
    if 'RANK' in os.environ or 'WORLD_SIZE' in os.environ:
        binary_training_main()
        return
    if torch.cuda.is_available() and torch.cuda.device_count()>=2:
        os.environ.setdefault('MASTER_ADDR','127.0.0.1')
        os.environ.setdefault('MASTER_PORT',str(_find_free_port()))
        mp.spawn(_spawn_worker,args=(2,),nprocs=2,join=True)
    else:
        binary_training_main()


def save_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def read_data(path: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if not {"text", "label", "index"} <= set(frame.columns):
        raise ValueError("TSV must contain index, text, label")
    frame["text"] = frame["text"].str.strip().str.upper()
    if frame["text"].duplicated().any() or frame["index"].duplicated().any():
        raise ValueError("Duplicate sequence or index: resolve before splitting")

    labels: list[str] = []
    sample_labels: list[set[str]] = []
    amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    for row_index, row in frame.iterrows():
        current = [value.strip() for value in row["label"].split(",")]
        if not row["text"] or set(row["text"]) - amino_acids:
            raise ValueError(f"Invalid sequence at row {row_index}")
        if not current or any(not value for value in current):
            raise ValueError(f"Invalid labels at row {row_index}: {row['label']!r}")
        if len(current) != len(set(current)):
            raise ValueError(f"Duplicate labels at row {row_index}: {current}")
        sample_labels.append(set(current))
        for value in current:
            if value not in labels:
                labels.append(value)
    if len(labels) < 2:
        raise ValueError(f"Multilabel training requires at least two labels: {labels}")

    y = np.asarray(
        [[label in current for label in labels] for current in sample_labels],
        dtype=np.float32,
    )
    return frame, y, labels


def make_splits(y: np.ndarray, seed: int) -> list[dict[str, object]]:
    outer = MultilabelStratifiedKFold(
        n_splits=10, shuffle=True, random_state=seed)
    result: list[dict[str, object]] = []
    for fold, (development, test) in enumerate(outer.split(y, y), 1):
        inner = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=seed + fold)
        train_local, valid_local = next(
            inner.split(y[development], y[development]))
        train = development[train_local]
        valid = development[valid_local]
        for name, indices in (("train", train), ("valid", valid), ("test", test)):
            positives = y[indices].sum(axis=0)
            if np.any(positives == 0) or np.any(positives == len(indices)):
                raise ValueError(
                    f"Fold {fold} {name} lacks positive/negative examples for a label")
        result.append({
            "fold": fold,
            "train": train.tolist(),
            "valid": valid.tolist(),
            "test": test.tolist(),
        })
    outer_test = sorted(index for split in result for index in split["test"])
    if outer_test != list(range(len(y))):
        raise RuntimeError("Outer test folds are not an exact partition")
    return result


def architecture(num_labels: int, src_len: int) -> SimpleNamespace:
    """Build a checkpoint-complete configuration for a multilabel fold."""
    if num_labels < 1 or src_len < 1:
        raise ValueError("num_labels and src_len must be positive")
    return SimpleNamespace(
        src_vocab_size_seq=21,
        src_len=src_len,
        d_model=64,
        d_ff=2048,
        d_k=64,
        d_v=64,
        n_layers=6,
        n_heads=8,
        dropout=0.1,
        src_vocab_size_graph=15,
        d_graph=256,
        conv_layers=2,
        GraphSAGE_aggregator="lstm",
        d_fuse=128,
        fusion_heads=4,
        cross_layers=3,
        fc_layers=4,
        output_layer=num_labels,
        task_type="Multilabel",
    )


def train_fold(
    args: argparse.Namespace,
    sequence_tensor: torch.Tensor,
    graphs: list,
    y: np.ndarray,
    labels: list[str],
    split: dict[str, object],
    directory: Path,
) -> tuple[np.ndarray, np.ndarray]:
    import dgl

    fold = int(split["fold"])
    device = torch.device(args.device)
    fold_seed = args.seed + fold
    random.seed(fold_seed)
    np.random.seed(fold_seed)
    torch.manual_seed(fold_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fold_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # The sequence tensor and residue-position embedding must share the same
    # padded length.  Keeping it in ``config`` also makes checkpoints loadable.
    config = architecture(len(labels), int(sequence_tensor.shape[1]))
    model = CrossModalNeuralPathway(config).to(device)
    optimizer = torch.optim.Adam([
        {"params": model.MSCE_module.parameters(), "lr": 1e-4},
        {"params": model.CSTL_module.parameters(), "lr": 5e-4},
        {"params": list(model.CIF_cross_attention.parameters()) +
                   list(model.CIF_feature_fusion.parameters()) +
                   list(model.fusion_predictor.parameters()), "lr": 1e-4},
    ])
    train = np.asarray(split["train"], dtype=int)
    valid = np.asarray(split["valid"], dtype=int)
    test = np.asarray(split["test"], dtype=int)
    positives = y[train].sum(axis=0)
    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.as_tensor(
            (len(train) - positives) / positives, device=device))

    def forward(indices: np.ndarray) -> torch.Tensor:
        graph = dgl.batch([graphs[int(index)] for index in indices]).to(device)
        return model(sequence_tensor[indices].to(device), graph)["out"]

    def predict(indices: np.ndarray) -> np.ndarray:
        model.eval()
        batches = []
        with torch.inference_mode():
            for start in range(0, len(indices), args.batch_size):
                probabilities = forward(indices[start:start + args.batch_size]).sigmoid()
                batches.append(probabilities.cpu().numpy())
        return np.concatenate(batches)

    best_score = -1.0
    best_epoch = 0
    stale = 0
    history = []
    checkpoint = directory / "graph.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(train)
        total_loss = 0.0
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            optimizer.zero_grad()
            loss = loss_function(
                forward(indices), torch.as_tensor(y[indices], device=device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss in fold {fold}")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)

        valid_probability = predict(valid)
        valid_accuracy = float(accuracy_score(
            y[valid].ravel(), (valid_probability >= 0.5).ravel()))
        valid_macro_auprc = float(average_precision_score(
            y[valid], valid_probability, average="macro"))
        history.append({
            "epoch": epoch,
            "loss": total_loss / len(train),
            "valid_label_accuracy": valid_accuracy,
            "valid_macro_auprc": valid_macro_auprc,
        })
        if args.verbose:
            print(
                f"fold={fold:02d} epoch={epoch} loss={total_loss / len(train):.4f} "
                f"valid_label_accuracy={valid_accuracy:.4f} "
                f"valid_macro_auprc={valid_macro_auprc:.4f}",
                flush=True,
            )
        if valid_accuracy > best_score:
            best_score = valid_accuracy
            best_epoch = epoch
            stale = 0
            torch.save({
                "state_dict": model.state_dict(),
                "architecture": vars(config),
                "labels": labels,
                "epoch": epoch,
                "selection_metric": "valid_label_accuracy",
                "valid_label_accuracy": valid_accuracy,
                "valid_macro_auprc": valid_macro_auprc,
            }, checkpoint)
        else:
            stale += 1
        if stale >= args.patience:
            break

    model.load_state_dict(torch.load(
        checkpoint, map_location=device, weights_only=False)["state_dict"])
    pd.DataFrame(history).to_csv(directory / "training.csv", index=False)
    stop_reason = "early stop" if stale >= args.patience else "finished"
    print(
        f"Fold {fold}: {stop_reason} at epoch {epoch}; "
        f"best epoch={best_epoch}, val Label-ACC={best_score:.4f}",
        flush=True,
    )
    return predict(test), predict(valid)


def multilabel_training_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/multilabel/CancerPPD.tsv")
    parser.add_argument("--run-name", default="cross_attention_cv10_seed5")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--check-only", action="store_true",
        help="Validate the local data and ten-fold splits without training",
    )
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1:
        parser.error("epochs, patience and batch-size must be positive")
    if Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}:
        parser.error("run-name must be a single directory name")

    data = (ROOT / args.data).resolve()
    if not data.is_relative_to(ROOT):
        parser.error(f"Data must reside inside the current project: {ROOT}")
    frame, y, labels = read_data(data)
    partitions = make_splits(y, args.seed)
    print(f"Detected {len(labels)} labels: {', '.join(labels)}", flush=True)
    print(
        f"Samples: {len(frame)}; positive counts: "
        f"{dict(zip(labels, y.sum(axis=0).astype(int).tolist()))}",
        flush=True,
    )
    if args.check_only:
        for split in partitions:
            sizes = {name: len(split[name]) for name in ("train", "valid", "test")}
            print(f"Fold {split['fold']}: {sizes}")
        return

    run = ROOT / "results" / "multilabel" / args.run_name
    if run.is_symlink() or run.parent.is_symlink():
        parser.error("Result directories must not be symbolic links")
    if run.exists():
        if not args.overwrite:
            parser.error(f"Run already exists: {run}; pass --overwrite to replace it")
        if not run.is_dir():
            parser.error(f"Run path is not a directory: {run}")
        shutil.rmtree(run)
    run.mkdir(parents=True)
    save_json(run / "config.json", {
        **vars(args),
        "labels": labels,
        "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
    })
    save_json(run / "splits.json", partitions)
    frame.to_csv(run / "dataset.tsv", sep="\t", index=False)


    sequences = frame["text"].tolist()
    graphs = make_data(sequences)
    sequence_tensor = make_seq_data(sequences, max(map(len, sequences)))
    oof = np.full(y.shape, np.nan, dtype=float)
    fold_ids = np.zeros(len(y), dtype=int)
    fold_metrics = []
    for split in partitions:
        fold = int(split["fold"])
        print(f"Fold {fold}/10", flush=True)
        directory = run / f"fold_{fold:02d}"
        directory.mkdir()
        test_probability, valid_probability = train_fold(
            args, sequence_tensor, graphs, y, labels, split, directory)
        test = np.asarray(split["test"], dtype=int)
        valid = np.asarray(split["valid"], dtype=int)
        oof[test] = test_probability
        fold_ids[test] = fold
        np.savez_compressed(
            directory / "graph_predictions.npz",
            rows=test,
            labels=labels,
            y_true=y[test],
            probabilities=test_probability,
        )
        fold_metrics.append({
            "fold": fold,
            "test_label_accuracy": float(accuracy_score(
                y[test].ravel(), (test_probability >= 0.5).ravel())),
            "test_macro_auprc": float(average_precision_score(
                y[test], test_probability, average="macro")),
            "valid_label_accuracy": float(accuracy_score(
                y[valid].ravel(), (valid_probability >= 0.5).ravel())),
        })

    if not np.isfinite(oof).all() or not np.all(fold_ids > 0):
        raise RuntimeError("OOF predictions are incomplete")
    save_json(run / "fold_metrics.json", fold_metrics)
    output = frame.copy()
    output["fold"] = fold_ids
    for label_index, label in enumerate(labels):
        output[f"{label}_true"] = y[:, label_index].astype(int)
        output[f"graph_{label}_probability"] = oof[:, label_index]
    output.to_csv(run / "oof_predictions.csv", index=False)
    print(f"Saved local cross-attention run: {run}", flush=True)


def extract_binary_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True,
                        help="Directory containing train.csv and test.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Newly trained cross-attention checkpoint")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    task, data = resolve_data(args.data)
    if task != "binary":
        parser.error("--data must be a binary dataset directory")
    train, test = binary_frames(data)
    train_sequences = train["Feature"].astype(str).str.strip().str.upper().tolist()
    test_sequences = test["Feature"].astype(str).str.strip().str.upper().tolist()
    y_train = train["Label"].to_numpy(dtype=int)
    _, valid_indices = train_test_split(
        np.arange(len(train)), test_size=0.2, random_state=5, stratify=y_train
    )

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    import dgl

    def all_probabilities(sequences: list[str]) -> dict[str, np.ndarray]:
        src_len = max(210, max(map(len, train_sequences + test_sequences), default=1))
        config = SimpleNamespace(
            task_type="Classification", output_layer=2, src_len=src_len,
            src_vocab_size_seq=21, d_model=64, d_ff=2048, d_k=64, d_v=64,
            n_layers=6, n_heads=8, dropout=0.1, d_graph=256,
            src_vocab_size_graph=15, d_fuse=128, conv_layers=2,
            GraphSAGE_aggregator="lstm", cross_layers=3,
            fusion_heads=4, fc_layers=4,
        )
        graphs = make_data(sequences)
        sequence_tensor = make_seq_data(sequences, src_len)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CrossModalNeuralPathway(config).to(device).eval()
        with torch.inference_mode():
            model(sequence_tensor[:1].to(device), dgl.batch(graphs[:1]).to(device))
        state = torch.load(checkpoint, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        allowed_missing = [name for name in missing
                           if name.endswith("norm.weight") or name.endswith("norm.bias")]
        if len(allowed_missing) != len(missing) or unexpected:
            raise RuntimeError(f"Incompatible checkpoint: missing={missing}, unexpected={unexpected}")
        output = {"fused": [], "sequence": [], "graph": []}
        with torch.inference_mode():
            for start in range(0, len(graphs), args.batch_size):
                result = model(
                    sequence_tensor[start:start + args.batch_size].to(device),
                    dgl.batch(graphs[start:start + args.batch_size]).to(device),
                )
                output["fused"].extend(result["out"].exp()[:, 1].cpu().tolist())
                output["sequence"].extend(result["out_seq"].exp()[:, 1].cpu().tolist())
                output["graph"].extend(result["out_graph"].exp()[:, 1].cpu().tolist())
        return {key: np.asarray(value, dtype=np.float64) for key, value in output.items()}

    valid = all_probabilities([train_sequences[index] for index in valid_indices])
    test_probabilities = all_probabilities(test_sequences)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        valid_indices=valid_indices,
        valid_probability=valid["fused"],
        test_probability=test_probabilities["fused"],
        valid_sequence_probability=valid["sequence"],
        test_sequence_probability=test_probabilities["sequence"],
        valid_graph_probability=valid["graph"],
        test_graph_probability=test_probabilities["graph"],
        checkpoint=np.asarray(str(checkpoint)),
    )
    print(json.dumps({
        "source_data": str(data),
        "n_valid": len(valid_indices),
        "n_test": len(test_sequences),
        "checkpoint": str(checkpoint),
        "output": str(args.output),
    }, ensure_ascii=False))


def enable_AutoDL_academic_acceleration() -> dict[str, str | None] | None:
    """Load AutoDL academic-acceleration proxy variables when available."""
    turbo_script = Path("/etc/network_turbo")
    if os.name == "nt" or not turbo_script.is_file():
        return None

    result = subprocess.run(
        [
            "bash",
            "-lc",
            "source /etc/network_turbo >/dev/null 2>&1 && env | grep -i proxy",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(
            "Warning: AutoDL academic acceleration could not be enabled; "
            "continuing with the existing network configuration.",
            flush=True,
        )
        return None

    proxy_updates = {}
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        if "proxy" in name.lower():
            proxy_updates[name] = value
    if not proxy_updates:
        print(
            "Warning: AutoDL academic acceleration returned no proxy variables; "
            "continuing with the existing network configuration.",
            flush=True,
        )
        return None

    previous = {name: os.environ.get(name) for name in proxy_updates}
    os.environ.update(proxy_updates)
    print("AutoDL academic acceleration enabled for ESM-2 loading.", flush=True)
    return previous


def restore_proxy_environment(previous: dict[str, str | None] | None) -> None:
    """Restore proxy variables changed by AutoDL academic acceleration."""
    if previous is None:
        return
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def PSRA_extraction_main() -> None:
    """Extract and cache the frozen ESM-2 representations used by PSRA."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True,
                        help="Binary directory or multilabel TSV/CSV")
    parser.add_argument("--model", choices=sorted(MODEL_SPECS),
                        default="esm2_t33_650M_UR50D")
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--save-residue", action="store_true",
        help="Also cache padded per-residue embeddings (float16) for sequence heads.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    loader_name, final_layer, default_layers = MODEL_SPECS[args.model]
    layers = sorted(set(args.layers or default_layers))
    if not layers or layers[0] < 1 or layers[-1] > final_layer:
        parser.error(f"layers must be within 1..{final_layer} for {args.model}")
    if args.batch_size < 1:
        parser.error("batch-size must be positive")

    output_dir = ROOT / "cache" / "esm" / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for key, sequences in esm_inputs(args.data):
        output = output_dir / f"{key}.npz"
        if output.exists() and not args.overwrite:
            with np.load(output, allow_pickle=False) as saved:
                if (saved["sequence_sha256"].item() == sequence_digest(sequences)
                        and saved["layers"].tolist() == layers
                        and saved["model_name"].item() == args.model
                        and (not args.save_residue or "lengths" in saved)):
                    print(f"Cached: {key} -> {output}", flush=True)
                    continue
        pending.append((key, sequences, output))
    if not pending:
        return

    previous_proxy = enable_AutoDL_academic_acceleration()
    try:
        import esm
        model, alphabet = getattr(esm.pretrained, loader_name)()
    finally:
        restore_proxy_environment(previous_proxy)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        model = model.half()
    model = model.to(device).eval()
    converter = alphabet.get_batch_converter()
    print(f"ESM device={device}; layers={layers}", flush=True)

    for key, sequences, output in pending:
        views = {f"layer{layer}_mean": [] for layer in layers}
        residue_views = ({f"layer{layer}_residue": [] for layer in layers}
                         if args.save_residue else {})
        for start in range(0, len(sequences), args.batch_size):
            current = sequences[start:start + args.batch_size]
            _, _, tokens = converter([(str(index), sequence)
                                      for index, sequence in enumerate(current, start)])
            tokens = tokens.to(device)
            with torch.inference_mode():
                representations = model(tokens, repr_layers=layers,
                                        return_contacts=False)["representations"]
            for local_index, sequence in enumerate(current):
                for layer in layers:
                    residue = representations[layer][local_index, 1:len(sequence) + 1].float()
                    views[f"layer{layer}_mean"].append(residue.mean(0).cpu().numpy())
                    if args.save_residue:
                        residue_views[f"layer{layer}_residue"].append(
                            residue.half().cpu().numpy()
                        )
            print(f"{key}: {min(start + args.batch_size, len(sequences))}/{len(sequences)}",
                  flush=True)
        payload = {name: np.asarray(value, dtype=np.float32)
                   for name, value in views.items()}
        if args.save_residue:
            max_length = max(map(len, sequences), default=0)
            for name, rows in residue_views.items():
                padded = np.zeros((len(rows), max_length, rows[0].shape[1]),
                                  dtype=np.float16)
                for row_index, row in enumerate(rows):
                    padded[row_index, :len(row)] = row
                payload[name] = padded
            payload["lengths"] = np.asarray(list(map(len, sequences)), dtype=np.int64)
        payload.update(sequences=np.asarray(sequences),
                       sequence_sha256=np.asarray(sequence_digest(sequences)),
                       layers=np.asarray(layers, dtype=np.int64),
                       model_name=np.asarray(args.model))
        np.savez_compressed(output, **payload)
        print(json.dumps({"dataset_id": key, "rows": len(sequences),
                          "output": str(output)}), flush=True)


# End-to-end training orchestration

def execute(command: list[str]) -> None:
    print("\n+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def train_registered_dataset(
        requested_name: str,
        epochs: int | None,
        patience: int,
        batch_size: int | None,
        micro_batch_size: int,
        esm_batch_size: int) -> dict:
    name, spec = dataset_spec(requested_name)
    data = Path(spec["data"]).resolve()
    key = dataset_key(data)
    if spec["task"] == "binary":
        epochs = 100 if epochs is None else epochs
        batch_size = 512 if batch_size is None else batch_size
        checkpoint = (
            ROOT / "models/two_branch/graph/binary" / f"{key}_seed5.pt")
        artifact = ROOT / "artifacts/graph" / f"{key}_seed5.npz"
        execute([
            sys.executable, "train.py", "_binary",
            "--data", str(data), "--mode", "train",
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--micro_batch_size", str(micro_batch_size),
            "--seed", "5",
        ])
        execute([
            sys.executable, "train.py", "_extract-binary",
            "--data", str(data), "--checkpoint", str(checkpoint),
            "--output", str(artifact),
        ])
    else:
        epochs = 300 if epochs is None else epochs
        batch_size = 512 if batch_size is None else batch_size
        run_name = f"cross_attention_{key}_seed5"
        execute([
            sys.executable, "train.py", "_multilabel",
            "--data", str(data), "--run-name", run_name, "--overwrite",
            "--epochs", str(epochs), "--patience", str(patience),
            "--batch-size", str(batch_size), "--seed", "5",
        ])

    execute([
        sys.executable, "train.py", "_extract-esm",
        "--data", str(data), "--layers", "12",
        "--batch-size", str(esm_batch_size), "--overwrite",
    ])
    result = run_datasets([name])[0]
    return result

INTERNAL_COMMANDS = {
    "_binary": launch_binary_training,
    "_multilabel": multilabel_training_main,
    "_extract-binary": extract_binary_main,
    "_extract-esm": PSRA_extraction_main,
}


def command_line() -> None:
    """Train HyCL-ACP on a registered dataset."""
    if len(sys.argv) > 1 and sys.argv[1] in INTERNAL_COMMANDS:
        command = sys.argv[1]
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        INTERNAL_COMMANDS[command]()
        return

    parser = argparse.ArgumentParser(
        description="Train HyCL-ACP on ACP or CancerPPD")
    parser.add_argument("dataset", choices=["ACP", "CancerPPD"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--micro-batch-size", type=int, default=512)
    parser.add_argument("--esm-batch-size", type=int, default=8)
    args = parser.parse_args()

    values = (
        args.patience,
        args.micro_batch_size,
        args.esm_batch_size,
        *(value for value in (args.epochs, args.batch_size)
          if value is not None),
    )
    if min(values) < 1:
        parser.error("epochs, patience and batch sizes must be positive")

    train_registered_dataset(
        args.dataset,
        args.epochs,
        args.patience,
        args.batch_size,
        args.micro_batch_size,
        args.esm_batch_size,
    )


if __name__ == "__main__":
    command_line()

