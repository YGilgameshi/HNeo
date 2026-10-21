"""
Per-Task性能分析模块

功能:
1. 计算每个task的AUROC, AUPRC, Accuracy等指标
2. 按HLA、Tissue、样本数等维度分析
3. 识别高性能和低性能的tasks
4. 生成可视化报告
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import json
from typing import Dict, List, Tuple
from collections import defaultdict


class PerTaskEvaluator:
    """Per-task性能评估器"""
    
    def __init__(self, model, device, task_manager, mode_config):
        """
        Args:
            model: 训练好的模型
            device: torch device
            task_manager: TaskManager实例
            mode_config: ModeConfig实例
        """
        self.model = model
        self.device = device
        self.task_manager = task_manager
        self.mode_config = mode_config
        self.model.eval()
        
        # 存储每个task的结果
        self.task_results = {}
    
    def evaluate_dataloader(self, dataloader, graph_wrapper):
        """
        评估整个dataloader，按task分组收集预测结果
        
        Args:
            dataloader: PyTorch DataLoader
            graph_wrapper: TaskGraphWrapper
            
        Returns:
            task_predictions: {task_id: {'y_true': [], 'y_pred': [], 'y_prob': []}}
        """
        task_predictions = defaultdict(lambda: {
            'y_true': [], 
            'y_pred': [], 
            'y_prob': [],
            'peptides': []
        })
        
        # 准备graph_data (和trainer中一致)
        graph_data = {
            'edge_index': graph_wrapper.edge_index,
            'edge_weight': graph_wrapper.edge_weight
        }

        all_tasks = self.task_manager.get_all_tasks()
        
        with torch.no_grad():
            for batch in dataloader:
                # 移动到device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}

                # 在 evaluate_dataloader 里，model forward 之前加
                if batch['peptide_len'].dim() == 0:
                    batch['peptide_len'] = batch['peptide_len'].unsqueeze(0)
                if batch['peptide'].dim() == 1:
                    batch['peptide'] = batch['peptide'].unsqueeze(0)

                # 前向传播
                logits = self.model(batch, graph_data)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                labels = batch['label'].cpu().numpy()
                task_idxs = batch['task_idx'].cpu().numpy()
                
                # 按task分组
                for i in range(len(labels)):
                    task_idx = task_idxs[i]
                    
                    # 找到对应的task_id
                    task_id = None
                    for tid, task in all_tasks.items():
                        if graph_wrapper.task_to_idx.get(tid) == task_idx:
                            task_id = tid
                            break
                    
                    if task_id is not None:
                        task_predictions[task_id]['y_true'].append(labels[i])
                        task_predictions[task_id]['y_pred'].append(preds[i])
                        task_predictions[task_id]['y_prob'].append(probs[i])
        
        return task_predictions
    
    def compute_task_metrics(self, task_predictions):
        """
        计算每个task的性能指标
        
        Args:
            task_predictions: evaluate_dataloader的输出
            
        Returns:
            task_metrics: {task_id: {metric_name: value}}
        """
        task_metrics = {}
        all_tasks = self.task_manager.get_all_tasks()
        
        for task_id, preds in task_predictions.items():
            y_true = np.array(preds['y_true'])
            y_pred = np.array(preds['y_pred'])
            y_prob = np.array(preds['y_prob'])
            
            # 基本信息
            task = all_tasks[task_id]
            n_samples = len(y_true)
            n_positive = np.sum(y_true)
            n_negative = n_samples - n_positive
            
            # 计算各种指标
            metrics = {
                'task_id': task_id,
                'hla': task.hla,
                'n_samples': n_samples,
                'n_positive': n_positive,
                'n_negative': n_negative,
                'pos_ratio': n_positive / n_samples if n_samples > 0 else 0
            }
            
            # 添加tissue信息（Mode 2）
            if hasattr(task, 'tissue'):
                metrics['tissue'] = task.tissue
            
            # 只有当有正负样本时才计算AUROC/AUPRC
            if n_positive > 0 and n_negative > 0:
                try:
                    metrics['auroc'] = roc_auc_score(y_true, y_prob)
                except:
                    metrics['auroc'] = np.nan
                
                try:
                    metrics['auprc'] = average_precision_score(y_true, y_prob)
                except:
                    metrics['auprc'] = np.nan
            else:
                metrics['auroc'] = np.nan
                metrics['auprc'] = np.nan
            
            # 计算准确率
            if n_samples > 0:
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
            else:
                metrics['accuracy'] = np.nan
            
            # 计算precision, recall, f1
            if n_samples > 0:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average='binary', zero_division=0
                )
                metrics['precision'] = precision
                metrics['recall'] = recall
                metrics['f1'] = f1
            else:
                metrics['precision'] = np.nan
                metrics['recall'] = np.nan
                metrics['f1'] = np.nan
            
            # 计算PPV@k (Positive Predictive Value at top k)
            if n_samples >= 10:
                top_k = min(10, n_samples)
                top_k_idx = np.argsort(y_prob)[-top_k:]
                metrics['ppv_at_10'] = np.mean(y_true[top_k_idx])
            else:
                metrics['ppv_at_10'] = np.nan
            
            task_metrics[task_id] = metrics
        
        return task_metrics
    
    def analyze_and_save(self, task_metrics, output_dir, split_name='test'):
        """
        分析并保存per-task结果
        
        Args:
            task_metrics: compute_task_metrics的输出
            output_dir: 输出目录
            split_name: 'test', 'val'等
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 转换为DataFrame
        df = pd.DataFrame([metrics for metrics in task_metrics.values()])
        
        # 排序（按AUROC降序）
        df = df.sort_values('auroc', ascending=False, na_position='last')
        
        # 保存完整结果
        csv_file = output_dir / f'per_task_metrics_{split_name}.csv'
        df.to_csv(csv_file, index=False)
        print(f"\n✓ Per-task metrics saved to: {csv_file}")
        
        # 生成统计摘要
        summary = self._generate_summary(df)
        
        # 保存摘要
        summary_file = output_dir / f'per_task_summary_{split_name}.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Summary saved to: {summary_file}")
        
        # 打印摘要
        self._print_summary(summary, df)
        
        # 识别高性能和低性能tasks
        self._identify_top_bottom_tasks(df, output_dir, split_name)
        
        return df, summary

    def _generate_summary(self, df):
        """生成统计摘要"""
        summary = {
            'overall': {
                'n_tasks': len(df),
                'mean_auroc': float(df['auroc'].mean()),
                'median_auroc': float(df['auroc'].median()),
                'std_auroc': float(df['auroc'].std()),
                'min_auroc': float(df['auroc'].min()),
                'max_auroc': float(df['auroc'].max()),
                'mean_auprc': float(df['auprc'].mean()),
                'median_auprc': float(df['auprc'].median()),
                'std_auprc': float(df['auprc'].std()),
                'mean_accuracy': float(df['accuracy'].mean()),
                'mean_f1': float(df['f1'].mean())
            }
        }

        # ========== 修复：按HLA统计 ==========
        if 'hla' in df.columns:
            hla_stats = df.groupby('hla').agg({
                'auroc': ['mean', 'count'],
                'accuracy': 'mean',
                'n_samples': 'sum'
            })

            # ★ 关键修复：正确处理 MultiIndex
            summary['by_hla'] = {}
            for hla in hla_stats.index:
                summary['by_hla'][hla] = {
                    'mean_auroc': float(hla_stats.loc[hla, ('auroc', 'mean')]),
                    'n_tasks': int(hla_stats.loc[hla, ('auroc', 'count')]),
                    'mean_accuracy': float(hla_stats.loc[hla, ('accuracy', 'mean')]),
                    'total_samples': int(hla_stats.loc[hla, ('n_samples', 'sum')])
                }

        # ========== 修复：按Tissue统计 ==========
        if 'tissue' in df.columns:
            tissue_stats = df.groupby('tissue').agg({
                'auroc': ['mean', 'count'],
                'accuracy': 'mean',
                'n_samples': 'sum'
            })

            summary['by_tissue'] = {}
            for tissue in tissue_stats.index:
                summary['by_tissue'][tissue] = {
                    'mean_auroc': float(tissue_stats.loc[tissue, ('auroc', 'mean')]),
                    'n_tasks': int(tissue_stats.loc[tissue, ('auroc', 'count')]),
                    'mean_accuracy': float(tissue_stats.loc[tissue, ('accuracy', 'mean')]),
                    'total_samples': int(tissue_stats.loc[tissue, ('n_samples', 'sum')])
                }

        # ========== 修复：按样本数分组 ==========
        df['sample_bin'] = pd.cut(df['n_samples'],
                                  bins=[0, 50, 100, 500, float('inf')],
                                  labels=['<50', '50-100', '100-500', '>500'])

        sample_stats = df.groupby('sample_bin').agg({
            'auroc': ['mean', 'count']
        })

        summary['by_sample_size'] = {}
        for bin_name in sample_stats.index:
            summary['by_sample_size'][str(bin_name)] = {
                'mean_auroc': float(sample_stats.loc[bin_name, ('auroc', 'mean')]),
                'n_tasks': int(sample_stats.loc[bin_name, ('auroc', 'count')])
            }

        return summary
    
    def _print_summary(self, summary, df):
        """打印摘要"""
        print("\n" + "="*80)
        print("Per-Task Performance Summary")
        print("="*80)
        
        # 总体统计
        overall = summary['overall']
        print(f"\n📊 Overall Statistics:")
        print(f"  Total tasks: {overall['n_tasks']}")
        print(f"  Mean AUROC: {overall['mean_auroc']:.4f} ± {overall['std_auroc']:.4f}")
        print(f"  Median AUROC: {overall['median_auroc']:.4f}")
        print(f"  AUROC range: [{overall['min_auroc']:.4f}, {overall['max_auroc']:.4f}]")
        print(f"  Mean Accuracy: {overall['mean_accuracy']:.4f}")
        print(f"  Mean F1: {overall['mean_f1']:.4f}")
        
        # 按样本数统计
        if 'by_sample_size' in summary:
            print(f"\n📈 Performance by Sample Size:")
            for size, stats in summary['by_sample_size'].items():
                print(f"  {size:>10} samples: AUROC {stats['mean_auroc']:.4f} ({stats['n_tasks']} tasks)")
        
        # 按HLA统计（只显示top 10）
        if 'by_hla' in summary:
            print(f"\n🧬 Top 10 HLAs by Mean AUROC:")
            hla_sorted = sorted(summary['by_hla'].items(), 
                               key=lambda x: x[1]['mean_auroc'], 
                               reverse=True)[:10]
            for hla, stats in hla_sorted:
                print(f"  {hla:20s}: AUROC {stats['mean_auroc']:.4f} "
                      f"({stats['n_tasks']} tasks, {stats['total_samples']} samples)")
        
        # 按Tissue统计
        if 'by_tissue' in summary:
            print(f"\n🔬 Performance by Tissue:")
            tissue_sorted = sorted(summary['by_tissue'].items(), 
                                  key=lambda x: x[1]['mean_auroc'], 
                                  reverse=True)
            for tissue, stats in tissue_sorted:
                print(f"  {tissue:20s}: AUROC {stats['mean_auroc']:.4f} "
                      f"({stats['n_tasks']} tasks, {stats['total_samples']} samples)")
    
    def _identify_top_bottom_tasks(self, df, output_dir, split_name):
        """识别并保存高性能和低性能tasks"""
        # Top 20 tasks
        top_20 = df.head(20)[['task_id', 'hla', 'auroc', 'accuracy', 'f1', 'n_samples']]
        if 'tissue' in df.columns:
            top_20 = df.head(20)[['task_id', 'hla', 'tissue', 'auroc', 'accuracy', 'f1', 'n_samples']]
        
        top_file = output_dir / f'top_20_tasks_{split_name}.csv'
        top_20.to_csv(top_file, index=False)
        
        print(f"\n🏆 Top 20 Tasks:")
        for idx, row in top_20.iterrows():
            if 'tissue' in row:
                print(f"  {row['task_id']:40s} | {row['hla']:15s} | {row['tissue']:15s} | "
                      f"AUROC: {row['auroc']:.4f} | Acc: {row['accuracy']:.4f}")
            else:
                print(f"  {row['task_id']:40s} | {row['hla']:15s} | "
                      f"AUROC: {row['auroc']:.4f} | Acc: {row['accuracy']:.4f}")
        
        # Bottom 20 tasks (只显示有效AUROC的)
        df_valid = df[~df['auroc'].isna()]
        if len(df_valid) >= 20:
            bottom_20 = df_valid.tail(20)[['task_id', 'hla', 'auroc', 'accuracy', 'f1', 'n_samples']]
            if 'tissue' in df.columns:
                bottom_20 = df_valid.tail(20)[['task_id', 'hla', 'tissue', 'auroc', 'accuracy', 'f1', 'n_samples']]
            
            bottom_file = output_dir / f'bottom_20_tasks_{split_name}.csv'
            bottom_20.to_csv(bottom_file, index=False)
            
            print(f"\n⚠️  Bottom 20 Tasks:")
            for idx, row in bottom_20.iterrows():
                if 'tissue' in row:
                    print(f"  {row['task_id']:40s} | {row['hla']:15s} | {row['tissue']:15s} | "
                          f"AUROC: {row['auroc']:.4f} | Acc: {row['accuracy']:.4f}")
                else:
                    print(f"  {row['task_id']:40s} | {row['hla']:15s} | "
                          f"AUROC: {row['auroc']:.4f} | Acc: {row['accuracy']:.4f}")


def evaluate_per_task_performance(
    model,
    test_loader,
    graph_wrapper,
    task_manager,
    mode_config,
    output_dir,
    test_task_datasets=None,
    device='cuda'
):
    """
    便捷函数：评估per-task性能

    Args:
        model:              训练好的模型
        test_loader:        测试集DataLoader
        graph_wrapper:      TaskGraphWrapper
        task_manager:       TaskManager
        mode_config:        ModeConfig
        output_dir:         输出目录
        test_task_datasets: {task_id: DataFrame}，Mode 2下传入以生成per-tissue汇总；
                            Mode 1不传（None），跳过tissue分析
        device:             torch device

    Returns:
        df:      Per-task结果DataFrame
        summary: 统计摘要
    """
    from src.config.mode_config import TrainingMode

    evaluator = PerTaskEvaluator(model, device, task_manager, mode_config)

    print("\n" + "="*80)
    print("Evaluating Per-Task Performance...")
    print("="*80)

    # ── 1. 收集预测结果 & 计算 per-task 指标 ──
    task_predictions = evaluator.evaluate_dataloader(test_loader, graph_wrapper)
    task_metrics = evaluator.compute_task_metrics(task_predictions)

    # ── 2. 分析并保存 per-task CSV / JSON ──
    df, summary = evaluator.analyze_and_save(task_metrics, output_dir, split_name='test')

    # ── 3. Per-tissue 分析（仅 Mode 2，且传入了 test_task_datasets）──
    is_mode2 = (hasattr(mode_config, 'mode') and
                mode_config.mode == TrainingMode.HLA_TISSUE)

    if not is_mode2:
        # Mode 1：无 tissue，直接返回
        return df, summary

    if test_task_datasets is None:
        print("\n  ℹ️  test_task_datasets 未传入，跳过 per-tissue 分析")
        print("      如需 tissue 层面汇总，请在调用时传入 test_task_datasets=test_task_datasets")
        return df, summary

    print("\n" + "="*80)
    print("Per-Tissue Analysis (Mode 2)")
    print("="*80)

    # Mode 2 每个 task 已经是 HLA×Tissue 组合
    # task.tissue 直接给出 tissue 标签，无需从列里提取
    all_tasks = task_manager.get_all_tasks()

    # 统计 test set tissue 分布
    tissue_pos_counts = defaultdict(int)
    for task_id, task_df in test_task_datasets.items():
        task = all_tasks.get(task_id)
        if task is None or not hasattr(task, 'tissue'):
            continue
        n_pos = int((task_df['label'] == 1).sum())
        tissue_pos_counts[task.tissue] += n_pos

    print(f"\n  Tissue distribution in test set (positive samples):")
    for tissue, cnt in sorted(tissue_pos_counts.items(), key=lambda x: -x[1]):
        print(f"    {tissue:25s}: {cnt:5d} positive samples")

    # 构建 per-tissue 记录（每条 = 一个 HLA×Tissue task 的性能）
    per_tissue_records = []
    for task_id, task_df in test_task_datasets.items():
        if task_id not in task_metrics:
            continue
        task = all_tasks.get(task_id)
        if task is None or not hasattr(task, 'tissue'):
            continue

        tm = task_metrics[task_id]
        per_tissue_records.append({
            'tissue':     task.tissue,
            'hla':        task.hla,
            'task_id':    task_id,
            'n_tasks':    1,
            'n_positive': int(tm['n_positive']),
            'n_negative': int(tm['n_negative']),
            'auroc':      tm.get('auroc',    np.nan),
            'auprc':      tm.get('auprc',    np.nan),
            'f1':         tm.get('f1',       np.nan),
            'ppv_at_10':  tm.get('ppv_at_10', np.nan),
        })

    if not per_tissue_records:
        print("  ⚠️  没有可用的 per-tissue 记录，跳过汇总")
        return df, summary

    per_tissue_df = pd.DataFrame(per_tissue_records)

    # 按 tissue 加权平均（权重 = n_positive，避免小任务主导）
    def _weighted_mean(vals, weights):
        """安全加权平均：过滤 nan，weights 与 vals 对齐"""
        mask = ~np.isnan(vals)
        if mask.sum() == 0:
            return np.nan
        return float(np.average(vals[mask], weights=weights[mask]))

    tissue_records = []
    for tissue, grp in per_tissue_df.groupby('tissue'):
        w = grp['n_positive'].values.astype(float)
        tissue_records.append({
            'tissue':      tissue,
            'n_hla_tasks': len(grp),
            'n_positive':  int(grp['n_positive'].sum()),
            'n_negative':  int(grp['n_negative'].sum()),
            'auroc':       _weighted_mean(grp['auroc'].values,    w),
            'auprc':       _weighted_mean(grp['auprc'].values,    w),
            'f1':          _weighted_mean(grp['f1'].values,       w),
            'ppv_at_10':   _weighted_mean(grp['ppv_at_10'].values, w),
        })

    tissue_summary_df = (pd.DataFrame(tissue_records)
                         .sort_values('auroc', ascending=False, na_position='last')
                         .reset_index(drop=True))

    # 保存
    tissue_file = Path(output_dir) / 'per_tissue_summary.csv'
    tissue_summary_df.to_csv(tissue_file, index=False)
    print(f"\n  ✓ Per-tissue summary saved to: {tissue_file}")

    # 保存明细（每个 HLA×Tissue task 一行）
    detail_file = Path(output_dir) / 'per_tissue_task_detail.csv'
    per_tissue_df.to_csv(detail_file, index=False)
    print(f"  ✓ Per-tissue task detail saved to: {detail_file}")

    # 打印汇总
    print(f"\n  {'Tissue':<25} {'Tasks':>6} {'Pos':>6} {'AUROC':>7} {'AUPRC':>7} {'F1':>7} {'PPV@10':>8}")
    print("  " + "-" * 68)
    for _, row in tissue_summary_df.iterrows():
        auroc_s   = f"{row['auroc']:.4f}"   if not np.isnan(row['auroc'])   else "  N/A "
        auprc_s   = f"{row['auprc']:.4f}"   if not np.isnan(row['auprc'])   else "  N/A "
        f1_s      = f"{row['f1']:.4f}"      if not np.isnan(row['f1'])      else "  N/A "
        ppv_s     = f"{row['ppv_at_10']:.4f}" if not np.isnan(row['ppv_at_10']) else "  N/A "
        print(f"  {row['tissue']:<25} {int(row['n_hla_tasks']):>6} "
              f"{int(row['n_positive']):>6} {auroc_s:>7} {auprc_s:>7} {f1_s:>7} {ppv_s:>8}")

    # 写入 summary JSON
    summary['by_tissue_weighted'] = {
        row['tissue']: {
            'n_hla_tasks': int(row['n_hla_tasks']),
            'n_positive':  int(row['n_positive']),
            'auroc':       row['auroc']   if not np.isnan(row['auroc'])   else None,
            'auprc':       row['auprc']   if not np.isnan(row['auprc'])   else None,
            'f1':          row['f1']      if not np.isnan(row['f1'])      else None,
            'ppv_at_10':   row['ppv_at_10'] if not np.isnan(row['ppv_at_10']) else None,
        }
        for _, row in tissue_summary_df.iterrows()
    }

    # 更新保存 summary JSON（加入 tissue 维度）
    summary_file = Path(output_dir) / 'per_task_summary_test.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Summary JSON updated with tissue stats: {summary_file}")

    return df, summary



if __name__ == "__main__":
    print("Per-Task Evaluator Module")
    print("="*80)
    print("\nFeatures:")
    print("  ✓ Per-task AUROC, AUPRC, Accuracy, F1")
    print("  ✓ Statistics by HLA, Tissue, Sample Size")
    print("  ✓ Top/Bottom task identification")
    print("  ✓ Comprehensive CSV reports")
