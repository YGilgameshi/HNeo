"""
Mode-Aware Unified Trainer

统一的训练接口,支持:
- Mode 1/2/3
- MAML / Standard 训练
- Task-Balanced Sampling
- Task-Weighted Loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import random
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.config.mode_config import ModeConfig, TrainingMode
from src.data.task_definition import TaskManager
from src.data.dataset import ModeAwareDataset, collate_fn_mode_aware
from src.models.full_model import ImmuneAppModel
from src.graph.task_graph import ModeAwareTaskGraphBuilder, TaskGraphWrapper


def set_reproducible_seed(seed: int):
    """设置本次训练使用的随机种子，并尽量固定CUDA/cuDNN行为。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_task_weights(task_datasets, smoothing='log'):
    """
    计算task权重 (用于loss weighting)

    Args:
        task_datasets: Dict[task_id, DataFrame]
        smoothing: 'none', 'sqrt', 'log'
            - none: 直接反比 (不稳定)
            - sqrt: 平方根平滑 (中等)
            - log: 对数平滑 (最稳定,推荐)

    Returns:
        torch.Tensor: [n_tasks] 权重向量,归一化到均值=1

    Example:
        >>> weights = compute_task_weights(task_datasets, smoothing='log')
        >>> # 训练时使用
        >>> loss = criterion(logits, labels)
        >>> weighted_loss = (loss * weights[task_idx]).mean()
    """
    task_sizes = np.array([len(df) for df in task_datasets.values()])
    total = task_sizes.sum()
    n = len(task_sizes)

    if smoothing == 'log':
        # 对数平滑 - 最稳定,适合极端不平衡 (600倍差距)
        weights = np.log(1 + total / (n * task_sizes))
    elif smoothing == 'sqrt':
        # 平方根平滑 - 中等平滑
        weights = np.sqrt(total / (n * task_sizes))
    else:
        # 不平滑 - 直接反比,可能不稳定
        weights = total / (n * task_sizes)

    # 归一化到均值=1 (保持总loss scale不变)
    weights = weights / weights.mean()

    print(f"\n✓ Task权重统计 (smoothing={smoothing}):")
    print(f"  最小: {weights.min():.3f}")
    print(f"  25%: {np.percentile(weights, 25):.3f}")
    print(f"  中位数: {np.median(weights):.3f}")
    print(f"  75%: {np.percentile(weights, 75):.3f}")
    print(f"  最大: {weights.max():.3f}")
    print(f"  均值: {weights.mean():.3f} (归一化)")

    return torch.tensor(weights, dtype=torch.float32)


def compute_pos_weight(task_datasets):
    """
    计算正负样本权重 (用于class imbalance)

    Args:
        task_datasets: Dict[task_id, DataFrame] 包含label列

    Returns:
        float: pos_weight = 负样本数 / 正样本数

    Example:
        >>> pos_weight = compute_pos_weight(task_datasets)
        >>> criterion = nn.BCEWithLogitsLoss(
        ...     pos_weight=torch.tensor([pos_weight]),
        ...     reduction='none'
        ... )
    """
    total_pos = 0
    total_neg = 0

    for task_df in task_datasets.values():
        pos = (task_df['label'] == 1).sum()
        neg = (task_df['label'] == 0).sum()
        total_pos += pos
        total_neg += neg

    if total_pos == 0:
        print(f"\n⚠️  警告: 数据中没有正样本!")
        return 1.0

    pos_weight = total_neg / total_pos

    print(f"\n✓ Class权重计算:")
    print(f"  正样本: {total_pos:,}")
    print(f"  负样本: {total_neg:,}")
    print(f"  比例 (负:正): {pos_weight:.2f}:1")
    print(f"  pos_weight: {pos_weight:.2f}")
    print(f"\n  说明: pos_weight会给正样本 {pos_weight:.1f}倍 的loss权重")
    print(f"        这样可以缓解类别不平衡问题")

    return pos_weight


class StandardTrainer:
    """
    标准训练器 (非MAML)

    支持:
    - Task权重 (处理task样本数不平衡)
    - Class权重 (处理正负样本不平衡)
    """

    def __init__(self, model, optimizer, device='cuda', pos_weight=None):
        """
        初始化Trainer

        Args:
            model: 模型
            optimizer: 优化器
            device: 设备
            pos_weight: 正样本权重 (float或None)
                如果提供,会用于BCEWithLogitsLoss处理类别不平衡
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device

        # === 创建loss函数 (支持pos_weight) ===
        if pos_weight is not None:
            print(f"\n✓ 使用Class权重:")
            print(f"  pos_weight = {pos_weight:.2f}")
            print(f"  正样本的loss权重是负样本的 {pos_weight:.1f} 倍")

            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight]).to(device),
                reduction='none'  # 必须是'none'才能应用task权重
            )
        else:
            print(f"\n  不使用Class权重 (标准BCE loss)")
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def train_epoch(self, train_loader, graph_wrapper, task_weights=None):
        """
        训练一个epoch

        Args:
            train_loader: DataLoader
            graph_wrapper: TaskGraphWrapper
            task_weights: torch.Tensor [n_tasks] 或 None
                如果提供,将对loss应用task权重

        Returns:
            dict: {'loss': average_loss}
        """
        self.model.train()
        epoch_losses = []

        graph_data = {
            'edge_index': graph_wrapper.edge_index,
            'edge_weight': graph_wrapper.edge_weight
        }

        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            # 移动到device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward
            logits = self.model(batch, graph_data)
            loss = self.criterion(logits.squeeze(), batch['label'])

            # === 应用task权重 (如果提供) ===
            if task_weights is not None:
                # 获取每个样本的task权重
                weights = task_weights[batch['task_idx']]
                # 加权平均loss
                loss = (loss * weights).mean()
            else:
                # 普通平均
                loss = loss.mean()

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return {'loss': np.mean(epoch_losses)}


class MAMLTrainer:
    """
    MAML训练器

    Model-Agnostic Meta-Learning
    """

    def __init__(self, model, inner_lr=0.01, meta_lr=0.001, inner_steps=5, device='cuda'):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps
        self.device = device

        # Meta optimizer
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=meta_lr)
        self.criterion = nn.BCEWithLogitsLoss()

    def adapt(self, support_batch, graph_data):
        """
        Inner loop: 在support set上适应

        Returns:
            adapted_params: 适应后的参数
        """
        # Clone当前参数
        adapted_params = {name: param.clone() for name, param in self.model.named_parameters()}

        for _ in range(self.inner_steps):
            # Forward
            logits = self.model(support_batch, graph_data)
            loss = self.criterion(logits.squeeze(), support_batch['label'])

            # Compute gradients
            grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)

            # Update adapted params
            adapted_params = {
                name: param - self.inner_lr * grad
                for (name, param), grad in zip(self.model.named_parameters(), grads)
            }

        return adapted_params, loss.item()

    def train_epoch(self, train_loader, graph_wrapper, episodes_per_epoch=100):
        """
        训练一个epoch (meta-training)

        Args:
            train_loader: DataLoader
            graph_wrapper: TaskGraphWrapper
            episodes_per_epoch: 每个epoch的episodes数
        """
        self.model.train()

        epoch_support_losses = []
        epoch_query_losses = []

        graph_data = {
            'edge_index': graph_wrapper.edge_index,
            'edge_weight': graph_wrapper.edge_weight
        }

        pbar = tqdm(range(episodes_per_epoch), desc="MAML Training")

        for _ in pbar:
            # 采样episode (support + query)
            try:
                batch = next(iter(train_loader))
            except:
                continue

            batch = {k: v.to(self.device) for k, v in batch.items()}

            # 分割support和query
            mid = len(batch['label']) // 2
            support_batch = {k: v[:mid] for k, v in batch.items()}
            query_batch = {k: v[mid:] for k, v in batch.items()}

            # Inner loop: adapt on support
            adapted_params, support_loss = self.adapt(support_batch, graph_data)

            # Outer loop: evaluate on query
            # 临时替换模型参数
            original_params = {name: param.clone() for name, param in self.model.named_parameters()}

            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    param.copy_(adapted_params[name])

            # Query loss
            query_logits = self.model(query_batch, graph_data)
            query_loss = self.criterion(query_logits.squeeze(), query_batch['label'])

            # Meta-update
            self.meta_optimizer.zero_grad()
            query_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.meta_optimizer.step()

            # 恢复原始参数
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    param.copy_(original_params[name])

            epoch_support_losses.append(support_loss)
            epoch_query_losses.append(query_loss.item())

            pbar.set_postfix({
                'support_loss': f'{support_loss:.4f}',
                'query_loss': f'{query_loss.item():.4f}'
            })

        return {
            'loss': np.mean(epoch_query_losses),
            'support_loss': np.mean(epoch_support_losses)
        }


class Evaluator:
    """评估器"""

    def __init__(self, device='cuda'):
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()

    def evaluate(self, model, eval_loader, graph_wrapper):
        """
        评估模型

        Returns:
            metrics: dict with 'loss', 'auroc', 'auprc', 'ppv', 'mean_task_auroc'
                - 'auroc': pooled AUROC（所有task的样本拼成一个数组算一次，旧口径，
                  保留下来仅用于参考/排错，不再用于挑选 best_model.pt）
                - 'mean_task_auroc': 每个 task 分别算 AUROC 再取平均（D2 修复：论文
                  实际声称使用的口径，现在用它来挑选 best_model.pt）
        """
        model.eval()

        all_logits = []
        all_labels = []
        all_task_idx = []
        epoch_losses = []

        graph_data = {
            'edge_index': graph_wrapper.edge_index,
            'edge_weight': graph_wrapper.edge_weight
        }

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                logits = model(batch, graph_data)
                loss = self.criterion(logits.squeeze(), batch['label'])

                all_logits.append(logits.cpu().numpy())
                all_labels.append(batch['label'].cpu().numpy())
                all_task_idx.append(batch['task_idx'].cpu().numpy())
                epoch_losses.append(loss.item())

        # 合并结果
        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels)
        all_task_idx = np.concatenate(all_task_idx)
        all_probs = torch.sigmoid(torch.tensor(all_logits)).numpy()

        # 计算metrics
        metrics = {
            'loss': np.mean(epoch_losses),
            'auroc': roc_auc_score(all_labels, all_probs),
            'auprc': average_precision_score(all_labels, all_probs),
            'ppv': self._compute_ppv_at_k(all_labels, all_probs, k=100),
            'mean_task_auroc': self._compute_mean_task_auroc(all_labels, all_probs, all_task_idx),
        }

        return metrics

    def _compute_mean_task_auroc(self, labels, probs, task_idx):
        """
        D2 修复：按 task 分别算 AUROC 再取平均，而不是把所有 task 的样本
        拼起来算一次 pooled AUROC。

        某个 task 在这次 eval 里只含单一类别（全正或全负）时 AUROC 无定义，
        跳过该 task、不计入平均（不当 0 分处理，避免拖累整体均值）。
        """
        task_aurocs = []
        for t in np.unique(task_idx):
            mask = task_idx == t
            task_labels = labels[mask]
            if len(np.unique(task_labels)) < 2:
                continue
            task_aurocs.append(roc_auc_score(task_labels, probs[mask]))

        if len(task_aurocs) == 0:
            return float('nan')
        return float(np.mean(task_aurocs))

    def _compute_ppv_at_k(self, labels, probs, k=100):
        """计算PPV@k"""
        sorted_idx = np.argsort(probs)[::-1][:k]
        return labels[sorted_idx].mean()


def train_model(
        task_manager: TaskManager,
        task_datasets: dict,
        mode_config: ModeConfig,
        output_dir: str = 'models/trained',
        n_epochs: int = 50,
        batch_size: int = 32,
        device: str = 'cuda',
        val_task_datasets: dict = None,
        test_task_datasets: dict = None,
        # === Task-Balanced参数 ===
        use_task_balanced: bool = False,
        samples_per_task: int = None,
        use_task_weighting: bool = False,
        weight_smoothing: str = 'log',
        **kwargs
):
    """
    统一训练函数 (支持Task-Balanced Sampling)

    支持所有Mode和训练方法 (MAML/Standard)

    Args:
        task_manager: TaskManager实例
        task_datasets: Dict[task_id, DataFrame] 训练数据
        mode_config: ModeConfig实例
        output_dir: 模型保存目录
        n_epochs: 训练轮数
        batch_size: batch大小
        device: 'cuda' or 'cpu'
        val_task_datasets: 验证数据 (可选)
        test_task_datasets: 测试数据 (可选)
        # === 新增Task-Balanced参数 ===
        use_task_balanced: 是否使用Task-Balanced采样
        samples_per_task: 每个task每epoch采样数 (None=自动)
        use_task_weighting: 是否使用task权重loss
        weight_smoothing: 权重平滑方法 ('none'/'sqrt'/'log')
        **kwargs: 其他参数

    Returns:
        model, history
    """

    random_seed = int(kwargs.get('random_seed', 42))
    set_reproducible_seed(random_seed)

    print(f"\n{'=' * 80}")
    print(f"Training {mode_config.task_type_name} Model")
    print(f"  Method: {'MAML' if mode_config.use_maml else 'Standard'}")
    print(f"  Device: {device}")
    print(f"  Random seed: {random_seed}")
    print(f"  Task-Balanced: {use_task_balanced}")
    if use_task_balanced:
        print(f"    Samples per task: {samples_per_task or 'auto'}")
    print(f"  Task-Weighting: {use_task_weighting}")
    if use_task_weighting:
        print(f"    Smoothing: {weight_smoothing}")
    print(f"{'=' * 80}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========== 1. 构建Task Graph ==========
    print(f"\n{'=' * 80}")
    print("Building Task Graph...")
    print(f"{'=' * 80}")

    graph_builder = ModeAwareTaskGraphBuilder(
        task_manager,
        mode_config,
        hla_sequences_file=kwargs.get('hla_sequences_file', 'configs/hla_sequences.json'),
        allow_random_hla_fallback=kwargs.get('allow_random_hla_fallback', False),
    )

    graph, similarity_matrix = graph_builder.build_and_save(
        output_dir / 'task_graph',
        threshold=kwargs.get('graph_threshold', 0.3)
    )

    graph_wrapper = TaskGraphWrapper(output_dir / 'task_graph')
    graph_wrapper.to(device)

    # ========== 2. 创建DataLoaders ==========
    print(f"\n{'=' * 80}")
    print("Creating DataLoaders...")
    print(f"{'=' * 80}")

    # 如果没有提供val/test,用train的一部分
    if val_task_datasets is None:
        val_task_datasets = task_datasets  # 简化处理
    if test_task_datasets is None:
        test_task_datasets = task_datasets

    # ========== 创建Datasets (支持Task-Balanced) ==========
    if use_task_balanced:
        # 使用分层采样的Task-Balanced Dataset
        from src.data.task_balanced_dataset import AdaptiveTaskBalancedDataset

        # 确定采样策略
        sampling_strategy = kwargs.get('sampling_strategy', 'adaptive')
        negative_ratio = mode_config.negative_ratio
        print(f"\n使用分层采样策略: {sampling_strategy}")

        # 创建Task-Balanced训练集（分层采样）
        train_dataset = AdaptiveTaskBalancedDataset(
            task_datasets=task_datasets,
            task_manager=task_manager,
            mode_config=mode_config,
            graph_wrapper=graph_wrapper,
            sampling_strategy=sampling_strategy,
            negative_ratio=negative_ratio,
            random_seed=random_seed,
        )
    else:
        # 使用原始Dataset
        train_dataset = ModeAwareDataset(
            task_datasets=task_datasets,
            task_manager=task_manager,
            mode_config=mode_config,
            graph_wrapper=graph_wrapper
        )

    # Val和Test始终用原始Dataset (不需要balance)
    val_dataset = ModeAwareDataset(
        task_datasets=val_task_datasets,
        task_manager=task_manager,
        mode_config=mode_config,
        graph_wrapper=graph_wrapper
    )

    test_dataset = ModeAwareDataset(
        task_datasets=test_task_datasets,
        task_manager=task_manager,
        mode_config=mode_config,
        graph_wrapper=graph_wrapper
    )

    # 创建DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_mode_aware,
        num_workers=kwargs.get('num_workers', 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_mode_aware,
        num_workers=kwargs.get('num_workers', 0)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_mode_aware,
        num_workers=kwargs.get('num_workers', 0)
    )

    print(f"\n✓ DataLoaders created:")
    print(f"  Train: {len(train_dataset):,} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset):,} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset):,} samples, {len(test_loader)} batches")

    # ========== 计算权重 ==========
    print(f"\n{'=' * 80}")
    print("Computing Loss Weights")
    print(f"{'=' * 80}")

    # Task权重 (如果启用)
    task_weights = None
    if use_task_weighting:
        task_weights = compute_task_weights(
            task_datasets,
            smoothing=weight_smoothing
        ).to(device)

    # Class权重 (如果启用)
    pos_weight = None
    if kwargs.get('use_class_weighting', False):
        # 自动计算pos_weight
        pos_weight = compute_pos_weight(task_datasets)

    # ========== 3. 创建Model ==========
    print(f"\n{'=' * 80}")
    print("Creating Model...")
    print(f"{'=' * 80}")

    model_kwargs = {
        'mode_config': mode_config,
        'n_tasks': len(task_manager.get_all_tasks()),
        'vocab_size': 21,
        'peptide_output_dim': kwargs.get('peptide_dim', 64),
        'task_output_dim': kwargs.get('task_dim', 64),
        'dropout': kwargs.get('dropout', 0.1),
        'use_gnn': kwargs.get('use_gnn', True),  # ← 新增
        'use_film': kwargs.get('use_film', True),  # ← 新增
    }

    # Mode 2需要n_tissues
    if mode_config.mode == TrainingMode.HLA_TISSUE:
        # 从train_loader的dataset获取
        all_tasks = task_manager.get_all_tasks()
        unique_tissues = set(task.tissue for task in all_tasks.values())
        model_kwargs['n_tissues'] = len(unique_tissues)
        model_kwargs['tissue_embed_dim'] = kwargs.get('tissue_dim', 32)

    model = ImmuneAppModel(**model_kwargs).to(device)

    params = model.get_parameters_count()
    print(f"\n✓ Model created:")
    for component, count in params.items():
        print(f"  {component}: {count:,}")

    # ========== 4. 创建Trainer ==========
    print(f"\n{'=' * 80}")
    print("Creating Trainer...")
    print(f"{'=' * 80}")

    if mode_config.use_maml:
        trainer = MAMLTrainer(
            model,
            inner_lr=mode_config.inner_lr,
            meta_lr=mode_config.meta_lr,
            inner_steps=mode_config.inner_steps,
            device=device
        )
        print(f"✓ MAML Trainer")
        print(f"  Inner LR: {mode_config.inner_lr}")
        print(f"  Meta LR: {mode_config.meta_lr}")
        print(f"  Inner steps: {mode_config.inner_steps}")
    else:
        optimizer = optim.Adam(model.parameters(), lr=mode_config.meta_lr)
        trainer = StandardTrainer(
            model,
            optimizer,
            device=device,
            pos_weight=pos_weight  # 传入class权重
        )
        print(f"✓ Standard Trainer")
        print(f"  LR: {mode_config.meta_lr}")

    evaluator = Evaluator(device)

    # ========== 5. 训练循环 ==========
    print(f"\n{'=' * 80}")
    print("Training...")
    print(f"{'=' * 80}")

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auroc': [],            # pooled（旧口径，仍记录，仅供参考/排错）
        'val_auprc': [],
        'val_mean_task_auroc': [],  # D2 修复：现在用这个口径挑选 best_model.pt
    }

    best_val_mean_task_auroc = 0.0

    for epoch in range(n_epochs):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch + 1}/{n_epochs}")
        print(f"{'=' * 80}")

        # Train (传入task_weights)
        if mode_config.use_maml:
            # MAML暂不支持task_weights
            train_metrics = trainer.train_epoch(train_loader, graph_wrapper)
        else:
            # Standard训练使用task_weights
            train_metrics = trainer.train_epoch(
                train_loader,
                graph_wrapper,
                task_weights=task_weights  # 传入权重
            )

        # Validate
        val_metrics = evaluator.evaluate(model, val_loader, graph_wrapper)

        # 记录
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_auroc'].append(val_metrics['auroc'])
        history['val_auprc'].append(val_metrics['auprc'])
        history['val_mean_task_auroc'].append(val_metrics['mean_task_auroc'])

        # 打印
        print(f"\nTrain Loss: {train_metrics['loss']:.4f}")
        if mode_config.use_maml:
            print(f"  Support Loss: {train_metrics.get('support_loss', 0):.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val AUROC (pooled, 仅参考): {val_metrics['auroc']:.4f}")
        print(f"Val Mean-Task-AUROC (D2 修复后的选模型口径): {val_metrics['mean_task_auroc']:.4f}")
        print(f"Val AUPRC: {val_metrics['auprc']:.4f}")
        print(f"Val PPV@2%: {val_metrics['ppv']:.4f}")

        # 保存最佳模型（D2 修复：用 mean_task_auroc 而不是 pooled auroc 来挑选）
        if val_metrics['mean_task_auroc'] > best_val_mean_task_auroc:
            best_val_mean_task_auroc = val_metrics['mean_task_auroc']

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_auroc': val_metrics['auroc'],                        # pooled，仅供参考
                'val_mean_task_auroc': val_metrics['mean_task_auroc'],    # 实际选模型用的口径
                'val_auprc': val_metrics['auprc'],
                'mode_config': mode_config.to_dict(),
                'random_seed': random_seed,
                'selection_metric': 'mean_task_auroc',
                'history': history
            }, output_dir / 'best_model.pt')

            print(f"✓ Saved best model (Mean-Task-AUROC: {best_val_mean_task_auroc:.4f}, "
                  f"pooled AUROC此时为: {val_metrics['auroc']:.4f})")

    # ========== 6. Final Per-Task Evaluation ==========
    print(f"\n{'=' * 80}")
    print("Step 6: Final Per-Task Evaluation")
    print(f"{'=' * 80}")

    # 导入per-task评估器
    from src.training.per_task_evaluator import evaluate_per_task_performance

    # 加载最佳模型
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 评估test set的per-task性能
    # Mode 2 传入 test_task_datasets 以生成 per-tissue 汇总
    # Mode 1 不传（None），跳过 tissue 分析
    _test_datasets_for_tissue = (
        test_task_datasets
        if hasattr(mode_config, 'mode') and str(mode_config.mode) == 'HLA_TISSUE'
        else None
    )

    per_task_df, per_task_summary = evaluate_per_task_performance(
        model=model,
        test_loader=test_loader,
        graph_wrapper=graph_wrapper,
        task_manager=task_manager,
        mode_config=mode_config,
        output_dir=output_dir,
        test_task_datasets=_test_datasets_for_tissue,
        device=device
    )

    # ========== 7. 保存最终结果 ==========
    # 转换numpy类型为Python原生类型
    def convert_to_native(obj):
        """递归转换numpy类型为Python原生类型"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        else:
            return obj

    results = {
        'mode': mode_config.task_type_name,
        'use_maml': mode_config.use_maml,
        'n_epochs': n_epochs,
        'random_seed': random_seed,
        'graph': {
            'hla_graph_weight': float(mode_config.hla_graph_weight),
            'tissue_graph_weight': float(mode_config.tissue_graph_weight),
            'threshold': float(kwargs.get('graph_threshold', 0.3)),
        },
        'selection_metric': 'mean_task_auroc',                                     # D2: 记录实际用的选模型口径
        'best_val_mean_task_auroc': float(best_val_mean_task_auroc),
        'final_val_mean_task_auroc': float(history['val_mean_task_auroc'][-1]),
        'best_val_auroc': float(max(history['val_auroc'])),                        # pooled，仅供参考
        'final_val_auroc': float(history['val_auroc'][-1]),                        # pooled，仅供参考
        'history': convert_to_native(history)
    }

    with open(output_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Training completed!")
    print(f"  Best Val Mean-Task-AUROC (选模型用的口径): {best_val_mean_task_auroc:.4f}")
    print(f"  Best Val AUROC (pooled, 仅供参考): {max(history['val_auroc']):.4f}")
    print(f"  Results saved to: {output_dir}")
    print(f"{'=' * 80}")

    return model, history


if __name__ == "__main__":
    print("Trainer module loaded successfully!")
    print("Use train_model() to train your model")
