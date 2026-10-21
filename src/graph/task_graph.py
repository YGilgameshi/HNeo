"""
Mode-Aware Task Graph Builder

支持Mode 1和Mode 2的Task Graph构建:
- Mode 1: 基于HLA序列相似性
- Mode 2: HLA序列相似性 + Tissue共现
"""

import numpy as np
import networkx as nx
from pathlib import Path
import json
import pickle
from collections import Counter
from tqdm import tqdm


import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.config.mode_config import ModeConfig, TrainingMode
from src.data.task_definition import TaskManager
from src.graph.weighting import combine_graph_similarity


class ModeAwareTaskGraphBuilder:
    """
    Mode-aware Task Graph构建器

    根据mode自动选择相似性计算方法:
    - Mode 1: HLA序列相似性
    - Mode 2: HLA序列相似性 + Tissue共现

    Args:
        task_manager: TaskManager实例
        mode_config: ModeConfig实例
        hla_sequences_file: HLA伪序列文件路径
    """

    def __init__(self,
                 task_manager: TaskManager,
                 mode_config: ModeConfig,
                 hla_sequences_file: str = 'configs/hla_sequences.json',
                 allow_random_hla_fallback: bool = False):

        self.task_manager = task_manager
        self.mode = mode_config.mode
        self.mode_config = mode_config
        self.tissue_graph_weight = float(mode_config.tissue_graph_weight)
        self.hla_graph_weight = float(mode_config.hla_graph_weight)

        # 获取所有tasks
        self.tasks = list(task_manager.get_all_tasks().keys())
        self.task_objects = task_manager.get_all_tasks()
        self.n_tasks = len(self.tasks)

        print(f"\n{'='*80}")
        print(f"Mode-Aware Task Graph Builder - {mode_config.task_type_name}")
        print(f"{'='*80}")
        print(f"  Tasks: {self.n_tasks}")
        if self.mode == TrainingMode.HLA_TISSUE:
            print(
                "  Edge-weight mixture: "
                f"HLA={self.hla_graph_weight:.2f}, "
                f"tissue={self.tissue_graph_weight:.2f}"
            )

        # 加载HLA序列
        self._load_hla_sequences(
            hla_sequences_file,
            allow_random_fallback=allow_random_hla_fallback,
        )

        # Mode 2需要构建tissue映射
        if self.mode == TrainingMode.HLA_TISSUE:
            self._build_tissue_mappings()

    def _load_hla_sequences(self, sequences_file, allow_random_fallback=False):
        """加载HLA伪序列"""
        print(f"\n加载HLA序列: {sequences_file}")

        try:
            with open(sequences_file) as f:
                self.hla_sequences = json.load(f)
            print(f"  ✓ 加载了 {len(self.hla_sequences)} 个HLA序列")
        except (OSError, json.JSONDecodeError) as exc:
            if not allow_random_fallback:
                raise RuntimeError(
                    f"无法读取有效的HLA伪序列文件: {sequences_file}. "
                    "Graph-weight实验禁止使用随机序列fallback。"
                ) from exc
            print(f"  ⚠ 无法加载HLA序列文件,显式启用随机序列调试fallback")
            # 生成随机序列作为fallback
            unique_hlas = set(task.hla for task in self.task_objects.values())
            self.hla_sequences = {}
            amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
            for hla in unique_hlas:
                seq = ''.join(np.random.choice(list(amino_acids), 34))
                self.hla_sequences[hla] = seq

        required_hlas = {task.hla for task in self.task_objects.values()}
        missing_hlas = sorted(required_hlas - set(self.hla_sequences))
        if missing_hlas:
            preview = ', '.join(missing_hlas[:10])
            raise ValueError(
                f"HLA伪序列文件缺少 {len(missing_hlas)} 个训练alleles: {preview}"
            )

        invalid_hlas = [
            hla for hla in required_hlas
            if not isinstance(self.hla_sequences[hla], str)
            or not self.hla_sequences[hla]
        ]
        if invalid_hlas:
            raise ValueError(
                f"HLA序列文件存在空值或非字符串: {invalid_hlas[:10]}"
            )

        length_counts = Counter(len(self.hla_sequences[hla]) for hla in required_hlas)
        self.hla_sequence_length_counts = dict(sorted(length_counts.items()))
        if len(length_counts) > 1:
            print(
                "  ⚠ HLA序列长度不一致；为复现原实验，不等长allele pair的"
                "HLA similarity将设为0"
            )
            print(f"  长度分布: {self.hla_sequence_length_counts}")

    def _build_tissue_mappings(self):
        """构建tissue映射 (Mode 2)"""
        print(f"\n构建Tissue映射...")

        unique_tissues = set(task.tissue for task in self.task_objects.values())
        self.tissues = sorted(unique_tissues)
        self.tissue_to_idx = {t: i for i, t in enumerate(self.tissues)}

        print(f"  ✓ {len(self.tissues)} 个tissues: {self.tissues}")

    def compute_hla_similarity(self, hla1, hla2):
        """
        计算HLA序列相似性

        保留原实验实现：等长序列使用逐位identity；不等长序列返回0。
        """
        seq1 = self.hla_sequences.get(hla1, '')
        seq2 = self.hla_sequences.get(hla2, '')

        if not seq1 or not seq2 or len(seq1) != len(seq2):
            return 0.0

        # 简单的序列一致性
        matches = sum(a == b for a, b in zip(seq1, seq2))
        return matches / len(seq1)

    def compute_tissue_cooccurrence(self, task1, task2):
        """
        计算tissue共现相似性 (Mode 2)

        如果两个task在同一tissue,返回1;否则返回0
        """
        if task1.tissue == task2.tissue:
            return 1.0
        else:
            return 0.0

    def build_similarity_matrix(self):
        """
        构建任务相似性矩阵

        Returns:
            np.ndarray: (n_tasks, n_tasks)
        """
        print(f"\n构建相似性矩阵...")

        similarity_matrix = np.zeros((self.n_tasks, self.n_tasks))

        for i in tqdm(range(self.n_tasks), desc="计算相似性"):
            task_i = self.task_objects[self.tasks[i]]

            for j in range(i, self.n_tasks):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    task_j = self.task_objects[self.tasks[j]]

                    if self.mode == TrainingMode.HLA_ONLY:
                        # Mode 1: 只用HLA序列相似性
                        sim = self.compute_hla_similarity(task_i.hla, task_j.hla)

                    elif self.mode == TrainingMode.HLA_TISSUE:
                        # Mode 2: HLA相似性 + Tissue共现
                        hla_sim = self.compute_hla_similarity(task_i.hla, task_j.hla)
                        tissue_sim = self.compute_tissue_cooccurrence(task_i, task_j)

                        # 权重由ModeConfig控制；两项权重之和固定为1。
                        sim = combine_graph_similarity(
                            hla_sim,
                            tissue_sim,
                            self.tissue_graph_weight,
                        )

                    else:
                        # Mode 3: 待定
                        sim = 0.0

                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim

        print(f"  ✓ 完成")
        print(f"  矩阵形状: {similarity_matrix.shape}")

        if self.n_tasks > 1:
            mask = ~np.eye(self.n_tasks, dtype=bool)
            print(f"  平均相似性: {similarity_matrix[mask].mean():.4f}")
            print(f"  最大相似性: {similarity_matrix[mask].max():.4f}")
            print(f"  最小相似性: {similarity_matrix[mask].min():.4f}")

        return similarity_matrix

    def build_graph(self, similarity_matrix, threshold=0.3):
        """
        构建Task Graph

        Args:
            similarity_matrix: 相似性矩阵
            threshold: 边的相似性阈值

        Returns:
            networkx.Graph
        """
        print(f"\n构建Task Graph (threshold={threshold})...")

        G = nx.Graph()

        # 添加节点
        for i, task_id in enumerate(self.tasks):
            task = self.task_objects[task_id]

            node_attrs = {
                'task_id': task_id,
                'hla': task.hla,
                'n_samples': task.n_samples
            }

            # Mode 2添加tissue信息
            if self.mode == TrainingMode.HLA_TISSUE:
                node_attrs['tissue'] = task.tissue

            G.add_node(i, **node_attrs)

        # 添加边
        n_edges = 0
        for i in range(self.n_tasks):
            for j in range(i + 1, self.n_tasks):
                sim = similarity_matrix[i, j]
                if sim >= threshold:
                    G.add_edge(i, j, weight=sim)
                    n_edges += 1

        print(f"  ✓ 节点: {G.number_of_nodes()}")
        print(f"  ✓ 边: {G.number_of_edges()}")

        if G.number_of_nodes() > 0:
            avg_degree = 2 * G.number_of_edges() / G.number_of_nodes()
            print(f"  ✓ 平均度: {avg_degree:.2f}")

        # 检查连通性
        if nx.is_connected(G):
            print(f"  ✓ 图是连通的")
        else:
            components = list(nx.connected_components(G))
            print(f"  ⚠ 图不连通,有 {len(components)} 个连通分量")
            print(f"    最大分量大小: {len(max(components, key=len))}")

        return G

    def save(self, graph, similarity_matrix, output_dir, threshold=None):
        """
        保存Task Graph和相关数据

        Args:
            graph: networkx.Graph
            similarity_matrix: 相似性矩阵
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n保存Task Graph到: {output_dir}")

        # 1. 保存graph
        with open(output_dir / 'task_graph.pkl', 'wb') as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 2. 保存相似性矩阵
        np.save(output_dir / 'similarity_matrix.npy', similarity_matrix)
        print(f"  ✓ similarity_matrix.npy")

        # 3. 保存task映射
        task_mapping = {
            'mode': self.mode.value,
            'tasks': self.tasks,
            'task_to_idx': {task_id: i for i, task_id in enumerate(self.tasks)},
            'idx_to_task': {i: task_id for i, task_id in enumerate(self.tasks)},
            'n_tasks': self.n_tasks,
            'hla_graph_weight': self.hla_graph_weight,
            'tissue_graph_weight': self.tissue_graph_weight,
        }

        # Mode 2添加tissue映射
        if self.mode == TrainingMode.HLA_TISSUE:
            task_mapping['tissues'] = self.tissues
            task_mapping['tissue_to_idx'] = self.tissue_to_idx

        with open(output_dir / 'task_mapping.json', 'w') as f:
            json.dump(task_mapping, f, indent=2)
        print(f"  ✓ task_mapping.json")

        # 4. 保存mode_config
        with open(output_dir / 'mode_config.json', 'w') as f:
            json.dump(self.mode_config.to_dict(), f, indent=2)
        print(f"  ✓ mode_config.json")

        # 5. 保存本次graph的可比较统计，供weight-sweep自动汇总。
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        possible_edges = n_nodes * (n_nodes - 1) / 2
        edge_weights = [data['weight'] for _, _, data in graph.edges(data=True)]
        components = (
            list(nx.connected_components(graph)) if n_nodes > 0 else []
        )
        graph_statistics = {
            'mode': self.mode.value,
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'density': float(n_edges / possible_edges) if possible_edges else 0.0,
            'average_degree': float(2 * n_edges / n_nodes) if n_nodes else 0.0,
            'n_connected_components': len(components),
            'largest_component_size': (
                len(max(components, key=len)) if components else 0
            ),
            'mean_edge_weight': float(np.mean(edge_weights)) if edge_weights else 0.0,
            'min_edge_weight': float(np.min(edge_weights)) if edge_weights else 0.0,
            'max_edge_weight': float(np.max(edge_weights)) if edge_weights else 0.0,
            'graph_threshold': None if threshold is None else float(threshold),
            'hla_graph_weight': self.hla_graph_weight,
            'tissue_graph_weight': self.tissue_graph_weight,
            'hla_sequence_length_counts': {
                str(k): v for k, v in self.hla_sequence_length_counts.items()
            },
            'hla_similarity_unequal_length_policy': 'zero',
            'formula': (
                '(1 - tissue_graph_weight) * sim_HLA + '
                'tissue_graph_weight * I_same_tissue'
            ),
        }
        with open(output_dir / 'graph_statistics.json', 'w') as f:
            json.dump(graph_statistics, f, indent=2)
        print(f"  ✓ graph_statistics.json")

    def build_and_save(self, output_dir, threshold=0.3):
        """
        一键构建并保存

        Args:
            output_dir: 输出目录
            threshold: 边的相似性阈值

        Returns:
            graph, similarity_matrix
        """
        print(f"\n{'='*80}")
        print(f"开始构建Task Graph")
        print(f"{'='*80}")

        # 1. 构建相似性矩阵
        similarity_matrix = self.build_similarity_matrix()

        # 2. 构建graph
        graph = self.build_graph(similarity_matrix, threshold)

        # 3. 保存
        self.save(graph, similarity_matrix, output_dir, threshold=threshold)

        print(f"\n{'='*80}")
        print(f"Task Graph构建完成!")
        print(f"{'='*80}")

        return graph, similarity_matrix


class TaskGraphWrapper:
    """
    Task Graph包装器 (用于训练时)

    加载已保存的Task Graph并提供便捷接口
    """

    def __init__(self, graph_dir):
        """
        Args:
            graph_dir: Task Graph保存目录
        """
        graph_dir = Path(graph_dir)

        # 加载graph
        self.graph = nx.read_gpickle(graph_dir / 'task_graph.pkl')

        # 加载相似性矩阵
        self.similarity_matrix = np.load(graph_dir / 'similarity_matrix.npy')

        # 加载task映射
        with open(graph_dir / 'task_mapping.json') as f:
            mapping = json.load(f)

        self.mode = mapping['mode']
        self.tasks = mapping['tasks']
        self.task_to_idx = mapping['task_to_idx']
        self.idx_to_task = {int(k): v for k, v in mapping['idx_to_task'].items()}
        self.n_tasks = mapping['n_tasks']

        # Mode 2的tissue映射
        if 'tissues' in mapping:
            self.tissues = mapping['tissues']
            self.tissue_to_idx = mapping['tissue_to_idx']

        # PyTorch需要的edge数据
        self.edge_index = self._get_edge_index()
        self.edge_weight = self._get_edge_weight()

        print(f"✓ TaskGraphWrapper loaded: {self.n_tasks} tasks, {len(self.edge_index[0])} edges")

    def _get_edge_index(self):
        """获取edge_index (PyTorch格式)"""
        import torch
        edges = list(self.graph.edges())
        if not edges:
            return torch.zeros((2, 0), dtype=torch.long)
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        return edge_index

    def _get_edge_weight(self):
        """获取edge_weight"""
        import torch
        edges = list(self.graph.edges(data='weight'))
        if not edges:
            return torch.zeros(0)
        weights = [w for _, _, w in edges]
        return torch.tensor(weights, dtype=torch.float)

    def to(self, device):
        """移动到指定device"""
        import torch
        self.edge_index = self.edge_index.to(device)
        self.edge_weight = self.edge_weight.to(device)
        return self


if __name__ == "__main__":
    print("="*80)
    print("Testing ModeAwareTaskGraphBuilder")
    print("="*80)

    from src.config.mode_config import create_mode1_config, create_mode2_config
    from src.data.task_definition import Task, TaskManager
    import pandas as pd

    # Mode 1测试
    print("\n" + "="*80)
    print("Testing Mode 1")
    print("="*80)

    config1 = create_mode1_config()
    manager1 = TaskManager(mode='hla_only')

    for i, hla in enumerate(['HLA-A*02:01', 'HLA-A*03:01', 'HLA-B*07:02']):
        task = Task.create_mode1_task(hla, pd.DataFrame({'label': [1]*20}))
        manager1.add_task(task)

    builder1 = ModeAwareTaskGraphBuilder(
        manager1,
        config1,
        hla_sequences_file='configs/hla_sequences.json'
    )

    graph1, sim1 = builder1.build_and_save('test_output/mode1_graph', threshold=0.2)

    # Mode 2测试
    print("\n" + "="*80)
    print("Testing Mode 2")
    print("="*80)

    config2 = create_mode2_config()
    manager2 = TaskManager(mode='hla_tissue')

    for hla in ['HLA-A*02:01', 'HLA-B*07:02']:
        for tissue in ['Liver', 'Lung']:
            task = Task.create_mode2_task(hla, tissue, pd.DataFrame({'label': [1]*15}))
            manager2.add_task(task)

    builder2 = ModeAwareTaskGraphBuilder(
        manager2,
        config2,
        hla_sequences_file='configs/hla_sequences.json'
    )

    graph2, sim2 = builder2.build_and_save('test_output/mode2_graph', threshold=0.2)

    # 测试Wrapper
    print("\n" + "="*80)
    print("Testing TaskGraphWrapper")
    print("="*80)

    wrapper = TaskGraphWrapper('test_output/mode1_graph')
    print(f"✓ Loaded: {wrapper.n_tasks} tasks")
    print(f"✓ Edge index shape: {wrapper.edge_index.shape}")
    print(f"✓ Edge weight shape: {wrapper.edge_weight.shape}")

    print("\n✓ All tests passed!")
