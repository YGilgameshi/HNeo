"""
运行模式配置

定义ImmuneApp的三种运行模式:
- Mode 1 (HLA_ONLY): 只基于HLA分型的任务定义,纯binding prediction
- Mode 2 (HLA_TISSUE): HLA×Tissue组合任务,tissue-specific presentation
- Mode 3 (HYBRID): 混合模式,联合训练或动态切换 (接口预留)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class TrainingMode(Enum):
    """训练模式枚举"""
    HLA_ONLY = "hla_only"           # Mode 1: 仅HLA
    HLA_TISSUE = "hla_tissue"       # Mode 2: HLA×Tissue
    HYBRID = "hybrid"               # Mode 3: 混合模式(未来实现)

    @classmethod
    def from_string(cls, mode_str: str):
        """从字符串创建模式"""
        mode_map = {
            'mode1': cls.HLA_ONLY,
            'hla_only': cls.HLA_ONLY,
            'hla': cls.HLA_ONLY,
            'mode2': cls.HLA_TISSUE,
            'hla_tissue': cls.HLA_TISSUE,
            'tissue': cls.HLA_TISSUE,
            'mode3': cls.HYBRID,
            'hybrid': cls.HYBRID,
        }
        return mode_map.get(mode_str.lower(), cls.HLA_ONLY)


@dataclass
class ModeConfig:
    """
    模式配置数据类

    不同模式的核心差异:
    - Task定义方式
    - 负样本生成策略
    - Task embedding构建
    - Graph结构
    """
    mode: TrainingMode

    # Task定义相关
    min_samples: int = 20                # 任务最小样本数
    exclude_unknown_tissue: bool = True  # Mode 2专用:排除Unknown tissue

    # 负样本生成相关
    # 注: 原 use_cross_hla_negatives / use_tissue_aware_negatives 已删除。
    # 现统一使用来源蛋白配对切割策略 (EnhancedNegativeSampler)。
    negative_ratio: int = 20             # 负样本比例 (1:20)

    # Tissue信息相关 (Mode 2)
    tissue_source: Optional[str] = 'Host'  # 组织来源列名
    use_tissue_expression: bool = False     # 是否使用表达谱数据

    # Task graph相关 (Mode 2)
    # w_ij = (1 - tissue_graph_weight) * sim_HLA +
    #        tissue_graph_weight * I[same tissue]
    tissue_graph_weight: float = 0.3

    # 训练方法相关
    use_maml: bool = True                   # 是否使用MAML (True=MAML, False=Standard)
    inner_lr: float = 0.01                  # MAML内循环学习率
    inner_steps: int = 5                    # MAML内循环步数
    meta_lr: float = 0.001                  # Meta/Standard学习率

    # Hybrid模式相关 (Mode 3 - 接口预留)
    hybrid_strategy: Optional[str] = None   # 'joint', 'sequential', 'dynamic'
    mode1_weight: float = 0.5              # Mode 1损失权重
    mode2_weight: float = 0.5             # Mode 2损失权重

    def __post_init__(self):
        """验证配置合理性"""
        if not 0.0 <= self.tissue_graph_weight <= 1.0:
            raise ValueError(
                "tissue_graph_weight must be between 0 and 1, "
                f"got {self.tissue_graph_weight}"
            )

        if self.mode == TrainingMode.HLA_ONLY:
            # Mode 1不使用tissue信息
            self.use_tissue_expression = False

        elif self.mode == TrainingMode.HLA_TISSUE:
            # Mode 2必须有tissue信息
            if self.tissue_source is None:
                raise ValueError("Mode 2 requires tissue_source")

        elif self.mode == TrainingMode.HYBRID:
            # Mode 3需要指定策略
            if self.hybrid_strategy is None:
                raise ValueError("Mode 3 requires hybrid_strategy")

    @property
    def task_type_name(self) -> str:
        """返回任务类型名称"""
        return {
            TrainingMode.HLA_ONLY: 'HLA-only',
            TrainingMode.HLA_TISSUE: 'HLA×Tissue',
            TrainingMode.HYBRID: 'Hybrid'
        }[self.mode]

    @property
    def hla_graph_weight(self) -> float:
        """HLA相似性权重；与tissue权重之和固定为1。"""
        return 1.0 - self.tissue_graph_weight

    def to_dict(self) -> dict:
        """转为字典（存入checkpoint）"""
        return {
            'mode': self.mode.value,
            'task_type': self.task_type_name,
            'min_samples': int(self.min_samples),
            'negative_ratio': int(self.negative_ratio),
            'tissue_source': self.tissue_source,
            'use_tissue_expression': bool(self.use_tissue_expression),
            'tissue_graph_weight': float(self.tissue_graph_weight),
            'hla_graph_weight': float(self.hla_graph_weight),
            'use_maml': bool(self.use_maml),
            'inner_lr': float(self.inner_lr),
            'inner_steps': int(self.inner_steps),
            'meta_lr': float(self.meta_lr),
        }


def create_mode1_config(**kwargs) -> ModeConfig:
    """创建Mode 1配置"""
    return ModeConfig(
        mode=TrainingMode.HLA_ONLY,
        min_samples=kwargs.get('min_samples', 20),
        negative_ratio=kwargs.get('negative_ratio', 20),
        use_maml=kwargs.get('use_maml', True),
        inner_lr=kwargs.get('inner_lr', 0.01),
        inner_steps=kwargs.get('inner_steps', 5),
        meta_lr=kwargs.get('meta_lr', 0.001),
        **{k: v for k, v in kwargs.items()
           if k not in ['min_samples', 'negative_ratio', 'use_maml',
                        'inner_lr', 'inner_steps', 'meta_lr']}
    )


def create_mode2_config(**kwargs) -> ModeConfig:
    """创建Mode 2配置"""
    return ModeConfig(
        mode=TrainingMode.HLA_TISSUE,
        min_samples=kwargs.get('min_samples', 10),
        negative_ratio=kwargs.get('negative_ratio', 20),
        tissue_source=kwargs.get('tissue_source', 'Host'),
        exclude_unknown_tissue=kwargs.get('exclude_unknown_tissue', True),
        tissue_graph_weight=kwargs.get('tissue_graph_weight', 0.3),
        use_maml=kwargs.get('use_maml', True),
        inner_lr=kwargs.get('inner_lr', 0.01),
        inner_steps=kwargs.get('inner_steps', 5),
        meta_lr=kwargs.get('meta_lr', 0.001),
        **{k: v for k, v in kwargs.items()
           if k not in ['min_samples', 'negative_ratio', 'tissue_source',
                        'exclude_unknown_tissue', 'tissue_graph_weight', 'use_maml',
                        'inner_lr', 'inner_steps', 'meta_lr']}
    )


def create_mode3_config(**kwargs) -> ModeConfig:
    """创建Mode 3配置 (接口预留)"""
    return ModeConfig(
        mode=TrainingMode.HYBRID,
        hybrid_strategy=kwargs.get('hybrid_strategy', 'joint'),
        mode1_weight=kwargs.get('mode1_weight', 0.5),
        mode2_weight=kwargs.get('mode2_weight', 0.5),
        **{k: v for k, v in kwargs.items()
           if k not in ['hybrid_strategy', 'mode1_weight', 'mode2_weight']}
    )


# 预设配置
PRESET_CONFIGS = {
    'mode1_default': ModeConfig(
        mode=TrainingMode.HLA_ONLY,
        min_samples=20,
        negative_ratio=20,
    ),
    'mode2_default': ModeConfig(
        mode=TrainingMode.HLA_TISSUE,
        min_samples=10,
        negative_ratio=20,
        tissue_source='Host',
        exclude_unknown_tissue=True,
    ),
    'mode2_with_expression': ModeConfig(
        mode=TrainingMode.HLA_TISSUE,
        min_samples=10,
        negative_ratio=20,
        tissue_source='Host',
        use_tissue_expression=True,
        exclude_unknown_tissue=True,
    ),
}


if __name__ == "__main__":
    print("=" * 80)
    print("Mode Configuration Test")
    print("=" * 80)

    config1 = create_mode1_config()
    print(f"\n✓ Mode 1 Config:")
    print(f"  Task Type: {config1.task_type_name}")
    print(f"  Min Samples: {config1.min_samples}")
    print(f"  Negative Ratio: 1:{config1.negative_ratio}")

    config2 = create_mode2_config(tissue_source='Host')
    print(f"\n✓ Mode 2 Config:")
    print(f"  Task Type: {config2.task_type_name}")
    print(f"  Min Samples: {config2.min_samples}")
    print(f"  Tissue Source: {config2.tissue_source}")

    print(f"\n✓ Mode 3 Config (Interface Reserved):")
    print(f"  Status: To be implemented")
