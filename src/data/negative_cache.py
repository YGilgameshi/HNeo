"""
负样本数据缓存管理器

功能:
1. 保存生成的负样本到磁盘
2. 从磁盘加载已有的负样本
3. 避免重复生成,节省时间
"""

import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime


class NegativeSampleCache:
    """负样本缓存管理器"""

    def __init__(self, cache_dir: str = 'data/negative_samples'):
        """
        初始化缓存管理器

        Args:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._fingerprint_cache = {}

    @staticmethod
    def _update_hash_from_file(hasher, file_path: Path):
        """流式读取文件，避免对大型split一次性占用内存。"""
        hasher.update(file_path.name.encode())
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                hasher.update(chunk)

    def _get_data_fingerprint(self, data_file: str) -> str:
        """为文件或train/val/test目录生成稳定的内容指纹。"""
        path = Path(data_file)
        cache_id = str(path.resolve())
        if cache_id in self._fingerprint_cache:
            return self._fingerprint_cache[cache_id]

        hasher = hashlib.sha256()
        if path.is_dir():
            split_files = [path / name for name in ('train.tsv', 'val.tsv', 'test.tsv')]
            split_files = [p for p in split_files if p.is_file()]
            if not split_files:
                raise FileNotFoundError(
                    f"缓存数据目录中未找到train.tsv/val.tsv/test.tsv: {path}"
                )
            for split_file in split_files:
                self._update_hash_from_file(hasher, split_file)
        elif path.is_file():
            self._update_hash_from_file(hasher, path)
        else:
            raise FileNotFoundError(f"缓存数据来源不存在: {path}")

        fingerprint = hasher.hexdigest()[:16]
        self._fingerprint_cache[cache_id] = fingerprint
        return fingerprint

    def _get_cache_key(self, data_file: str, mode: str, config: dict) -> str:
        """
        生成缓存key

        基于:
        - 数据文件名
        - 训练模式
        - 关键配置参数
        """
        # 提取关键配置
        key_config = {
            'mode': mode,
            'negative_ratio': config.get('negative_ratio', 20),
            'use_tissue_aware': config.get('use_tissue_aware_negatives', False),
            'min_samples': config.get('min_samples', 10)
        }

        # 组合为字符串
        data_fingerprint = self._get_data_fingerprint(data_file)
        key_str = f"{data_fingerprint}_{json.dumps(key_config, sort_keys=True)}"

        # 生成hash (避免太长)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()[:8]

        source_name = Path(data_file).name or Path(data_file).stem
        return f"{source_name}_{mode}_{key_hash}"

    def _get_cache_paths(self, cache_key: str) -> dict:
        """获取缓存文件路径"""
        return {
            'train': self.cache_dir / f"{cache_key}_train.pkl",
            'val': self.cache_dir / f"{cache_key}_val.pkl",
            'test': self.cache_dir / f"{cache_key}_test.pkl",
            'metadata': self.cache_dir / f"{cache_key}_metadata.json"
        }

    def exists(self, data_file: str, mode: str, config: dict) -> bool:
        """
        检查缓存是否存在

        Args:
            data_file: 数据文件路径
            mode: 训练模式 ('mode1' or 'mode2')
            config: 配置字典

        Returns:
            是否存在完整的缓存
        """
        cache_key = self._get_cache_key(data_file, mode, config)
        paths = self._get_cache_paths(cache_key)

        # 检查所有文件是否存在
        return all(p.exists() for p in paths.values())

    def save(
        self,
        data_file: str,
        mode: str,
        config: dict,
        train_datasets: Dict,
        val_datasets: Dict,
        test_datasets: Dict
    ):
        """
        保存负样本数据集

        Args:
            data_file: 数据文件路径
            mode: 训练模式
            config: 配置字典
            train_datasets: 训练集 {task_id: DataFrame}
            val_datasets: 验证集
            test_datasets: 测试集
        """
        cache_key = self._get_cache_key(data_file, mode, config)
        paths = self._get_cache_paths(cache_key)

        print(f"\n💾 Saving negative samples to cache...")
        print(f"  Cache key: {cache_key}")
        print(f"  Location: {self.cache_dir}")

        # 保存数据集
        with open(paths['train'], 'wb') as f:
            pickle.dump(train_datasets, f)
        print(f"  ✓ Train: {len(train_datasets)} tasks")

        with open(paths['val'], 'wb') as f:
            pickle.dump(val_datasets, f)
        print(f"  ✓ Val: {len(val_datasets)} tasks")

        with open(paths['test'], 'wb') as f:
            pickle.dump(test_datasets, f)
        print(f"  ✓ Test: {len(test_datasets)} tasks")

        # 保存元数据
        metadata = {
            'data_file': str(data_file),
            'data_fingerprint': self._get_data_fingerprint(data_file),
            'mode': mode,
            'config': config,
            'created_at': datetime.now().isoformat(),
            'n_train_tasks': len(train_datasets),
            'n_val_tasks': len(val_datasets),
            'n_test_tasks': len(test_datasets),
            'total_train_samples': sum(len(df) for df in train_datasets.values()),
            'total_val_samples': sum(len(df) for df in val_datasets.values()),
            'total_test_samples': sum(len(df) for df in test_datasets.values()),
        }

        with open(paths['metadata'], 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  ✓ Metadata saved")
        print(f"\n✓ Cache saved successfully!")

    def load(
        self,
        data_file: str,
        mode: str,
        config: dict
    ) -> Optional[tuple]:
        """
        加载缓存的负样本

        Args:
            data_file: 数据文件路径
            mode: 训练模式
            config: 配置字典

        Returns:
            (train_datasets, val_datasets, test_datasets) 或 None
        """
        if not self.exists(data_file, mode, config):
            return None

        cache_key = self._get_cache_key(data_file, mode, config)
        paths = self._get_cache_paths(cache_key)

        print(f"\n📂 Loading negative samples from cache...")
        print(f"  Cache key: {cache_key}")

        # 加载元数据
        with open(paths['metadata'], 'r') as f:
            metadata = json.load(f)

        print(f"  Created: {metadata['created_at']}")
        print(f"  Train tasks: {metadata['n_train_tasks']}")
        print(f"  Val tasks: {metadata['n_val_tasks']}")
        print(f"  Test tasks: {metadata['n_test_tasks']}")

        # 加载数据集
        with open(paths['train'], 'rb') as f:
            train_datasets = pickle.load(f)
        print(f"  ✓ Train loaded: {len(train_datasets)} tasks")

        with open(paths['val'], 'rb') as f:
            val_datasets = pickle.load(f)
        print(f"  ✓ Val loaded: {len(val_datasets)} tasks")

        with open(paths['test'], 'rb') as f:
            test_datasets = pickle.load(f)
        print(f"  ✓ Test loaded: {len(test_datasets)} tasks")

        print(f"\n✓ Cache loaded successfully!")

        return train_datasets, val_datasets, test_datasets

    def list_caches(self) -> List[dict]:
        """列出所有可用的缓存"""
        metadata_files = list(self.cache_dir.glob("*_metadata.json"))

        caches = []
        for meta_file in metadata_files:
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            caches.append(metadata)

        return caches

    def clear(self, data_file: Optional[str] = None, mode: Optional[str] = None):
        """
        清除缓存

        Args:
            data_file: 如果指定,只清除该文件的缓存
            mode: 如果指定,只清除该模式的缓存
        """
        if data_file is None and mode is None:
            # 清除所有缓存
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ All caches cleared from {self.cache_dir}")
        else:
            # 清除特定缓存
            pattern = "*"
            if data_file:
                pattern = f"{Path(data_file).stem}_*"
            if mode:
                pattern = f"*_{mode}_*"

            files = list(self.cache_dir.glob(pattern))
            for f in files:
                f.unlink()
            print(f"✓ {len(files)} cache files removed")


if __name__ == "__main__":
    # 测试缓存管理器
    print("="*80)
    print("Negative Sample Cache Manager Test")
    print("="*80)

    cache = NegativeSampleCache()

    # 测试配置
    test_config = {
        'negative_ratio': 20,
        'use_tissue_aware_negatives': False,
        'min_samples': 10
    }

    print(f"\nCache directory: {cache.cache_dir}")
    print(f"Exists: {cache.exists('data/test.tsv', 'mode2', test_config)}")

    # 列出所有缓存
    caches = cache.list_caches()
    print(f"\nAvailable caches: {len(caches)}")
    for c in caches:
        print(f"  - {c['data_file']} ({c['mode']}): {c['n_train_tasks']} tasks")

    print("\n✓ Cache manager test completed!")
