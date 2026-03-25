from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List

import yaml
from torch.utils.data import Sampler


class Stage2V3DistributedSampler(Sampler[int]):
    def __init__(
        self,
        dataset,
        recipe_path: str,
        split: str,
        seed: int = 42,
        num_replicas: int = 1,
        rank: int = 0,
    ):
        self.dataset = dataset
        self.recipe_path = recipe_path
        self.split = split
        self.seed = seed
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.recipe = yaml.safe_load(open(recipe_path, "r", encoding="utf-8"))
        self.virtual_epoch_samples = int(self.recipe["sampling"]["virtual_epoch_samples"])
        self.num_samples = (self.virtual_epoch_samples + num_replicas - 1) // num_replicas
        self.total_size = self.num_samples * num_replicas
        self.top_level_probs = self.recipe["sampling"]["top_level"]
        self.age_probs = self.recipe["sampling"]["age"]
        self.emotion_probs = self.recipe["sampling"]["emotion"]
        self.pool_to_indices = dataset.pool_to_indices
        self.subpool_to_indices = dataset.subpool_to_indices

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _weighted_choice(self, rng: random.Random, weights: Dict[str, float]) -> str:
        keys = list(weights.keys())
        values = [float(weights[k]) for k in keys]
        total = sum(values)
        if total <= 0:
            raise ValueError("Weights must sum to a positive value.")
        pick = rng.random() * total
        cursor = 0.0
        for key, value in zip(keys, values):
            cursor += value
            if pick <= cursor:
                return key
        return keys[-1]

    def _fallback_index(self, rng: random.Random) -> int:
        replay = self.pool_to_indices.get("replay_general", [])
        if replay:
            return rng.choice(replay)
        return rng.randrange(len(self.dataset))

    def _sample_for_pool(self, pool_name: str, rng: random.Random) -> int:
        if pool_name == "age_main":
            age_key = self._weighted_choice(rng, self.age_probs)
            indices = self.subpool_to_indices.get(f"age:{age_key}", [])
            if indices:
                return rng.choice(indices)
        elif pool_name == "emotion_gt":
            emotion_key = self._weighted_choice(rng, self.emotion_probs)
            indices = self.subpool_to_indices.get(f"emotion:{emotion_key}", [])
            if indices:
                return rng.choice(indices)

        indices = self.pool_to_indices.get(pool_name, [])
        if indices:
            return rng.choice(indices)
        return self._fallback_index(rng)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        global_indices: List[int] = []
        while len(global_indices) < self.total_size:
            pool_name = self._weighted_choice(rng, self.top_level_probs)
            global_indices.append(self._sample_for_pool(pool_name, rng))
        shard = global_indices[self.rank : self.total_size : self.num_replicas]
        return iter(shard[: self.num_samples])


def build_stage2_v3_sampler(dataset, recipe_path: str, split: str, seed: int, num_replicas: int, rank: int):
    return Stage2V3DistributedSampler(
        dataset=dataset,
        recipe_path=recipe_path,
        split=split,
        seed=seed,
        num_replicas=num_replicas,
        rank=rank,
    )
