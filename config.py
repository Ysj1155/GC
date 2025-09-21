from dataclasses import dataclass

@dataclass
class SimConfig:
    num_blocks: int = 256
    pages_per_block: int = 64
    gc_free_block_threshold: float = 0.02  # ✅ 조금 더 낮춤
    rng_seed: int = 42
    user_capacity_ratio: float = 0.9      # ✅ 논리 용량(=유저에게 보이는 용량) 비율

    @property
    def total_pages(self) -> int:
        return self.num_blocks * self.pages_per_block

    @property
    def user_total_pages(self) -> int:
        return int(self.total_pages * self.user_capacity_ratio)