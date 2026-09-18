from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataSample:
    input: str
    output: str
    quality_score: float
    source: str
    hash: str


@dataclass
class DatasetStats:
    total_samples: int
    avg_quality: float
    duplicates_removed: int
    low_quality_removed: int
    categories: dict[str, int]


class DatasetEngine:
    def __init__(self):
        self.samples: list[DataSample] = []
        self.quality_threshold = 0.5
    
    def load_data(self, data_path: Path) -> None:
        with data_path.open('r') as f:
            data = json.load(f)
        
        for item in data:
            if "input" in item and "output" in item:
                sample_hash = hashlib.sha256(f"{item['input']}{item['output']}".encode()).hexdigest()
                self.samples.append(DataSample(
                    input=item["input"],
                    output=item["output"],
                    quality_score=0.5,
                    source=data_path.name,
                    hash=sample_hash
                ))
    
    def calculate_quality_scores(self) -> None:
        for sample in self.samples:
            score = 0.0
            
            input_len = len(sample.input)
            output_len = len(sample.output)
            
            if 10 <= input_len <= 1000:
                score += 0.3
            if 10 <= output_len <= 2000:
                score += 0.3
            
            if not sample.input.isspace() and not sample.output.isspace():
                score += 0.2
            
            if sample.input.strip() and sample.output.strip():
                score += 0.2
            
            sample.quality_score = min(score, 1.0)
    
    def remove_duplicates(self) -> int:
        seen_hashes = set()
        unique_samples = []
        
        for sample in self.samples:
            if sample.hash not in seen_hashes:
                seen_hashes.add(sample.hash)
                unique_samples.append(sample)
        
        removed = len(self.samples) - len(unique_samples)
        self.samples = unique_samples
        return removed
    
    def filter_by_quality(self, threshold: float | None = None) -> int:
        threshold = threshold or self.quality_threshold
        original_count = len(self.samples)
        self.samples = [s for s in self.samples if s.quality_score >= threshold]
        return original_count - len(self.samples)
    
    def balance_dataset(self) -> dict[str, int]:
        categories = {}
        for sample in self.samples:
            category = self._categorize_sample(sample)
            categories[category] = categories.get(category, 0) + 1
        
        target_size = min(categories.values()) if categories else 0
        balanced_samples = []
        
        category_counts = {cat: 0 for cat in categories}
        
        for sample in self.samples:
            category = self._categorize_sample(sample)
            if category_counts[category] < target_size * 2:
                balanced_samples.append(sample)
                category_counts[category] += 1
        
        self.samples = balanced_samples
        return category_counts
    
    def _categorize_sample(self, sample: DataSample) -> str:
        input_lower = sample.input.lower()
        
        if "code" in input_lower or "function" in input_lower or "class" in input_lower:
            return "coding"
        elif "math" in input_lower or "calculate" in input_lower or "solve" in input_lower:
            return "math"
        elif "write" in input_lower or "story" in input_lower or "text" in input_lower:
            return "writing"
        elif "explain" in input_lower or "what is" in input_lower or "describe" in input_lower:
            return "knowledge"
        else:
            return "general"
    
    def get_statistics(self) -> DatasetStats:
        if not self.samples:
            return DatasetStats(0, 0.0, 0, 0, {})
        
        avg_quality = sum(s.quality_score for s in self.samples) / len(self.samples)
        categories = {}
        
        for sample in self.samples:
            category = self._categorize_sample(sample)
            categories[category] = categories.get(category, 0) + 1
        
        return DatasetStats(
            total_samples=len(self.samples),
            avg_quality=avg_quality,
            duplicates_removed=0,
            low_quality_removed=0,
            categories=categories
        )
    
    def create_train_val_split(self, val_ratio: float = 0.1) -> tuple[list[DataSample], list[DataSample]]:
        sorted_samples = sorted(self.samples, key=lambda x: x.quality_score, reverse=True)
        split_idx = int(len(sorted_samples) * (1 - val_ratio))
        
        return sorted_samples[:split_idx], sorted_samples[split_idx:]
    
    def export_cleaned_data(self, output_path: Path) -> None:
        output_data = [
            {"input": s.input, "output": s.output, "quality": s.quality_score}
            for s in self.samples
        ]
        
        with output_path.open('w') as f:
            json.dump(output_data, f, indent=2)
    
    def process_pipeline(self, input_path: Path, output_path: Path) -> DatasetStats:
        self.load_data(input_path)
        self.calculate_quality_scores()
        
        duplicates_removed = self.remove_duplicates()
        low_quality_removed = self.filter_by_quality()
        self.balance_dataset()
        
        stats = self.get_statistics()
        stats.duplicates_removed = duplicates_removed
        stats.low_quality_removed = low_quality_removed
        
        self.export_cleaned_data(output_path)
        
        return stats
