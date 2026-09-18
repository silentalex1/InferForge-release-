from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ABTestConfig:
    test_id: str
    model_a: str
    model_b: str
    traffic_split: float
    metrics: list[str]
    duration: int
    sample_size: int


@dataclass
class ABTestResult:
    test_id: str
    winner: str
    confidence: float
    metric_scores: dict[str, dict[str, float]]
    statistical_significance: bool
    recommendation: str


class ABTestingEngine:
    def __init__(self):
        self.active_tests: dict[str, ABTestConfig] = {}
        self.test_results: dict[str, ABTestResult] = {}
        self.test_history: list[dict[str, Any]] = []
    
    def create_test(self, model_a: str, model_b: str, traffic_split: float = 0.5, 
                   metrics: list[str] | None = None, duration: int = 3600, sample_size: int = 1000) -> ABTestConfig:
        test_id = self._generate_test_id(model_a, model_b)
        
        config = ABTestConfig(
            test_id=test_id,
            model_a=model_a,
            model_b=model_b,
            traffic_split=traffic_split,
            metrics=metrics or ["accuracy", "latency", "user_satisfaction"],
            duration=duration,
            sample_size=sample_size
        )
        
        self.active_tests[test_id] = config
        return config
    
    def _generate_test_id(self, model_a: str, model_b: str) -> str:
        import hashlib
        content = f"{model_a}-{model_b}-{random.randint(1000, 9999)}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def assign_model(self, test_id: str, user_id: str) -> str:
        config = self.active_tests.get(test_id)
        if not config:
            raise ValueError(f"Test {test_id} not found")
        
        user_hash = hash(user_id) % 100
        if user_hash < config.traffic_split * 100:
            return config.model_a
        else:
            return config.model_b
    
    def record_metric(self, test_id: str, model: str, metric_name: str, value: float) -> None:
        test_key = f"{test_id}_{model}_{metric_name}"
        
        if test_key not in self.test_history:
            self.test_history.append({
                "test_id": test_id,
                "model": model,
                "metric": metric_name,
                "values": []
            })
        
        for entry in self.test_history:
            if entry["test_id"] == test_id and entry["model"] == model and entry["metric"] == metric_name:
                entry["values"].append(value)
                break
    
    def conclude_test(self, test_id: str) -> ABTestResult:
        config = self.active_tests.get(test_id)
        if not config:
            raise ValueError(f"Test {test_id} not found")
        
        metric_scores = {}
        
        for metric in config.metrics:
            scores_a = self._get_metric_values(test_id, config.model_a, metric)
            scores_b = self._get_metric_values(test_id, config.model_b, metric)
            
            avg_a = sum(scores_a) / len(scores_a) if scores_a else 0.0
            avg_b = sum(scores_b) / len(scores_b) if scores_b else 0.0
            
            metric_scores[metric] = {
                config.model_a: avg_a,
                config.model_b: avg_b,
                "improvement": avg_b - avg_a
            }
        
        winner, confidence, significance = self._determine_winner(metric_scores)
        recommendation = self._generate_recommendation(winner, confidence, metric_scores)
        
        result = ABTestResult(
            test_id=test_id,
            winner=winner,
            confidence=confidence,
            metric_scores=metric_scores,
            statistical_significance=significance,
            recommendation=recommendation
        )
        
        self.test_results[test_id] = result
        del self.active_tests[test_id]
        
        return result
    
    def _get_metric_values(self, test_id: str, model: str, metric: str) -> list[float]:
        values = []
        for entry in self.test_history:
            if entry["test_id"] == test_id and entry["model"] == model and entry["metric"] == metric:
                values.extend(entry["values"])
        return values
    
    def _determine_winner(self, metric_scores: dict[str, dict[str, float]]) -> tuple[str, float, bool]:
        model_a_wins = 0
        model_b_wins = 0
        
        for metric, scores in metric_scores.items():
            if scores.get("improvement", 0) > 0:
                model_b_wins += 1
            else:
                model_a_wins += 1
        
        if model_a_wins > model_b_wins:
            winner = list(metric_scores.values())[0].keys()[0]
        elif model_b_wins > model_a_wins:
            winner = list(metric_scores.values())[0].keys()[1]
        else:
            winner = "tie"
        
        confidence = abs(model_a_wins - model_b_wins) / len(metric_scores)
        significance = confidence > 0.6
        
        return winner, confidence, significance
    
    def _generate_recommendation(self, winner: str, confidence: float, metric_scores: dict[str, dict[str, float]]) -> str:
        if winner == "tie":
            return "No clear winner. Consider running test longer or with different metrics."
        
        if confidence > 0.8:
            return f"Strong confidence. Deploy {winner} to production."
        elif confidence > 0.6:
            return f"Moderate confidence. Consider deploying {winner} with monitoring."
        else:
            return "Low confidence. Continue testing or analyze specific metrics."
    
    def get_test_status(self, test_id: str) -> dict[str, Any]:
        if test_id in self.active_tests:
            config = self.active_tests[test_id]
            return {
                "status": "active",
                "config": config.__dict__,
                "samples_collected": self._count_samples(test_id)
            }
        elif test_id in self.test_results:
            return {
                "status": "completed",
                "result": self.test_results[test_id].__dict__
            }
        else:
            return {"status": "not_found"}
    
    def _count_samples(self, test_id: str) -> int:
        count = 0
        for entry in self.test_history:
            if entry["test_id"] == test_id:
                count += len(entry["values"])
        return count
    
    def get_all_tests(self) -> dict[str, Any]:
        return {
            "active": {k: v.__dict__ for k, v in self.active_tests.items()},
            "completed": {k: v.__dict__ for k, v in self.test_results.items()}
        }
    
    def save_test_results(self, output_path: Path) -> None:
        results = {
            "active_tests": {k: v.__dict__ for k, v in self.active_tests.items()},
            "completed_tests": {k: v.__dict__ for k, v in self.test_results.items()},
            "history": self.test_history
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w') as f:
            json.dump(results, f, indent=2)
    
    def load_test_results(self, input_path: Path) -> None:
        with input_path.open('r') as f:
            data = json.load(f)
        
        for test_id, config_data in data["active_tests"].items():
            self.active_tests[test_id] = ABTestConfig(**config_data)
        
        for test_id, result_data in data["completed_tests"].items():
            self.test_results[test_id] = ABTestResult(**result_data)
        
        self.test_history = data.get("history", [])
    
    def create_multi_arm_test(self, models: list[str], traffic_splits: list[float] | None = None) -> dict[str, ABTestConfig]:
        if not traffic_splits:
            traffic_splits = [1.0 / len(models)] * len(models)
        
        tests = {}
        
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                test = self.create_test(models[i], models[j], traffic_splits[i])
                tests[test.test_id] = test
        
        return tests
    
    def get_winner_summary(self) -> list[dict[str, Any]]:
        summary = []
        
        for test_id, result in self.test_results.items():
            summary.append({
                "test_id": test_id,
                "winner": result.winner,
                "confidence": result.confidence,
                "recommendation": result.recommendation
            })
        
        return sorted(summary, key=lambda x: x["confidence"], reverse=True)
