from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from platformdirs import user_data_dir


class AnalyticsManager:
    def __init__(self):
        self.data_dir = Path(user_data_dir("inferforge"))
        self.analytics_file = self.data_dir / "analytics.json"
        self.analytics_data: dict[str, Any] = {}
        self._load_analytics()
    
    def _load_analytics(self) -> None:
        if self.analytics_file.exists():
            with open(self.analytics_file, 'r') as f:
                self.analytics_data = json.load(f)
        else:
            self.analytics_data = {
                "model_usage": defaultdict(lambda: {"count": 0, "tokens": 0, "time": 0}),
                "command_usage": defaultdict(lambda: {"count": 0, "last_used": None}),
                "errors": defaultdict(lambda: {"count": 0, "last_occurred": None}),
                "performance": {},
                "daily_stats": defaultdict(lambda: {"requests": 0, "tokens": 0, "errors": 0})
            }
    
    def _save_analytics(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        serializable_data = {
            "model_usage": {k: dict(v) for k, v in self.analytics_data["model_usage"].items()},
            "command_usage": {k: dict(v) for k, v in self.analytics_data["command_usage"].items()},
            "errors": {k: dict(v) for k, v in self.analytics_data["errors"].items()},
            "performance": self.analytics_data["performance"],
            "daily_stats": {k: dict(v) for k, v in self.analytics_data["daily_stats"].items()}
        }
        
        with open(self.analytics_file, 'w') as f:
            json.dump(serializable_data, f, indent=2)
    
    def track_model_usage(self, model_name: str, tokens: int, inference_time: float) -> None:
        self.analytics_data["model_usage"][model_name]["count"] += 1
        self.analytics_data["model_usage"][model_name]["tokens"] += tokens
        self.analytics_data["model_usage"][model_name]["time"] += inference_time
        
        today = datetime.now().strftime("%Y-%m-%d")
        self.analytics_data["daily_stats"][today]["requests"] += 1
        self.analytics_data["daily_stats"][today]["tokens"] += tokens
        
        self._save_analytics()
    
    def track_command_usage(self, command: str) -> None:
        self.analytics_data["command_usage"][command]["count"] += 1
        self.analytics_data["command_usage"][command]["last_used"] = datetime.now().isoformat()
        self._save_analytics()
    
    def track_error(self, error_type: str) -> None:
        self.analytics_data["errors"][error_type]["count"] += 1
        self.analytics_data["errors"][error_type]["last_occurred"] = datetime.now().isoformat()
        
        today = datetime.now().strftime("%Y-%m-%d")
        self.analytics_data["daily_stats"][today]["errors"] += 1
        
        self._save_analytics()
    
    def record_performance(self, model_name: str, metric: str, value: float) -> None:
        if model_name not in self.analytics_data["performance"]:
            self.analytics_data["performance"][model_name] = {}
        
        if metric not in self.analytics_data["performance"][model_name]:
            self.analytics_data["performance"][model_name][metric] = []
        
        self.analytics_data["performance"][model_name][metric].append({
            "value": value,
            "timestamp": datetime.now().isoformat()
        })
        
        self._save_analytics()
    
    def get_model_stats(self, model_name: str | None = None) -> dict:
        if model_name:
            return dict(self.analytics_data["model_usage"].get(model_name, {}))
        return {k: dict(v) for k, v in self.analytics_data["model_usage"].items()}
    
    def get_command_stats(self) -> dict:
        return {k: dict(v) for k, v in self.analytics_data["command_usage"].items()}
    
    def get_error_stats(self) -> dict:
        return {k: dict(v) for k, v in self.analytics_data["errors"].items()}
    
    def get_daily_stats(self, days: int = 7) -> dict:
        stats = {}
        today = datetime.now()
        
        for i in range(days):
            date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            stats[date] = dict(self.analytics_data["daily_stats"].get(date, {"requests": 0, "tokens": 0, "errors": 0}))
        
        return stats
    
    def get_performance_stats(self, model_name: str | None = None) -> dict:
        if model_name:
            return self.analytics_data["performance"].get(model_name, {})
        return self.analytics_data["performance"]
    
    def export_analytics(self, export_path: Path, format: str = "json") -> bool:
        try:
            if format == "json":
                with open(export_path, 'w') as f:
                    json.dump(self.analytics_data, f, indent=2)
            elif format == "csv":
                import csv
                
                with open(export_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Model", "Count", "Tokens", "Time"])
                    
                    for model, stats in self.analytics_data["model_usage"].items():
                        writer.writerow([model, stats["count"], stats["tokens"], stats["time"]])
            
            return True
        except Exception as e:
            return False
    
    def clear_analytics(self, older_than_days: int | None = None) -> None:
        if older_than_days:
            cutoff_date = datetime.now() - timedelta(days=older_than_days)
            
            for date in list(self.analytics_data["daily_stats"].keys()):
                try:
                    date_obj = datetime.strptime(date, "%Y-%m-%d")
                    if date_obj < cutoff_date:
                        del self.analytics_data["daily_stats"][date]
                except:
                    pass
        else:
            self.analytics_data = {
                "model_usage": defaultdict(lambda: {"count": 0, "tokens": 0, "time": 0}),
                "command_usage": defaultdict(lambda: {"count": 0, "last_used": None}),
                "errors": defaultdict(lambda: {"count": 0, "last_occurred": None}),
                "performance": {},
                "daily_stats": defaultdict(lambda: {"requests": 0, "tokens": 0, "errors": 0})
            }
        
        self._save_analytics()


_analytics_manager: AnalyticsManager | None = None


def get_analytics_manager() -> AnalyticsManager:
    global _analytics_manager
    if _analytics_manager is None:
        _analytics_manager = AnalyticsManager()
    return _analytics_manager