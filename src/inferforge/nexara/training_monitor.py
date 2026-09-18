"""
Training Monitor for Nexara AI-native programming language.

Provides real-time monitoring and visualization of training progress,
metrics, and performance for Nexara training sessions.
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class TrainingStatus(Enum):
    """Training status enumeration."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class MonitorSnapshot:
    """Snapshot of training state at a specific time."""
    timestamp: float
    epoch: int
    step: int
    loss: float
    learning_rate: float
    throughput: float                       
    memory_usage: float         
    gpu_utilization: float              
    status: TrainingStatus
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class LogSnapshot:
    """Snapshot returned by TrainingMonitor.log()."""
    step: int
    loss: float
    nan_skips: int
    timestamp: float
    metrics: Dict[str, float] = field(default_factory=dict)


class TrainingMonitor:
    """Monitor for tracking Nexara training progress and metrics."""

    def __init__(self, log_dir=None, log_interval: int = 10):
        """
        Initialize training monitor.

        Args:
            log_dir: Directory where training_log.jsonl is written (optional)
            log_interval: Seconds between metric snapshots
        """
        self.log_interval = log_interval
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.snapshots: List[MonitorSnapshot] = []
        self.log_entries: List[LogSnapshot] = []
        self.start_time: Optional[float] = None
        self.current_status: TrainingStatus = TrainingStatus.IDLE
        self.best_loss: float = float('inf')
        self.best_epoch: int = 0
        self.nan_skips: int = 0
        self.last_loss: Optional[float] = None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._log_path = self.log_dir / "training_log.jsonl"
        else:
            self._log_path = None

    def record_nan_skip(self) -> None:
        """Record a NaN/overflow gradient skip."""
        self.nan_skips += 1
        self._append_log({"event": "nan_skip", "nan_skips": self.nan_skips, "timestamp": time.time()})

    def log(self, step: int, **metrics) -> LogSnapshot:
        """Log a training step and return a snapshot."""
        loss = float(metrics.pop("loss", 0.0))
        if loss < self.best_loss:
            self.best_loss = loss
        self.last_loss = loss
        snapshot = LogSnapshot(
            step=step,
            loss=loss,
            nan_skips=self.nan_skips,
            timestamp=time.time(),
            metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        )
        self.log_entries.append(snapshot)
        self._append_log({
            "step": snapshot.step,
            "loss": snapshot.loss,
            "nan_skips": snapshot.nan_skips,
            "timestamp": snapshot.timestamp,
            **snapshot.metrics,
        })
        return snapshot

    def summary(self) -> Dict[str, Any]:
        """Summarize the training session so far."""
        return {
            "steps": self.log_entries[-1].step if self.log_entries else 0,
            "last_loss": self.last_loss,
            "best_loss": self.best_loss if self.best_loss != float("inf") else None,
            "nan_skips": self.nan_skips,
            "num_entries": len(self.log_entries),
        }

    def _append_log(self, payload: Dict[str, Any]) -> None:
        if self._log_path is None:
            return
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except OSError:
            pass
        
    def start_training(self):
        """Start monitoring a training session."""
        self.start_time = time.time()
        self.current_status = TrainingStatus.RUNNING
        self._take_snapshot()
    
    def pause_training(self):
        """Pause training monitoring."""
        self.current_status = TrainingStatus.PAUSED
        self._take_snapshot()
    
    def resume_training(self):
        """Resume training monitoring."""
        self.current_status = TrainingStatus.RUNNING
        self._take_snapshot()
    
    def stop_training(self, success: bool = True):
        """Stop monitoring a training session."""
        self.current_status = TrainingStatus.COMPLETED if success else TrainingStatus.FAILED
        self._take_snapshot()
    
    def log_metrics(self, epoch: int, step: int, loss: float, learning_rate: float, **kwargs):
        """
        Log training metrics at a specific point.
        
        Args:
            epoch: Current epoch number
            step: Current step number
            loss: Current loss value
            learning_rate: Current learning rate
            **kwargs: Additional metrics
        """
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
        
        snapshot = MonitorSnapshot(
            timestamp=time.time(),
            epoch=epoch,
            step=step,
            loss=loss,
            learning_rate=learning_rate,
            throughput=kwargs.get('throughput', 0.0),
            memory_usage=kwargs.get('memory_usage', 0.0),
            gpu_utilization=kwargs.get('gpu_utilization', 0.0),
            status=self.current_status,
            metrics=kwargs
        )
        
        self.snapshots.append(snapshot)
    
    def _take_snapshot(self):
        """Take a snapshot of current training state."""
                                                      
        pass
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary of the training session.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.snapshots:
            return {}
        
        return {
            "total_duration": time.time() - self.start_time if self.start_time else 0,
            "num_snapshots": len(self.snapshots),
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "final_loss": self.snapshots[-1].loss if self.snapshots else 0.0,
            "final_status": self.current_status,
            "average_loss": sum(s.loss for s in self.snapshots) / len(self.snapshots),
        }
    
    def get_recent_snapshots(self, count: int = 10) -> List[MonitorSnapshot]:
        """
        Get the most recent training snapshots.
        
        Args:
            count: Number of recent snapshots to return
        
        Returns:
            List of recent snapshots
        """
        return self.snapshots[-count:] if self.snapshots else []
    
    def export_metrics(self, filepath: str):
        """
        Export training metrics to a file.
        
        Args:
            filepath: Path to export metrics to
        """
        import json
        with open(filepath, 'w') as f:
            json.dump({
                'snapshots': [
                    {
                        'timestamp': s.timestamp,
                        'epoch': s.epoch,
                        'step': s.step,
                        'loss': s.loss,
                        'learning_rate': s.learning_rate,
                        'throughput': s.throughput,
                        'memory_usage': s.memory_usage,
                        'gpu_utilization': s.gpu_utilization,
                        'status': s.status.value,
                        'metrics': s.metrics
                    }
                    for s in self.snapshots
                ],
                'summary': self.get_performance_summary()
            }, f, indent=2)
    
    def plot_training_curve(self, save_path: Optional[str] = None):
        """
        Plot training loss curve.
        
        Args:
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            epochs = [s.epoch for s in self.snapshots]
            losses = [s.loss for s in self.snapshots]
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, losses, label='Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Progress')
            plt.legend()
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
            
            plt.close()
        except ImportError:
            print("Matplotlib not available for plotting")
    
    def reset(self):
        """Reset the monitor for a new training session."""
        self.snapshots = []
        self.start_time = None
        self.current_status = TrainingStatus.IDLE
        self.best_loss = float('inf')
        self.best_epoch = 0