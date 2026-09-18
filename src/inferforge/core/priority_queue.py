"""Priority queue system for premium user resource allocation."""

from __future__ import annotations

import heapq
import json
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from inferforge.core.premium import Tier, get_premium_manager


class Priority(Enum):
    """Priority levels for training jobs."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3
    CRITICAL = 4


@dataclass
class TrainingJob:
    """Training job in the priority queue."""
    job_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    model_name: str = ""
    estimated_duration: float = 0.0
    estimated_cost: float = 0.0
    priority: Priority = Priority.NORMAL
    tier: Tier = Tier.COMMUNITY
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    status: str = "queued"  # queued, running, completed, failed, cancelled
    progress: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other: "TrainingJob") -> bool:
        """Compare jobs for priority queue ordering."""
        # Higher priority first
        if self.priority != other.priority:
            return self.priority.value > other.priority.value
        
        # Within same priority, older jobs first
        return self.submitted_at < other.submitted_at


@dataclass
class QueueStats:
    """Statistics for the priority queue."""
    total_jobs: int = 0
    queued_jobs: int = 0
    running_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    average_wait_time: float = 0.0
    average_execution_time: float = 0.0
    premium_jobs_processed: int = 0
    community_jobs_processed: int = 0


class PriorityTrainingQueue:
    """Priority queue for training jobs with premium user benefits."""
    
    def __init__(self, max_concurrent_jobs: int = 4):
        self.premium_manager = get_premium_manager()
        self.max_concurrent_jobs = max_concurrent_jobs
        self.queue: list[TrainingJob] = []
        self.running_jobs: dict[str, TrainingJob] = {}
        self.completed_jobs: dict[str, TrainingJob] = {}
        self.lock = threading.Lock()
        self.stats = QueueStats()
        self._load_state()
    
    def _state_path(self) -> Path:
        from inferforge.core.config import data_dir
        return data_dir() / "training_queue_state.json"
    
    def _load_state(self) -> None:
        """Load queue state from disk."""
        path = self._state_path()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                # Reconstruct queue from saved state
                for job_data in data.get("queue", []):
                    job = TrainingJob(**job_data)
                    if job.status == "queued":
                        heapq.heappush(self.queue, job)
                    elif job.status == "running":
                        self.running_jobs[job.job_id] = job
                    elif job.status in ["completed", "failed"]:
                        self.completed_jobs[job.job_id] = job
            except Exception:
                pass
    
    def _save_state(self) -> None:
        """Save queue state to disk."""
        path = self._state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "queue": [
                job.__dict__ for job in self.queue + list(self.running_jobs.values()) + list(self.completed_jobs.values())
            ],
            "stats": self.stats.__dict__,
        }
        
        with path.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)
    
    def _calculate_priority(self, job: TrainingJob) -> Priority:
        """Calculate priority based on user tier and job characteristics."""
        tier = job.tier
        priority_mult = self.premium_manager.get_priority_score()
        
        # Base priority from tier
        if tier == Tier.ENTERPRISE:
            base_priority = Priority.URGENT
        elif tier == Tier.PREMIUM_PLUS:
            base_priority = Priority.HIGH
        elif tier == Tier.PREMIUM:
            base_priority = Priority.HIGH
        elif tier == Tier.STARTER:
            base_priority = Priority.NORMAL
        else:
            base_priority = Priority.NORMAL
        
        # Apply priority multiplier
        if priority_mult >= 10.0:
            return min(Priority.CRITICAL, Priority(base_priority.value + 1))
        elif priority_mult >= 5.0:
            return base_priority
        elif priority_mult >= 2.0:
            return max(Priority.NORMAL, Priority(base_priority.value - 1))
        else:
            return max(Priority.LOW, Priority(base_priority.value - 1))
    
    def submit_job(
        self,
        user_id: str,
        model_name: str,
        estimated_duration: float,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, float]:
        """Submit a new training job to the queue."""
        with self.lock:
            # Check if user can start training
            can_train, message = self.premium_manager.check_training_limit(estimated_duration / 3600.0)
            if not can_train:
                raise ValueError(message)
            
            # Create job
            tier = self.premium_manager.get_current_tier()
            job = TrainingJob(
                user_id=user_id,
                model_name=model_name,
                estimated_duration=estimated_duration,
                tier=tier,
                metadata=metadata or {},
            )
            
            # Calculate priority
            job.priority = self._calculate_priority(job)
            
            # Add to queue
            heapq.heappush(self.queue, job)
            self.stats.total_jobs += 1
            self.stats.queued_jobs += 1
            
            # Estimate wait time
            estimated_wait = self._estimate_wait_time(job)
            
            self._save_state()
            
            return job.job_id, estimated_wait
    
    def _estimate_wait_time(self, job: TrainingJob) -> float:
        """Estimate wait time for a job."""
        if not self.queue:
            return 0.0
        
        # Calculate total duration of jobs ahead in queue
        total_duration = 0.0
        higher_priority_jobs = 0
        
        for queued_job in self.queue:
            if queued_job.priority.value > job.priority.value:
                total_duration += queued_job.estimated_duration
                higher_priority_jobs += 1
        
        # Add running jobs duration
        for running_job in self.running_jobs.values():
            if running_job.priority.value >= job.priority.value:
                elapsed = time.time() - (running_job.started_at or running_job.submitted_at)
                remaining = max(0, running_job.estimated_duration - elapsed)
                total_duration += remaining
        
        return total_duration
    
    def get_next_job(self) -> TrainingJob | None:
        """Get the next job to run."""
        with self.lock:
            if not self.queue:
                return None
            
            # Check if we can start more jobs
            if len(self.running_jobs) >= self.max_concurrent_jobs:
                return None
            
            # Get highest priority job
            job = heapq.heappop(self.queue)
            job.status = "running"
            job.started_at = time.time()
            
            self.running_jobs[job.job_id] = job
            self.stats.queued_jobs -= 1
            self.stats.running_jobs += 1
            
            # Record as premium or community job
            if job.tier in [Tier.PREMIUM, Tier.PREMIUM_PLUS, Tier.ENTERPRISE]:
                self.stats.premium_jobs_processed += 1
            else:
                self.stats.community_jobs_processed += 1
            
            self._save_state()
            
            return job
    
    def complete_job(self, job_id: str, success: bool = True) -> None:
        """Mark a job as completed."""
        with self.lock:
            if job_id in self.running_jobs:
                job = self.running_jobs.pop(job_id)
                job.status = "completed" if success else "failed"
                job.completed_at = time.time()
                job.progress = 1.0
                
                self.completed_jobs[job_id] = job
                self.stats.running_jobs -= 1
                self.stats.completed_jobs += 1
                
                # Update statistics
                execution_time = job.completed_at - (job.started_at or job.submitted_at)
                wait_time = (job.started_at or job.submitted_at) - job.submitted_at
                
                # Update averages
                total_completed = self.stats.completed_jobs
                self.stats.average_execution_time = (
                    (self.stats.average_execution_time * (total_completed - 1) + execution_time) / total_completed
                )
                self.stats.average_wait_time = (
                    (self.stats.average_wait_time * (total_completed - 1) + wait_time) / total_completed
                )
                
                # Record usage with premium manager
                actual_hours = execution_time / 3600.0
                self.premium_manager.end_training(actual_hours)
                
                self._save_state()
    
    def update_progress(self, job_id: str, progress: float) -> None:
        """Update progress for a running job."""
        with self.lock:
            if job_id in self.running_jobs:
                self.running_jobs[job_id].progress = min(1.0, max(0.0, progress))
                self._save_state()
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued job."""
        with self.lock:
            # Check running jobs
            if job_id in self.running_jobs:
                self.running_jobs[job_id].status = "cancelled"
                self.complete_job(job_id, success=False)
                return True
            
            # Check queued jobs
            for i, job in enumerate(self.queue):
                if job.job_id == job_id:
                    job.status = "cancelled"
                    self.queue.pop(i)
                    heapq.heapify(self.queue)
                    self.stats.queued_jobs -= 1
                    self._save_state()
                    return True
            
            return False
    
    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Get status of a specific job."""
        with self.lock:
            # Check running jobs
            if job_id in self.running_jobs:
                job = self.running_jobs[job_id]
                return {
                    "job_id": job.job_id,
                    "status": job.status,
                    "progress": job.progress,
                    "started_at": job.started_at,
                    "estimated_remaining": max(0, job.estimated_duration - (time.time() - (job.started_at or job.submitted_at))),
                }
            
            # Check queued jobs
            for job in self.queue:
                if job.job_id == job_id:
                    return {
                        "job_id": job.job_id,
                        "status": job.status,
                        "progress": job.progress,
                        "estimated_wait": self._estimate_wait_time(job),
                    }
            
            # Check completed jobs
            if job_id in self.completed_jobs:
                job = self.completed_jobs[job_id]
                return {
                    "job_id": job.job_id,
                    "status": job.status,
                    "progress": job.progress,
                    "completed_at": job.completed_at,
                }
            
            return None
    
    def get_queue_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        with self.lock:
            return {
                "total_jobs": self.stats.total_jobs,
                "queued_jobs": self.stats.queued_jobs,
                "running_jobs": self.stats.running_jobs,
                "completed_jobs": self.stats.completed_jobs,
                "failed_jobs": self.stats.failed_jobs,
                "average_wait_time": self.stats.average_wait_time,
                "average_execution_time": self.stats.average_execution_time,
                "premium_jobs_processed": self.stats.premium_jobs_processed,
                "community_jobs_processed": self.stats.community_jobs_processed,
                "current_tier": self.premium_manager.get_current_tier().value,
                "priority_multiplier": self.premium_manager.get_priority_score(),
            }
    
    def get_user_jobs(self, user_id: str) -> list[dict[str, Any]]:
        """Get all jobs for a specific user."""
        with self.lock:
            user_jobs = []
            
            # Check queued jobs
            for job in self.queue:
                if job.user_id == user_id:
                    user_jobs.append({
                        "job_id": job.job_id,
                        "model_name": job.model_name,
                        "status": job.status,
                        "priority": job.priority.name,
                        "submitted_at": job.submitted_at,
                        "estimated_wait": self._estimate_wait_time(job),
                    })
            
            # Check running jobs
            for job in self.running_jobs.values():
                if job.user_id == user_id:
                    user_jobs.append({
                        "job_id": job.job_id,
                        "model_name": job.model_name,
                        "status": job.status,
                        "priority": job.priority.name,
                        "started_at": job.started_at,
                        "progress": job.progress,
                    })
            
            # Check completed jobs
            for job in self.completed_jobs.values():
                if job.user_id == user_id:
                    user_jobs.append({
                        "job_id": job.job_id,
                        "model_name": job.model_name,
                        "status": job.status,
                        "priority": job.priority.name,
                        "completed_at": job.completed_at,
                    })
            
            return sorted(user_jobs, key=lambda x: x.get("submitted_at", x.get("started_at", x.get("completed_at", 0))), reverse=True)


# Global queue instance
_priority_queue = None

def get_priority_queue() -> PriorityTrainingQueue:
    """Get global priority queue instance."""
    global _priority_queue
    if _priority_queue is None:
        _priority_queue = PriorityTrainingQueue()
    return _priority_queue
