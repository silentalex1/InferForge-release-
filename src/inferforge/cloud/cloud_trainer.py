"""Cloud integration and hybrid training capabilities for premium users."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

from inferforge.core.premium import get_premium_manager


class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    CUSTOM = "custom"


class TrainingMode(Enum):
    """Training modes."""
    LOCAL_ONLY = "local_only"
    CLOUD_ONLY = "cloud_only"
    HYBRID = "hybrid"


@dataclass
class CloudConfig:
    """Configuration for cloud training."""
    provider: CloudProvider = CloudProvider.AWS
    region: str = "us-east-1"
    instance_type: str = "p3.2xlarge"
    min_instances: int = 1
    max_instances: int = 4
    spot_instances: bool = True
    storage_bucket: str = ""
    storage_path: str = "inferforge-models"
    use_iam_role: bool = True
    iam_role_arn: str = ""
    security_group_ids: list[str] = field(default_factory=list)
    subnet_ids: list[str] = field(default_factory=list)
    key_name: str = ""
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class HybridConfig:
    """Configuration for hybrid training."""
    local_gpu_count: int = 1
    cloud_gpu_count: int = 2
    sync_frequency: int = 100  # Steps between syncs
    local_batch_size: int = 4
    cloud_batch_size: int = 8
    load_balancing: str = "round_robin"  # round_robin, performance_based, cost_based
    fallback_to_cloud: bool = True
    max_local_memory_gb: float = 16.0


@dataclass
class TrainingJob:
    """Cloud training job."""
    job_id: str = ""
    status: str = "pending"  # pending, running, completed, failed, stopped
    instance_id: str = ""
    instance_type: str = ""
    region: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0
    cost: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)


class CloudTrainer:
    """Cloud training manager for premium users."""
    
    def __init__(self, config: CloudConfig | None = None):
        self.premium_manager = get_premium_manager()
        self.config = config or CloudConfig()
        self.jobs: dict[str, TrainingJob] = {}
        self._client = None
        self._resource = None
        
        if not self.premium_manager.has_feature("cloud_training"):
            raise PermissionError("Cloud training is not available in your current tier")
    
    def _setup_aws_client(self) -> bool:
        """Setup AWS client for cloud operations."""
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for AWS cloud training. Install with: pip install boto3")
        
        try:
            # Use IAM role if available, otherwise use credentials from environment
            if self.config.use_iam_role and self.config.iam_role_arn:
                sts_client = boto3.client('sts')
                assumed_role = sts_client.assume_role(
                    RoleArn=self.config.iam_role_arn,
                    RoleSessionName='InferForgeTraining'
                )
                credentials = assumed_role['Credentials']
                
                self._client = boto3.client(
                    'ec2',
                    region_name=self.config.region,
                    aws_access_key_id=credentials['AccessKeyId'],
                    aws_secret_access_key=credentials['SecretAccessKey'],
                    aws_session_token=credentials['SessionToken']
                )
            else:
                self._client = boto3.client('ec2', region_name=self.config.region)
            
            self._resource = boto3.resource('ec2', region_name=self.config.region)
            return True
            
        except Exception as e:
            print(f"Failed to setup AWS client: {e}")
            return False
    
    def estimate_cost(self, duration_hours: float, instance_type: str | None = None) -> float:
        """Estimate training cost on cloud."""
        instance_type = instance_type or self.config.instance_type
        
        # Simplified pricing (in $/hour) - should be replaced with actual API calls
        pricing = {
            "p3.2xlarge": 3.06,  # V100
            "p3.8xlarge": 12.24,  # 4x V100
            "p3.16xlarge": 24.48,  # 8x V100
            "p4d.24xlarge": 32.77,  # 8x A100
            "g4dn.xlarge": 0.526,  # T4
            "g4dn.2xlarge": 0.752,  # 1x T4
            "g5.xlarge": 1.006,  # A10G
            "g5.2xlarge": 1.693,  # 1x A10G
        }
        
        base_cost = pricing.get(instance_type, 5.0)
        
        # Apply spot instance discount
        if self.config.spot_instances:
            base_cost *= 0.3  # 70% discount for spot instances
        
        return base_cost * duration_hours
    
    def launch_training_instance(
        self,
        model_name: str,
        training_script: str,
        data_path: str,
        duration_hours: float = 2.0,
    ) -> TrainingJob:
        """Launch a cloud training instance."""
        if not self._setup_aws_client():
            raise RuntimeError("Failed to setup cloud client")
        
        # Check limits
        limits = self.premium_manager.get_limits()
        if not limits.cloud_training_enabled:
            raise PermissionError("Cloud training not enabled in your tier")
        
        # Create job
        job = TrainingJob(
            job_id=f"job-{int(time.time())}",
            status="pending",
            instance_type=self.config.instance_type,
            region=self.config.region,
        )
        
        try:
            # User data script for instance setup
            user_data = f"""#!/bin/bash
# Setup InferForge training environment
cd /home/ubuntu
git clone https://github.com/inferforge/inferforge.git
cd inferforge
pip install -e .
# Download training data
aws s3 cp s3://{self.config.storage_bucket}/{data_path} /tmp/training_data.json
# Run training
python -m inferforge.training.cloud_worker {model_name} /tmp/training_data.json --cloud-mode
# Upload results
aws s3 cp /tmp/trained_model s3://{self.config.storage_bucket}/{self.config.storage_path}/{job.job_id}/
"""
            
            # Launch instance
            response = self._client.run_instances(
                ImageId='ami-0c55b159cbfafe1f0',  # Ubuntu 22.04 LTS
                InstanceType=self.config.instance_type,
                MinCount=1,
                MaxCount=1,
                UserData=user_data,
                TagSpecifications=[
                    {
                        'ResourceType': 'instance',
                        'Tags': [
                            {'Key': 'Name', 'Value': f'InferForge-{job.job_id}'},
                            {'Key': 'Purpose', 'Value': 'Training'},
                            {'Key': 'JobID', 'Value': job.job_id},
                        ]
                    }
                ],
                InstanceMarketOptions={
                    'MarketType': 'spot' if self.config.spot_instances else 'on-demand',
                    'SpotOptions': {
                        'SpotInstanceType': 'one-time',
                        'InstanceInterruptionBehavior': 'terminate',
                    } if self.config.spot_instances else {}
                } if self.config.spot_instances else None,
            )
            
            instance_id = response['Instances'][0]['InstanceId']
            job.instance_id = instance_id
            job.status = "running"
            job.started_at = time.time()
            
            # Estimate cost
            job.cost = self.estimate_cost(duration_hours)
            
            self.jobs[job.job_id] = job
            
            return job
            
        except ClientError as e:
            job.status = "failed"
            self.jobs[job.job_id] = job
            raise RuntimeError(f"Failed to launch instance: {e}")
    
    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Get status of a cloud training job."""
        if job_id not in self.jobs:
            return {"error": "Job not found"}
        
        job = self.jobs[job_id]
        
        try:
            # Get instance status from AWS
            if self._client and job.instance_id:
                response = self._client.describe_instances(InstanceIds=[job.instance_id])
                instance_state = response['Reservations'][0]['Instances'][0]['State']['Name']
                
                if instance_state == 'terminated':
                    job.status = "completed"
                    job.completed_at = time.time()
                elif instance_state == 'stopped':
                    job.status = "stopped"
                    job.completed_at = time.time()
                elif instance_state == 'running':
                    job.status = "running"
                
                # Update cost estimate
                elapsed_hours = (time.time() - job.started_at) / 3600.0 if job.started_at else 0
                job.cost = self.estimate_cost(elapsed_hours, job.instance_type)
            
            return {
                "job_id": job.job_id,
                "status": job.status,
                "instance_id": job.instance_id,
                "instance_type": job.instance_type,
                "region": job.region,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
                "cost": job.cost,
                "metrics": job.metrics,
            }
            
        except Exception as e:
            return {
                "job_id": job.job_id,
                "status": "error",
                "error": str(e),
            }
    
    def stop_job(self, job_id: str) -> bool:
        """Stop a running training job."""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        
        try:
            if self._client and job.instance_id:
                self._client.terminate_instances(InstanceIds=[job.instance_id])
                job.status = "stopped"
                job.completed_at = time.time()
                return True
        except Exception as e:
            print(f"Failed to stop job: {e}")
        
        return False
    
    def list_jobs(self) -> list[dict[str, Any]]:
        """List all cloud training jobs."""
        return [self.get_job_status(job_id) for job_id in self.jobs.keys()]


class HybridTrainer:
    """Hybrid training combining local and cloud resources."""
    
    def __init__(self, cloud_config: CloudConfig | None = None, hybrid_config: HybridConfig | None = None):
        self.premium_manager = get_premium_manager()
        self.cloud_trainer = CloudTrainer(cloud_config) if cloud_config else None
        self.hybrid_config = hybrid_config or HybridConfig()
        
        if not self.premium_manager.has_feature("hybrid_training"):
            raise PermissionError("Hybrid training is not available in your current tier")
    
    def setup_hybrid_training(
        self,
        model_name: str,
        local_training_fn: Callable,
        cloud_training_fn: Callable,
        data_path: str,
    ) -> dict[str, Any]:
        """Setup hybrid training with local and cloud components."""
        
        # Check local resources
        try:
            import torch
            if torch.cuda.is_available():
                local_gpu_count = torch.cuda.device_count()
            else:
                local_gpu_count = 0
        except ImportError:
            local_gpu_count = 0
        
        self.hybrid_config.local_gpu_count = min(local_gpu_count, self.hybrid_config.local_gpu_count)
        
        # If no local GPUs, fallback to cloud
        if self.hybrid_config.local_gpu_count == 0 and self.hybrid_config.fallback_to_cloud:
            print("No local GPUs detected, falling back to cloud-only training")
            return self._cloud_only_training(model_name, cloud_training_fn, data_path)
        
        # Setup hybrid training
        setup_info = {
            "mode": "hybrid",
            "local_gpus": self.hybrid_config.local_gpu_count,
            "cloud_gpus": self.hybrid_config.cloud_gpu_count,
            "local_batch_size": self.hybrid_config.local_batch_size,
            "cloud_batch_size": self.hybrid_config.cloud_batch_size,
            "sync_frequency": self.hybrid_config.sync_frequency,
            "load_balancing": self.hybrid_config.load_balancing,
        }
        
        return setup_info
    
    def _cloud_only_training(
        self,
        model_name: str,
        cloud_training_fn: Callable,
        data_path: str,
    ) -> dict[str, Any]:
        """Fallback to cloud-only training."""
        if not self.cloud_trainer:
            raise RuntimeError("Cloud trainer not configured")
        
        job = self.cloud_trainer.launch_training_instance(
            model_name=model_name,
            training_script="",
            data_path=data_path,
            duration_hours=2.0,
        )
        
        return {
            "mode": "cloud_only",
            "job_id": job.job_id,
            "instance_id": job.instance_id,
            "estimated_cost": job.cost,
        }
    
    def run_hybrid_training(
        self,
        model_name: str,
        local_training_fn: Callable,
        cloud_training_fn: Callable,
        progress_callback: Callable | None = None,
    ) -> dict[str, Any]:
        """Execute hybrid training with automatic load balancing."""
        
        results = {
            "mode": "hybrid",
            "local_results": None,
            "cloud_results": None,
            "sync_points": [],
            "total_time": 0.0,
            "total_cost": 0.0,
        }
        
        start_time = time.time()
        
        try:
            # Start local training
            if self.hybrid_config.local_gpu_count > 0:
                if progress_callback:
                    progress_callback("Starting local training...", 0.1)
                
                # Run local training with smaller batch size
                local_results = local_training_fn(
                    batch_size=self.hybrid_config.local_batch_size,
                    gpu_count=self.hybrid_config.local_gpu_count,
                )
                results["local_results"] = local_results
            
            # Start cloud training if configured
            if self.hybrid_config.cloud_gpu_count > 0 and self.cloud_trainer:
                if progress_callback:
                    progress_callback("Starting cloud training...", 0.3)
                
                # Launch cloud training
                cloud_job = self.cloud_trainer.launch_training_instance(
                    model_name=model_name,
                    training_script="",
                    data_path="",
                    duration_hours=2.0,
                )
                
                # Monitor cloud job
                while True:
                    job_status = self.cloud_trainer.get_job_status(cloud_job.job_id)
                    if job_status["status"] in ["completed", "stopped", "failed"]:
                        break
                    time.sleep(30)
                
                results["cloud_results"] = job_status
                results["total_cost"] = job_status.get("cost", 0.0)
            
            results["total_time"] = time.time() - start_time
            
            if progress_callback:
                progress_callback("Hybrid training completed", 1.0)
            
            return results
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"Hybrid training failed: {e}", 0.0)
            raise
    
    def get_hybrid_status(self) -> dict[str, Any]:
        """Get status of hybrid training setup."""
        return {
            "local_gpu_count": self.hybrid_config.local_gpu_count,
            "cloud_gpu_count": self.hybrid_config.cloud_gpu_count,
            "mode": "hybrid" if self.hybrid_config.local_gpu_count > 0 and self.hybrid_config.cloud_gpu_count > 0 else "cloud_only",
            "load_balancing": self.hybrid_config.load_balancing,
            "sync_frequency": self.hybrid_config.sync_frequency,
            "premium_tier": self.premium_manager.get_current_tier().value,
        }
