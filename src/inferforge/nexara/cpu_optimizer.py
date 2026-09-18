"""CPU-optimized training for reducing GPU dependency."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    raise ImportError("CPU optimization requires: pip install torch transformers")

class CPUOptimizer:
    """CPU-optimized training configuration and utilities."""
    
    def __init__(self):
        self.cpu_info = self._detect_cpu_capabilities()
    
    def _detect_cpu_capabilities(self) -> dict[str, Any]:
        """Detect CPU capabilities and optimizations."""
        import platform

        import psutil
        
        cpu_info = {
            "architecture": platform.machine(),
            "cpu_count": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
            "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            "available_memory_gb": psutil.virtual_memory().available / (1024**3),
            "cpu_frequency": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
        }
        
                                              
        cpu_info["supports_avx"] = self._check_avx_support()
        cpu_info["supports_avx2"] = self._check_avx2_support()
        cpu_info["supports_avx512"] = self._check_avx512_support()
        cpu_info["supports_mkl"] = self._check_mkl_support()
        
        return cpu_info
    
    def _check_avx_support(self) -> bool:
        """Check for AVX support."""
        try:
            import cpuinfo
            return 'avx' in cpuinfo.get_cpu_info()['flags']
        except:
            return False
    
    def _check_avx2_support(self) -> bool:
        """Check for AVX2 support."""
        try:
            import cpuinfo
            return 'avx2' in cpuinfo.get_cpu_info()['flags']
        except:
            return False
    
    def _check_avx512_support(self) -> bool:
        """Check for AVX512 support."""
        try:
            import cpuinfo
            return 'avx512f' in cpuinfo.get_cpu_info()['flags']
        except:
            return False
    
    def _check_mkl_support(self) -> bool:
        """Check for Intel MKL support."""
        try:
            import torch
            return torch.backends.mkldnn.is_available()
        except:
            return False
    
    def get_cpu_optimized_config(self, base_config: dict[str, Any]) -> dict[str, Any]:
        """
        Optimize training configuration for CPU execution.
        
        Args:
            base_config: Base training configuration
        
        Returns:
            CPU-optimized configuration
        """
        optimized = base_config.copy()
        
                              
        ram_gb = self.cpu_info["available_memory_gb"]
        
        if ram_gb < 8:
                                      
            optimized["batch_size"] = 1
            optimized["gradient_accumulation_steps"] = 16
            optimized["gradient_checkpointing"] = True
            optimized["max_memory_mb"] = int(ram_gb * 1024 * 0.6)
        elif ram_gb < 16:
                                         
            optimized["batch_size"] = 2
            optimized["gradient_accumulation_steps"] = 8
            optimized["gradient_checkpointing"] = True
            optimized["max_memory_mb"] = int(ram_gb * 1024 * 0.7)
        elif ram_gb < 32:
                                       
            optimized["batch_size"] = 4
            optimized["gradient_accumulation_steps"] = 4
            optimized["gradient_checkpointing"] = False
            optimized["max_memory_mb"] = int(ram_gb * 1024 * 0.8)
        else:
                                            
            optimized["batch_size"] = 8
            optimized["gradient_accumulation_steps"] = 2
            optimized["gradient_checkpointing"] = False
            optimized["max_memory_mb"] = int(ram_gb * 1024 * 0.85)
        
                                    
        optimized["use_cuda"] = False
        optimized["fp16"] = False                                      
        optimized["bf16"] = False
        optimized["torch_dtype"] = "float32"
        
                                       
        optimized["dataloader_num_workers"] = min(self.cpu_info["physical_cores"], 4)
        optimized["pin_memory"] = False                           
        
                                                    
        if "learning_rate" in optimized:
                                                                                  
            optimized["learning_rate"] = optimized["learning_rate"] * 1.5
        
                                                                   
        if "num_train_epochs" in optimized:
            optimized["num_train_epochs"] = max(1, optimized["num_train_epochs"] // 2)
        
                                                                    
        if "logging_steps" in optimized:
            optimized["logging_steps"] = max(1, optimized["logging_steps"] // 2)
        
        return optimized
    
    def enable_cpu_specific_optimizations(self) -> None:
        """Enable CPU-specific PyTorch optimizations."""
        import torch
        
                                 
        if self.cpu_info["supports_mkl"]:
            torch.backends.mkldnn.enabled = True
            print("MKL optimizations enabled")
        
                               
        torch.set_num_threads(self.cpu_info["physical_cores"])
        print(f"PyTorch threads set to {self.cpu_info['physical_cores']}")
        
                                                             
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
                                                    
        os.environ["OMP_NUM_THREADS"] = str(self.cpu_info["physical_cores"])
        os.environ["MKL_NUM_THREADS"] = str(self.cpu_info["physical_cores"])
        os.environ["NUMEXPR_NUM_THREADS"] = str(self.cpu_info["physical_cores"])
    
    def get_cpu_efficient_model(self, model_name: str, use_quantization: bool = True) -> tuple[Any, Any]:
        """
        Load model with CPU-specific optimizations.
        
        Args:
            model_name: Name or path of the model
            use_quantization: Whether to use quantization for CPU efficiency
        
        Returns:
            Tuple of (model, tokenizer)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading model with CPU optimizations: {model_name}")
        
                        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
                                           
        if use_quantization:
            try:
                from transformers import BitsAndBytesConfig
                
                                                           
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                )
                print("8-bit quantization enabled for CPU efficiency")
            except Exception as e:
                print(f"Quantization not available: {e}, using float32")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cpu",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
        
                                                
        model = self._optimize_model_for_cpu(model)
        
        return model, tokenizer
    
    def _optimize_model_for_cpu(self, model: Any) -> Any:
        """Apply CPU-specific optimizations to model."""
        import torch
        
                                        
        if hasattr(torch.jit, "script"):
            try:
                                                                   
                pass                                       
            except:
                pass
        
                                                               
        model.eval()
        
        return model
    
    def estimate_cpu_training_time(
        self,
        model_size_params: int,
        num_samples: int,
        num_epochs: int,
        batch_size: int
    ) -> dict[str, float]:
        """
        Estimate CPU training time based on hardware and model size.
        
        Args:
            model_size_params: Number of parameters in the model
            num_samples: Number of training samples
            num_epochs: Number of training epochs
            batch_size: Batch size
        
        Returns:
            Dictionary with time estimates
        """
                                                    
        cpu_score = (
            self.cpu_info["physical_cores"] * 
            (2 if self.cpu_info["supports_avx2"] else 1) *
            (1.5 if self.cpu_info["supports_avx512"] else 1) *
            (1.2 if self.cpu_info["supports_mkl"] else 1)
        )
        
                                           
        steps_per_epoch = num_samples // batch_size
        total_steps = steps_per_epoch * num_epochs
        
                                             
        time_per_step = (model_size_params / 1e6) / cpu_score                    
        
        total_time_seconds = total_steps * time_per_step
        total_time_minutes = total_time_seconds / 60
        total_time_hours = total_time_minutes / 60
        
        return {
            "estimated_seconds": total_time_seconds,
            "estimated_minutes": total_time_minutes,
            "estimated_hours": total_time_hours,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "cpu_score": cpu_score,
        }
    
    def generate_cpu_training_report(self) -> dict[str, Any]:
        """Generate comprehensive CPU training capability report."""
        return {
            "cpu_info": self.cpu_info,
            "optimization_support": {
                "avx": self.cpu_info["supports_avx"],
                "avx2": self.cpu_info["supports_avx2"],
                "avx512": self.cpu_info["supports_avx512"],
                "mkl": self.cpu_info["supports_mkl"],
            },
            "memory_info": {
                "total_gb": self.cpu_info["total_memory_gb"],
                "available_gb": self.cpu_info["available_memory_gb"],
                "recommended_batch_size": self._get_recommended_batch_size(),
            },
            "training_recommendations": self._get_training_recommendations(),
        }
    
    def _get_recommended_batch_size(self) -> int:
        """Get recommended batch size based on available memory."""
        ram_gb = self.cpu_info["available_memory_gb"]
        
        if ram_gb < 8:
            return 1
        elif ram_gb < 16:
            return 2
        elif ram_gb < 32:
            return 4
        else:
            return 8
    
    def _get_training_recommendations(self) -> list[str]:
        """Get CPU training recommendations."""
        recommendations = []
        
        if self.cpu_info["available_memory_gb"] < 16:
            recommendations.append("Use gradient checkpointing to reduce memory usage")
            recommendations.append("Use small batch sizes with high gradient accumulation")
        
        if not self.cpu_info["supports_avx2"]:
            recommendations.append("Consider upgrading CPU for better AVX2 support")
        
        if self.cpu_info["physical_cores"] < 8:
            recommendations.append("Training will be slow due to limited CPU cores")
        
        if self.cpu_info["supports_mkl"]:
            recommendations.append("MKL optimizations will be automatically enabled")
        
        recommendations.append("Use 8-bit quantization for faster CPU training")
        recommendations.append("Consider using smaller models for CPU training")
        recommendations.append("Enable mixed precision if your CPU supports it")
        
        return recommendations

def enable_global_cpu_optimizations() -> None:
    """Enable global CPU optimizations for PyTorch."""
    import os

    import torch
    
                                  
    physical_cores = os.cpu_count() or 4
    torch.set_num_threads(physical_cores)
    
                           
    os.environ["OMP_NUM_THREADS"] = str(physical_cores)
    os.environ["MKL_NUM_THREADS"] = str(physical_cores)
    os.environ["NUMEXPR_NUM_THREADS"] = str(physical_cores)
    
                           
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
                             
    try:
        torch.backends.mkldnn.enabled = True
    except:
        pass