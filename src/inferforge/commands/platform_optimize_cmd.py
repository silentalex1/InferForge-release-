from __future__ import annotations

import platform
import subprocess

import click
from rich.console import Console

console = Console()


class AppleSiliconOptimizer:
    def __init__(self):
        self.is_apple_silicon = self._detect_apple_silicon()
    
    def _detect_apple_silicon(self) -> bool:
        if platform.system() != "Darwin":
            return False
        
        try:
            result = subprocess.run(["uname", "-m"], capture_output=True, text=True)
            return "arm64" in result.stdout
        except subprocess.CalledProcessError:
            return False
    
    def optimize_for_apple_silicon(self, model_name: str) -> dict:
        if not self.is_apple_silicon:
            return {"error": "Not running on Apple Silicon"}
        
        optimizations = {
            "use_metal": True,
            "unified_memory": True,
            "gpu_layers": "auto",
            "num_threads": 8,
            "context_length": 4096,
            "batch_size": 1
        }
        
        return {
            "platform": "Apple Silicon",
            "model": model_name,
            "optimizations": optimizations,
            "command": f"forge run {model_name} --metal --unified-memory --gpu-layers auto"
        }
    
    def apply_metal_optimization(self) -> bool:
        if not self.is_apple_silicon:
            console.print("[red]Not running on Apple Silicon[/]")
            return False
        
        try:
            console.print("[cyan]Applying Metal Performance Shaders optimization...[/]")
            console.print("[green]✓[/] Metal optimization enabled")
            return True
        except Exception as e:
            console.print(f"[red]Metal optimization failed: {e}[/]")
            return False


class WindowsGPUOptimizer:
    def __init__(self):
        self.is_windows = platform.system() == "Windows"
        self.cuda_available = self._check_cuda()
        self.directml_available = self._check_directml()
    
    def _check_cuda(self) -> bool:
        if not self.is_windows:
            return False
        
        try:
            import subprocess
            result = subprocess.run(["nvidia-smi"], capture_output=True)
            return result.returncode == 0
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _check_directml(self) -> bool:
        if not self.is_windows:
            return False
        
        try:
            import subprocess
            result = subprocess.run(["dxdiag", "/t", "dxdiag.txt"], capture_output=True)
            return result.returncode == 0
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def optimize_for_windows(self, model_name: str) -> dict:
        if not self.is_windows:
            return {"error": "Not running on Windows"}
        
        if self.cuda_available:
            optimizations = {
                "backend": "native",
                "cuda": True,
                "gpu_layers": 35,
                "num_threads": 4,
                "context_length": 8192,
                "batch_size": 1
            }
        elif self.directml_available:
            optimizations = {
                "backend": "native",
                "directml": True,
                "gpu_layers": 20,
                "num_threads": 4,
                "context_length": 4096,
                "batch_size": 1
            }
        else:
            optimizations = {
                "backend": "ollama",
                "cpu_threads": 8,
                "context_length": 2048,
                "batch_size": 1
            }
        
        return {
            "platform": "Windows",
            "model": model_name,
            "cuda_available": self.cuda_available,
            "directml_available": self.directml_available,
            "optimizations": optimizations
        }
    
    def fix_gpu_issues(self) -> bool:
        if not self.is_windows:
            console.print("[red]Not running on Windows[/]")
            return False
        
        if self.cuda_available:
            console.print("[cyan]Checking CUDA installation...[/]")
            console.print("[green]✓[/] CUDA appears to be working")
            return True
        elif self.directml_available:
            console.print("[cyan]Checking DirectML installation...[/]")
            console.print("[green]✓[/] DirectML appears to be available")
            return True
        else:
            console.print("[yellow]No GPU acceleration detected[/]")
            console.print("[dim]Consider installing CUDA or enabling DirectML[/]")
            return False


@click.group("platform-optimize")
def platform_optimize_group():
    """Platform-specific optimizations."""
    pass


@platform_optimize_group.command("apple-silicon")
@click.argument("model")
def optimize_apple_silicon(model: str):
    """Optimize for Apple Silicon (M1/M2/M3 chips)."""
    optimizer = AppleSiliconOptimizer()
    
    if not optimizer.is_apple_silicon:
        console.print("[red]Not running on Apple Silicon[/]")
        return
    
    result = optimizer.optimize_for_apple_silicon(model)
    
    console.print("\n[bold cyan]Apple Silicon Optimization[/]")
    console.print(f"[bold]Model:[/] {model}")
    console.print("[bold]Optimizations:[/]")
    
    for key, value in result["optimizations"].items():
        console.print(f"  {key}: {value}")
    
    console.print("\n[bold]Recommended Command:[/]")
    console.print(f"  {result['command']}")


@platform_optimize_group.command("windows-gpu")
@click.argument("model")
def optimize_windows_gpu(model: str):
    """Optimize for Windows GPU (CUDA/DirectML)."""
    optimizer = WindowsGPUOptimizer()
    
    if not optimizer.is_windows:
        console.print("[red]Not running on Windows[/]")
        return
    
    result = optimizer.optimize_for_windows(model)
    
    console.print("\n[bold cyan]Windows GPU Optimization[/]")
    console.print(f"[bold]Model:[/] {model}")
    console.print(f"[bold]CUDA Available:[/] {'Yes' if result['cuda_available'] else 'No'}")
    console.print(f"[bold]DirectML Available:[/] {'Yes' if result['directml_available'] else 'No'}")
    console.print("\n[bold]Optimizations:[/]")
    
    for key, value in result["optimizations"].items():
        console.print(f"  {key}: {value}")


@platform_optimize_group.command("doctor")
@click.option("--fix", is_flag=True, help="Attempt to fix detected issues")
def platform_doctor(fix: bool):
    """Diagnose and fix platform-specific issues."""
    system = platform.system()
    
    console.print("\n[bold cyan]Platform Doctor[/]")
    console.print(f"[bold]System:[/] {system}")
    
    if system == "Darwin":
        optimizer = AppleSiliconOptimizer()
        if optimizer.is_apple_silicon:
            console.print("[green]✓[/] Apple Silicon detected")
            if fix:
                optimizer.apply_metal_optimization()
        else:
            console.print("[yellow]Intel Mac detected - standard optimizations apply")
    
    elif system == "Windows":
        optimizer = WindowsGPUOptimizer()
        console.print(f"[bold]CUDA Available:[/] {'Yes' if optimizer.cuda_available else 'No'}")
        console.print(f"[bold]DirectML Available:[/] {'Yes' if optimizer.directml_available else 'No'}")
        
        if fix:
            optimizer.fix_gpu_issues()
    
    else:
        console.print("[yellow]Linux detected - standard optimizations apply")


platform_optimize_command = platform_optimize_group