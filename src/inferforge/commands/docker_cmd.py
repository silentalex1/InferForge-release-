from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

console = Console()


class DockerManager:
    def __init__(self):
        self.docker_available = self._check_docker()
    
    def _check_docker(self) -> bool:
        try:
            import subprocess
            subprocess.run(["docker", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def create_dockerfile(self, model_name: str, output_dir: Path) -> bool:
        dockerfile_content = f"""FROM python:3.11-slim

WORKDIR /app

RUN pip install inferforge

COPY . /app

ENV INFERFORGE_MODEL={model_name}
ENV INFERFORGE_PORT=11435

EXPOSE 11435

CMD ["forge", "serve"]
"""
        
        dockerfile_path = output_dir / "Dockerfile"
        dockerfile_path.write_text(dockerfile_content)
        return True
    
    def create_docker_compose(self, model_name: str, output_dir: Path, replicas: int = 1) -> bool:
        compose_content = f"""version: '3.8'

services:
  inferforge:
    build: .
    ports:
      - "11435:11435"
    environment:
      - INFERFORGE_MODEL={model_name}
      - INFERFORGE_PORT=11435
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    deploy:
      replicas: {replicas}
    restart: unless-stopped

volumes:
  models:
  data:
"""
        
        compose_path = output_dir / "docker-compose.yml"
        compose_path.write_text(compose_content)
        return True
    
    def create_kubernetes_deployment(self, model_name: str, output_dir: Path, replicas: int = 3) -> bool:
        k8s_deployment = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: inferforge
  labels:
    app: inferforge
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: inferforge
  template:
    metadata:
      labels:
        app: inferforge
    spec:
      containers:
      - name: inferforge
        image: inferforge:latest
        ports:
        - containerPort: 11435
        env:
        - name: INFERFORGE_MODEL
          value: "{model_name}"
        - name: INFERFORGE_PORT
          value: "11435"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
---
apiVersion: v1
kind: Service
metadata:
  name: inferforge-service
spec:
  selector:
    app: inferforge
  ports:
  - protocol: TCP
    port: 11435
    targetPort: 11435
  type: LoadBalancer
"""
        
        k8s_path = output_dir / "k8s-deployment.yaml"
        k8s_path.write_text(k8s_deployment)
        return True
    
    def build_image(self, model_name: str, tag: str = "latest") -> bool:
        if not self.docker_available:
            console.print("[red]Docker not available[/]")
            return False
        
        try:
            import subprocess
            subprocess.run(
                ["docker", "build", "-t", f"inferforge:{tag}", "."],
                check=True
            )
            return True
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Docker build failed: {e}[/]")
            return False
    
    def run_container(self, model_name: str, tag: str = "latest", port: int = 11435) -> bool:
        if not self.docker_available:
            console.print("[red]Docker not available[/]")
            return False
        
        try:
            import subprocess
            subprocess.run(
                ["docker", "run", "-d", "-p", f"{port}:11435", 
                 "-e", f"INFERFORGE_MODEL={model_name}",
                 f"inferforge:{tag}"],
                check=True
            )
            return True
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Docker run failed: {e}[/]")
            return False


@click.group("docker")
def docker_group():
    """Docker and container support for deployment."""
    pass


@docker_group.command("build")
@click.argument("model")
@click.option("--tag", default="latest", help="Docker image tag")
@click.option("--output", "-o", help="Output directory for Docker files")
def docker_build(model: str, tag: str, output: str | None):
    """Build Docker image for a model."""
    manager = DockerManager()
    
    if not manager.docker_available:
        console.print("[red]Docker not available. Please install Docker first.[/]")
        return
    
    output_dir = Path(output) if output else Path.cwd() / "docker"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if manager.create_dockerfile(model, output_dir):
        console.print(f"[green]✓[/] Created Dockerfile in {output_dir}")
    
    if manager.build_image(model, tag):
        console.print(f"[green]✓[/] Built image: inferforge:{tag}")


@docker_group.command("run")
@click.argument("model")
@click.option("--tag", default="latest", help="Docker image tag")
@click.option("--port", type=int, default=11435, help="Port to expose")
def docker_run(model: str, tag: str, port: int):
    """Run InferForge in a Docker container."""
    manager = DockerManager()
    
    if manager.run_container(model, tag, port):
        console.print(f"[green]✓[/] Container running on port {port}")
        console.print(f"[dim]Access API at: http://localhost:{port}[/]")


@docker_group.command("compose")
@click.argument("model")
@click.option("--replicas", type=int, default=1, help="Number of replicas")
@click.option("--output", "-o", help="Output directory")
def docker_compose(model: str, replicas: int, output: str | None):
    """Create docker-compose configuration."""
    manager = DockerManager()
    
    output_dir = Path(output) if output else Path.cwd() / "docker"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if manager.create_docker_compose(model, output_dir, replicas):
        console.print(f"[green]✓[/] Created docker-compose.yml in {output_dir}")
        console.print(f"[dim]Run with: cd {output_dir} && docker-compose up -d[/]")


@docker_group.command("kubernetes")
@click.argument("model")
@click.option("--replicas", type=int, default=3, help="Number of replicas")
@click.option("--output", "-o", help="Output directory")
def docker_kubernetes(model: str, replicas: int, output: str | None):
    """Create Kubernetes deployment configuration."""
    manager = DockerManager()
    
    output_dir = Path(output) if output else Path.cwd() / "k8s"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if manager.create_kubernetes_deployment(model, output_dir, replicas):
        console.print(f"[green]✓[/] Created Kubernetes deployment in {output_dir}")
        console.print(f"[dim]Deploy with: kubectl apply -f {output_dir}/k8s-deployment.yaml[/]")


@docker_group.command("status")
def docker_status():
    """Check Docker and container status."""
    manager = DockerManager()
    
    console.print("\n[bold cyan]Docker Status[/]")
    console.print(f"[bold]Docker Available:[/] {'Yes' if manager.docker_available else 'No'}")
    
    if manager.docker_available:
        try:
            import subprocess
            result = subprocess.run(
                ["docker", "ps", "--filter", "ancestor=inferforge", "--format", "table {{.ID}}\t{{.Image}}\t{{.Status}}"],
                capture_output=True,
                text=True
            )
            
            if result.stdout.strip():
                console.print("\n[bold]Running InferForge Containers:[/]\n")
                console.print(result.stdout)
            else:
                console.print("\n[yellow]No InferForge containers running[/]")
        except subprocess.CalledProcessError:
            console.print("[red]Failed to get container status[/]")


docker_command = docker_group