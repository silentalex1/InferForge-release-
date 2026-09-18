from __future__ import annotations

from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from inferforge.commands.pull_cmd import _detect_source, _estimate_hardware_fit
from inferforge.importers.huggingface import HuggingFaceImporter


def test_detect_source_routing():
    # Direct URLs
    assert _detect_source("https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf")[0] == "url"
    assert _detect_source("http://example.com/models/custom-model.gguf")[0] == "url"
    
    # Hugging Face URLs and Model IDs
    assert _detect_source("https://huggingface.co/meta-llama/Llama-3.1-8B")[0] == "huggingface"
    assert _detect_source("TheBloke/Mistral-7B-Instruct-v0.2-GGUF")[0] == "huggingface"
    assert _detect_source("unsloth/Meta-Llama-3.1-8B-Instruct")[0] == "huggingface"
    
    # Ollama names and URLs
    assert _detect_source("llama3.1:8b")[0] == "ollama"
    assert _detect_source("qwen2.5-coder:7b")[0] == "ollama"
    assert _detect_source("https://ollama.com/library/llama3.1")[0] == "ollama"


def test_hardware_fit_estimation(capsys):
    # Should run smoothly without exception and print estimation
    _estimate_hardware_fit("llama3.1:8b", quant="q4_k_m")
    captured = capsys.readouterr()
    assert "8.0B" in captured.out or "8.0B" in captured.err or "needed" in captured.out or "needed" in captured.err


def test_smart_gguf_selection():
    importer = HuggingFaceImporter()
    # Mock list_repo_files and hf_hub_download
    mock_files = [
        "README.md",
        "config.json",
        "llama-3-8b.Q2_K.gguf",
        "llama-3-8b.Q4_K_M.gguf",
        "llama-3-8b.Q5_K_M.gguf",
        "llama-3-8b.Q8_0.gguf",
    ]
    with patch("huggingface_hub.list_repo_files", return_value=mock_files):
        with patch("huggingface_hub.hf_hub_download", return_value="/tmp/llama-3-8b.Q4_K_M.gguf") as mock_download:
            result = importer.pull_model("fake/repo", quant="q4_k_m")
            mock_download.assert_called_once()
            call_kwargs = mock_download.call_args.kwargs
            assert call_kwargs["filename"] == "llama-3-8b.Q4_K_M.gguf"
