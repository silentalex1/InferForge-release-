"""Advanced dataset management for real training data integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

try:
    import pandas as pd
    from datasets import Dataset, DatasetDict, load_dataset
    from transformers import AutoTokenizer
except ImportError:
    raise ImportError("Dataset support requires: pip install datasets pandas")

class DatasetManager:
    """Advanced dataset manager for real training data integration."""
    
    def __init__(self, tokenizer: AutoTokenizer | None = None):
        self.tokenizer = tokenizer
        self.supported_formats = [
            "huggingface", "json", "jsonl", "csv", "txt", "parquet", "custom"
        ]
    
    def load_dataset(
        self,
        source: str,
        format_type: str = "auto",
        split: str = "train",
        text_field: str = "text",
        max_samples: int | None = None,
        **kwargs
    ) -> Dataset:
        """
        Load dataset from various sources with automatic format detection.
        
        Args:
            source: Dataset source (Hugging Face path, file path, or URL)
            format_type: Dataset format (auto, huggingface, json, jsonl, csv, txt, parquet, custom)
            split: Dataset split to load
            text_field: Field name containing text data
            max_samples: Maximum number of samples to load
            **kwargs: Additional arguments for dataset loading
        
        Returns:
            Hugging Face Dataset object
        """
        if format_type == "auto":
            format_type = self._detect_format(source)
        
        if format_type == "huggingface":
            return self._load_huggingface(source, split, max_samples, **kwargs)
        elif format_type == "json":
            return self._load_json(source, text_field, max_samples)
        elif format_type == "jsonl":
            return self._load_jsonl(source, text_field, max_samples)
        elif format_type == "csv":
            return self._load_csv(source, text_field, max_samples)
        elif format_type == "txt":
            return self._load_txt(source, max_samples)
        elif format_type == "parquet":
            return self._load_parquet(source, text_field, max_samples)
        elif format_type == "custom":
            return self._load_custom(source, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _detect_format(self, source: str) -> str:
        """Auto-detect dataset format from source."""
                                              
        if "/" in source and not Path(source).exists():
            return "huggingface"
        
        path = Path(source)
        if not path.exists():
            return "huggingface"                                                     
        
        suffix = path.suffix.lower()
        if suffix == ".json":
            return "json"
        elif suffix == ".jsonl":
            return "jsonl"
        elif suffix == ".csv":
            return "csv"
        elif suffix == ".txt":
            return "txt"
        elif suffix == ".parquet":
            return "parquet"
        else:
            return "json"                   
    
    def _load_huggingface(
        self, 
        source: str, 
        split: str = "train", 
        max_samples: int | None = None,
        **kwargs
    ) -> Dataset:
        """Load dataset from Hugging Face Hub."""
        try:
            dataset = load_dataset(source, split=split, **kwargs)
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            return dataset
        except Exception as e:
            raise ValueError(f"Failed to load Hugging Face dataset: {e}")
    
    def _load_json(self, source: str, text_field: str = "text", max_samples: int | None = None) -> Dataset:
        """Load dataset from JSON file."""
        path = Path(source)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, list):
            texts = [item.get(text_field, str(item)) for item in data]
        elif isinstance(data, dict):
            texts = data.get(text_field, [])
        else:
            raise ValueError("JSON must be a list or dict")
        
        if max_samples:
            texts = texts[:max_samples]
        
        return Dataset.from_dict({"text": texts})
    
    def _load_jsonl(self, source: str, text_field: str = "text", max_samples: int | None = None) -> Dataset:
        """Load dataset from JSONL file."""
        path = Path(source)
        texts = []
        
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    texts.append(item.get(text_field, str(item)))
        
        if max_samples:
            texts = texts[:max_samples]
        
        return Dataset.from_dict({"text": texts})
    
    def _load_csv(self, source: str, text_field: str = "text", max_samples: int | None = None) -> Dataset:
        """Load dataset from CSV file."""
        path = Path(source)
        df = pd.read_csv(path)
        
        if text_field not in df.columns:
                                   
            text_columns = df.select_dtypes(include=['object']).columns
            if len(text_columns) > 0:
                text_field = text_columns[0]
            else:
                raise ValueError("No text column found in CSV")
        
        texts = df[text_field].astype(str).tolist()
        
        if max_samples:
            texts = texts[:max_samples]
        
        return Dataset.from_dict({"text": texts})
    
    def _load_txt(self, source: str, max_samples: int | None = None) -> Dataset:
        """Load dataset from plain text file."""
        path = Path(source)
        with path.open("r", encoding="utf-8") as f:
            text = f.read()
        
                                         
        texts = [line.strip() for line in text.split("\n") if line.strip()]
        
        if max_samples:
            texts = texts[:max_samples]
        
        return Dataset.from_dict({"text": texts})
    
    def _load_parquet(self, source: str, text_field: str = "text", max_samples: int | None = None) -> Dataset:
        """Load dataset from Parquet file."""
        path = Path(source)
        df = pd.read_parquet(path)
        
        if text_field not in df.columns:
            text_columns = df.select_dtypes(include=['object']).columns
            if len(text_columns) > 0:
                text_field = text_columns[0]
            else:
                raise ValueError("No text column found in Parquet file")
        
        texts = df[text_field].astype(str).tolist()
        
        if max_samples:
            texts = texts[:max_samples]
        
        return Dataset.from_dict({"text": texts})
    
    def _load_custom(self, source: str, **kwargs) -> Dataset:
        """Load custom dataset using user-provided function."""
                                                              
                                                     
        raise NotImplementedError("Custom dataset loading requires user-provided function")
    
    def preprocess_dataset(
        self,
        dataset: Dataset,
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        text_field: str = "text",
        preprocessing_fn: Callable | None = None
    ) -> Dataset:
        """
        Preprocess dataset with tokenization and optional custom preprocessing.
        
        Args:
            dataset: Input dataset
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            text_field: Field containing text data
            preprocessing_fn: Optional custom preprocessing function
        
        Returns:
            Preprocessed and tokenized dataset
        """
        if preprocessing_fn:
            dataset = dataset.map(preprocessing_fn, batched=True)
        
        def tokenize_function(examples):
            return tokenizer(
                examples[text_field],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[text_field] if text_field in dataset.column_names else []
        )
        
        return tokenized_dataset
    
    def create_train_val_split(
        self,
        dataset: Dataset,
        validation_split: float = 0.1,
        test_split: float = 0.0,
        seed: int = 42
    ) -> DatasetDict:
        """
        Create train/validation/test splits from dataset.
        
        Args:
            dataset: Input dataset
            validation_split: Fraction for validation
            test_split: Fraction for test
            seed: Random seed for reproducibility
        
        Returns:
            DatasetDict with train, validation, and test splits
        """
        splits = dataset.train_test_split(
            test_size=validation_split + test_split,
            seed=seed
        )
        
        if test_split > 0:
                                                         
            test_size = test_split / (validation_split + test_split)
            test_splits = splits["test"].train_test_split(test_size=test_size, seed=seed)
            return DatasetDict({
                "train": splits["train"],
                "validation": test_splits["train"],
                "test": test_splits["test"]
            })
        else:
            return DatasetDict({
                "train": splits["train"],
                "validation": splits["test"]
            })
    
    def get_popular_datasets(self) -> dict[str, str]:
        """Get popular pre-built datasets for common tasks."""
        return {
            "general": [
                "wikitext/wikitext-103-raw-v1",
                "c4",
                "openwebtext",
                "pile",
            ],
            "code": [
                "bigcode/the-stack",
                "github-code",
                "codeparrot/codeparrot-clean",
            ],
            "instruction": [
                "databricks/databricks-dolly-15k",
                "Open-Orca/OpenOrca",
                "tatsu-lab/alpaca",
            ],
            "dialogue": [
                "Salesforce/dialogstudio",
                "Anthropic/hh-rlhf",
                "openai/webgpt-comparisons",
            ],
            "reasoning": [
                "bigbench/epistemic_reasoning",
                "commonsense_qa",
                "aqua_rat",
            ]
        }
    
    def create_synthetic_dataset(
        self,
        task: str = "general",
        num_samples: int = 1000,
        complexity: str = "medium"
    ) -> Dataset:
        """
        Create synthetic dataset for testing or pre-training.
        
        Args:
            task: Type of task (general, code, instruction, dialogue, reasoning)
            num_samples: Number of samples to generate
            complexity: Complexity level (simple, medium, complex)
        
        Returns:
            Synthetic dataset
        """
        generators = {
            "general": self._generate_general_text,
            "code": self._generate_code_samples,
            "instruction": self._generate_instructions,
            "dialogue": self._generate_dialogues,
            "reasoning": self._generate_reasoning_examples
        }
        
        generator = generators.get(task, self._generate_general_text)
        texts = generator(num_samples, complexity)
        
        return Dataset.from_dict({"text": texts})
    
    def _generate_general_text(self, num_samples: int, complexity: str) -> list[str]:
        """Generate general text samples."""
        texts = []
        for i in range(num_samples):
            if complexity == "simple":
                text = f"This is sample text number {i}. It contains basic information."
            elif complexity == "medium":
                text = f"Sample text {i}: This paragraph contains more detailed information about various topics, including technology, science, and daily life events."
            else:           
                text = f"Complex sample {i}: In the realm of advanced artificial intelligence, neural networks process vast amounts of data through sophisticated algorithms, enabling machines to understand language, recognize patterns, and make informed decisions across diverse domains from healthcare to autonomous systems."
            texts.append(text)
        return texts
    
    def _generate_code_samples(self, num_samples: int, complexity: str) -> list[str]:
        """Generate code samples."""
        code_templates = [
            "def function_{0}(param):\n    return param * 2",
            "class Class_{0}:\n    def method(self):\n        pass",
            "import library_{0}\nresult = library_{0}.process()",
            "for i in range({0}):\n    print(i)",
            "if condition_{0}:\n    execute()\nelse:\n    alternative()"
        ]
        
        texts = []
        for i in range(num_samples):
            template = code_templates[i % len(code_templates)]
            texts.append(template.format(i))
        return texts
    
    def _generate_instructions(self, num_samples: int, complexity: str) -> list[str]:
        """Generate instruction-following samples."""
        instructions = [
            lambda i: f"Write a function that {['sorts', 'filters', 'transforms', 'analyzes', 'processes'][i % 5]} data",
            lambda i: f"Explain the concept of {['machine learning', 'neural networks', 'deep learning', 'AI ethics', 'data science'][i % 5]}",
            lambda i: f"Create a {['simple', 'efficient', 'robust', 'scalable', 'maintainable'][i % 5]} algorithm for task {i}",
            lambda i: f"Summarize the following text about topic {i}",
            lambda i: f"Generate code for {['web scraping', 'data analysis', 'machine learning', 'web development', 'database management'][i % 5]}",
        ]
        
        texts = []
        for i in range(num_samples):
            instruction = instructions[i % len(instructions)](i)
            texts.append(f"Instruction: {instruction}\nResponse: [Detailed response for {instruction}]")
        return texts
    
    def _generate_dialogues(self, num_samples: int, complexity: str) -> list[str]:
        """Generate dialogue samples."""
        dialogues = [
            lambda i: f"User: What is the weather like?\nAssistant: The weather is {['sunny', 'cloudy', 'rainy', 'windy', 'snowy'][i % 5]} today.",
            lambda i: f"User: How do I {['install', 'configure', 'use', 'troubleshoot', 'optimize'][i % 5]} this software?\nAssistant: Here are the steps...",
            lambda i: f"User: Can you explain {['quantum computing', 'blockchain', 'machine learning', 'artificial intelligence', 'neural networks'][i % 5]}?\nAssistant: Let me explain the concept...",
        ]
        
        texts = []
        for i in range(num_samples):
            texts.append(dialogues[i % len(dialogues)](i))
        return texts
    
    def _generate_reasoning_examples(self, num_samples: int, complexity: str) -> list[str]:
        """Generate reasoning examples."""
        reasoning_templates = [
            lambda i: f"Problem: Solve {i + 1} + {i + 2}\nReasoning: Add the numbers together\nSolution: {2*i + 3}",
            lambda i: "Question: If all A are B and all B are C, are all A C?\nReasoning: By transitive property, yes\nAnswer: Yes",
            lambda i: f"Task: Determine the pattern in sequence {i}, {i+1}, {i+2}, {i+3}\nReasoning: Each number increases by 1\nNext: {i+4}",
        ]
        
        texts = []
        for i in range(num_samples):
            texts.append(reasoning_templates[i % len(reasoning_templates)](i))
        return texts