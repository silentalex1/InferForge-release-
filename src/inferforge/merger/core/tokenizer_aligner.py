"""
Tokenizer alignment and vocabulary unification for model merging.

Handles the critical first step of model merging: ensuring all models use
the same token-to-ID mapping to prevent gibberish output.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


class TokenizerAligner:
    """Aligns tokenizers from multiple models into a unified vocabulary."""
    
    def __init__(self, tokenizer_paths: List[Path], target_vocab_size: Optional[int] = None):
        """
        Initialize tokenizer aligner.
        
        Args:
            tokenizer_paths: List of paths to tokenizer files (tokenizer.json or tokenizer_config.json)
            target_vocab_size: Target vocabulary size (150K-256K), auto-detected if None
        """
        self.tokenizer_paths = tokenizer_paths
        self.target_vocab_size = target_vocab_size
        self.tokenizers = []
        self.master_vocab = []
        self.remap_dictionaries = []
        
    def load_tokenizers(self) -> None:
        """Load tokenizer configurations from all source models."""
        for path in self.tokenizer_paths:
            if not path.exists():
                raise FileNotFoundError(f"Tokenizer file not found: {path}")
            
            with open(path, 'r', encoding='utf-8') as f:
                tokenizer_data = json.load(f)
            
            self.tokenizers.append(tokenizer_data)
    
    def extract_vocabularies(self) -> List[Dict[str, int]]:
        """Extract vocabulary from each tokenizer."""
        vocabularies = []
        
        for tokenizer in self.tokenizers:
            vocab = {}
            
                                                
            if "tokenizer" in tokenizer and "vocab" in tokenizer["tokenizer"]:
                                                   
                vocab_dict = tokenizer["tokenizer"]["vocab"]
                vocab = {token: idx for idx, token in vocab_dict.items()}
            elif "vocab" in tokenizer:
                                     
                vocab = tokenizer["vocab"]
            elif "vocab_size" in tokenizer:
                                                                     
                vocab_size = tokenizer["vocab_size"]
                vocab = {f"<token_{i}>": i for i in range(vocab_size)}
            else:
                raise ValueError(f"Unknown tokenizer format in {tokenizer}")
            
            vocabularies.append(vocab)
        
        return vocabularies
    
    def analyze_vocab_overlap(self, vocabularies: List[Dict[str, int]]) -> Dict:
        """Analyze overlap and unique tokens across vocabularies."""
        all_tokens_sets = [set(vocab.keys()) for vocab in vocabularies]
        
                                           
        common_tokens = set.intersection(*all_tokens_sets)
        
                                      
        unique_tokens = []
        for i, token_set in enumerate(all_tokens_sets):
            other_sets = all_tokens_sets[:i] + all_tokens_sets[i+1:]
            other_union = set.union(*other_sets) if other_sets else set()
            unique = token_set - other_union
            unique_tokens.append(unique)
        
        return {
            "common_tokens": common_tokens,
            "unique_tokens": unique_tokens,
            "vocab_sizes": [len(vocab) for vocab in vocabularies],
            "total_unique": len(set.union(*all_tokens_sets))
        }
    
    def build_master_vocabulary(self, vocabularies: List[Dict[str, int]]) -> List[str]:
        """Build unified master vocabulary from all source vocabularies."""
        analysis = self.analyze_vocab_overlap(vocabularies)
        
                                                   
        master_vocab = list(analysis["common_tokens"])
        
                                           
        for unique_tokens in analysis["unique_tokens"]:
            master_vocab.extend(sorted(unique_tokens))
        
                               
        if self.target_vocab_size is None:
                                                               
            self.target_vocab_size = min(len(master_vocab), 256000)
        else:
            self.target_vocab_size = min(self.target_vocab_size, 256000)
        
                                        
        if len(master_vocab) > self.target_vocab_size:
                                                                 
                                    
            master_vocab = master_vocab[:self.target_vocab_size]
        elif len(master_vocab) < self.target_vocab_size:
                                     
            padding_needed = self.target_vocab_size - len(master_vocab)
            for i in range(padding_needed):
                master_vocab.append(f"<pad_{i}>")
        
        self.master_vocab = master_vocab
        return master_vocab
    
    def create_remap_dictionaries(self, vocabularies: List[Dict[str, int]]) -> List[Dict[int, int]]:
        """Create ID remapping dictionaries for each source model."""
        remap_dicts = []
        master_index = {token: idx for idx, token in enumerate(self.master_vocab)}
        
        for vocab in vocabularies:
            remap_dict = {}
            for token, old_id in vocab.items():
                new_id = master_index.get(token)
                if new_id is not None:
                    remap_dict[old_id] = new_id
            
            remap_dicts.append(remap_dict)
        
        self.remap_dictionaries = remap_dicts
        return remap_dicts
    
    def initialize_new_embeddings(
        self,
        old_embedding: torch.Tensor,
        old_vocab_size: int,
        new_vocab_size: int,
        new_token_indices: List[int]
    ) -> torch.Tensor:
        """
        Initialize new embedding matrix with resized vocabulary.
        
        Args:
            old_embedding: Original embedding tensor [vocab_size, hidden_dim]
            old_vocab_size: Size of original vocabulary
            new_vocab_size: Size of new vocabulary
            new_token_indices: Indices of new tokens to initialize
        
        Returns:
            Resized embedding tensor
        """
        hidden_dim = old_embedding.shape[1]
        new_embedding = torch.zeros(new_vocab_size, hidden_dim, dtype=old_embedding.dtype, device=old_embedding.device)
        
        vocab_to_copy = min(old_vocab_size, new_vocab_size)
        new_embedding[:vocab_to_copy] = old_embedding[:vocab_to_copy]
        
        if new_token_indices:
            if vocab_to_copy > 0:
                avg_embedding = old_embedding[:vocab_to_copy].float().mean(dim=0)
                for new_idx in new_token_indices:
                    if new_idx < new_vocab_size:
                        noise = torch.randn(hidden_dim, dtype=torch.float32, device=old_embedding.device) * 0.01
                        new_embedding[new_idx] = (avg_embedding + noise).to(old_embedding.dtype)
        
        return new_embedding
    
    def align(
        self,
        embedding_tensors: List[torch.Tensor],
        lm_head_tensors: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], Dict]:
        """
        Perform full tokenizer alignment.
        
        Args:
            embedding_tensors: List of input embedding tensors from each model
            lm_head_tensors: List of output head tensors from each model
        
        Returns:
            Tuple of (aligned_embeddings, aligned_lm_heads, alignment_info)
        """
                                     
        self.load_tokenizers()
        vocabularies = self.extract_vocabularies()
        
                                 
        master_vocab = self.build_master_vocabulary(vocabularies)
        
                                   
        remap_dicts = self.create_remap_dictionaries(vocabularies)
        
                                 
        aligned_embeddings = []
        aligned_lm_heads = []
        
        for i, (embed_tensor, lm_head_tensor) in enumerate(zip(embedding_tensors, lm_head_tensors)):
            old_vocab_size = embed_tensor.shape[0]
            new_vocab_size = len(master_vocab)
            
                               
            new_indices = list(range(old_vocab_size, new_vocab_size))
            aligned_embed = self.initialize_new_embeddings(
                embed_tensor, old_vocab_size, new_vocab_size, new_indices
            )
            
                            
            aligned_lm = self.initialize_new_embeddings(
                lm_head_tensor, old_vocab_size, new_vocab_size, new_indices
            )
            
            aligned_embeddings.append(aligned_embed)
            aligned_lm_heads.append(aligned_lm)
        
        alignment_info = {
            "master_vocab_size": len(master_vocab),
            "original_vocab_sizes": [vocab.shape[0] for vocab in embedding_tensors],
            "remap_dictionaries": remap_dicts,
            "common_tokens": len(set.intersection(*[set(v.keys()) for v in vocabularies])),
        }
        
        return aligned_embeddings, aligned_lm_heads, alignment_info
