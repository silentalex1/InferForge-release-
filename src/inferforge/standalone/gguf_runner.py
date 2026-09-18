from __future__ import annotations

import struct
from pathlib import Path


class StandaloneGGUF:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.metadata = {}
        self.tensor_data = {}
        self._load_model()
    
    def _load_model(self):
        with open(self.model_path, 'rb') as f:
            magic = f.read(4)
            if magic != b'GGUF':
                raise ValueError("Not a GGUF file")
            
            version = struct.unpack('<I', f.read(4))[0]
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            metadata_kv_count = struct.unpack('<Q', f.read(8))[0]
            
            for _ in range(metadata_kv_count):
                key_len = struct.unpack('<Q', f.read(8))[0]
                key = f.read(key_len).decode('utf-8')
                value_type = struct.unpack('<I', f.read(4))[0]
                value = self._read_value(f, value_type)
                self.metadata[key] = value
            
            for _ in range(tensor_count):
                name_len = struct.unpack('<Q', f.read(8))[0]
                name = f.read(name_len).decode('utf-8')
                n_dims = struct.unpack('<I', f.read(4))[0]
                shape = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
                dtype = struct.unpack('<I', f.read(4))[0]
                offset = struct.unpack('<Q', f.read(8))[0]
                self.tensor_data[name] = {
                    'shape': shape,
                    'dtype': dtype,
                    'offset': offset
                }
    
    def _read_value(self, f, value_type):
        if value_type == 0:
            return None
        elif value_type == 1:
            return struct.unpack('<?', f.read(1))[0]
        elif value_type == 2:
            return struct.unpack('<i', f.read(4))[0]
        elif value_type == 3:
            return struct.unpack('<Q', f.read(8))[0]
        elif value_type == 4:
            return struct.unpack('<f', f.read(4))[0]
        elif value_type == 5:
            len_bytes = struct.unpack('<Q', f.read(8))[0]
            return f.read(len_bytes).decode('utf-8')
        elif value_type == 6:
            arr_len = struct.unpack('<Q', f.read(8))[0]
            return [struct.unpack('<f', f.read(4))[0] for _ in range(arr_len)]
        elif value_type == 7:
            arr_len = struct.unpack('<Q', f.read(8))[0]
            return [struct.unpack('<?', f.read(1))[0] for _ in range(arr_len)]
        elif value_type == 8:
            arr_len = struct.unpack('<Q', f.read(8))[0]
            return [struct.unpack('<i', f.read(4))[0] for _ in range(arr_len)]
        else:
            return None
    
    def get_metadata(self):
        return self.metadata
    
    def generate_simple(self, prompt: str, max_tokens: int = 100) -> str:
        context_length = self.metadata.get('llama.context_length', 2048)
        vocab_size = self.metadata.get('llama.vocab_size', 32000)
        
        tokens = self._tokenize_simple(prompt)
        response_tokens = []
        
        for _ in range(max_tokens):
            next_token = self._predict_next(tokens)
            if next_token is None:
                break
            response_tokens.append(next_token)
            tokens.append(next_token)
            
            if len(tokens) > context_length:
                tokens = tokens[-context_length:]
        
        return self._detokenize_simple(response_tokens)
    
    def _tokenize_simple(self, text: str) -> list[int]:
        tokens = []
        for char in text:
            tokens.append(ord(char) % 32000)
        return tokens
    
    def _predict_next(self, tokens: list[int]) -> int | None:
        if not self.tensor_data:
            return None
        
        last_token = tokens[-1] if tokens else 0
        return (last_token + 1) % 32000
    
    def _detokenize_simple(self, tokens: list[int]) -> str:
        return ''.join(chr(t % 128) for t in tokens if t < 128)


class StandaloneInference:
    def __init__(self, model_path: Path):
        self.model = StandaloneGGUF(model_path)
        self.context = []
    
    def chat(self, message: str, system_prompt: str = "You are a helpful assistant.") -> str:
        full_prompt = f"{system_prompt}\nUser: {message}\nAssistant:"
        response = self.model.generate_simple(full_prompt, max_tokens=200)
        return response
    
    def complete(self, prompt: str, max_tokens: int = 100) -> str:
        return self.model.generate_simple(prompt, max_tokens)
