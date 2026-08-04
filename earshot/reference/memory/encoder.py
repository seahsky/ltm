"""
对话编码器 (Dialogue Encoder)
将文本编码为向量表示
"""

import numpy as np
from typing import List, Union, Optional
from abc import ABC, abstractmethod


class BaseEncoder(ABC):
    """编码器基类"""

    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        pass

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """批量编码"""
        return np.array([self.encode(t) for t in texts])

    @property
    def embed_dim(self) -> int:
        """embedding 维度（子类应重写）"""
        return 768


class OllamaEncoder(BaseEncoder):
    """
    使用 Ollama 的 Embedding 接口

    需要本地运行 Ollama 服务
    """

    def __init__(self, model: str = "llama3.2:3b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
        self._embed_dim = None
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from .ollama_client import OllamaClient
            self._client = OllamaClient(model=self.model, host=self.host)
        return self._client

    @property
    def embed_dim(self) -> int:
        if self._embed_dim is None:
            # 通过一次调用获取维度
            test = self.encode("test")
            self._embed_dim = len(test)
        return self._embed_dim

    def encode(self, text: str) -> np.ndarray:
        return self.client.embed(text)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        return self.client.embed_batch(texts)


class SentenceTransformerEncoder(BaseEncoder):
    """
    使用 SentenceTransformer 进行编码

    推荐模型:
    - all-MiniLM-L6-v2: 快速，384 维
    - all-mpnet-base-v2: 更准确，768 维
    - paraphrase-multilingual-mpnet-base-v2: 多语言支持
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)

    @property
    def embed_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()


class OpenAIEncoder(BaseEncoder):
    """
    使用 OpenAI embeddings API
    """

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def encode(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        return np.array(response.data[0].embedding)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        return np.array([d.embedding for d in response.data])

    @property
    def embed_dim(self) -> int:
        # text-embedding-3-small: 1536
        # text-embedding-3-large: 3072
        dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        return dims.get(self.model_name, 1536)


class MockEncoder(BaseEncoder):
    """
    模拟编码器（用于测试）
    返回随机向量
    """

    def __init__(self, embed_dim: int = 768, seed: int = 42):
        self.embed_dim = embed_dim
        self.rng = np.random.RandomState(seed)
        self._cache = {}

    def encode(self, text: str) -> np.ndarray:
        # 使用文本 hash 作为随机种子，保证相同文本产生相同编码
        text_hash = hash(text)
        if text_hash not in self._cache:
            # 基于 hash 生成确定性随机向量
            rng = np.random.RandomState(text_hash % (2**31))
            self._cache[text_hash] = rng.randn(self.embed_dim).astype('float32')
        return self._cache[text_hash]


def get_encoder(encoder_type: str = "sentence_transformer", **kwargs) -> BaseEncoder:
    """
    工厂函数: 获取编码器实例

    Args:
        encoder_type: "sentence_transformer", "openai", "ollama", "mock"
        **kwargs: 传递给编码器的参数

    Examples:
        # Ollama (本地 LLM)
        encoder = get_encoder("ollama", model="llama3.2:3b")

        # SentenceTransformer
        encoder = get_encoder("sentence_transformer", model_name="all-MiniLM-L6-v2")

        # Mock (测试用)
        encoder = get_encoder("mock", embed_dim=768)
    """
    encoders = {
        "sentence_transformer": SentenceTransformerEncoder,
        "openai": OpenAIEncoder,
        "ollama": OllamaEncoder,
        "mock": MockEncoder
    }

    if encoder_type not in encoders:
        raise ValueError(f"Unknown encoder type: {encoder_type}. Available: {list(encoders.keys())}")

    return encoders[encoder_type](**kwargs)
