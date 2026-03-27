"""
Ollama 客户端
连接本地 Ollama 服务，支持 embedding 和生成

使用方法:
    from dialogue_memory.ollama_client import OllamaClient

    client = OllamaClient(model="llama3.2:3b")

    # Embedding
    emb = client.embed("Hello world")

    # 生成
    response = client.generate("Say hello:")
"""

import requests
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class OllamaConfig:
    """Ollama 配置"""
    host: str = "http://localhost:11434"
    model: str = "llama3.2:3b"
    embed_model: str = None  # 如果为 None，使用 model
    timeout: int = 60

    @property
    def effective_embed_model(self) -> str:
        return self.embed_model or self.model


class OllamaClient:
    """
    Ollama 客户端

    支持:
    1. 文本 embedding (通过 /api/embeddings)
    2. 文本生成 (通过 /api/generate)
    3. 对话生成 (通过 /api/chat)
    """

    def __init__(self,
                 model: str = "llama3.2:3b",
                 host: str = "http://localhost:11434",
                 embed_model: str = None,
                 timeout: int = 60):
        self.config = OllamaConfig(
            host=host,
            model=model,
            embed_model=embed_model,
            timeout=timeout
        )
        self._embed_dim = None

    @property
    def embed_dim(self) -> int:
        """获取 embedding 维度（延迟获取）"""
        if self._embed_dim is None:
            # 通过一次调用来获取维度
            test_emb = self.embed("test")
            self._embed_dim = len(test_emb)
        return self._embed_dim

    def embed(self, text: str) -> np.ndarray:
        """
        获取文本的 embedding

        Args:
            text: 输入文本

        Returns:
            embedding 向量 (numpy array)
        """
        url = f"{self.config.host}/api/embeddings"
        payload = {
            "model": self.config.effective_embed_model,
            "prompt": text
        }

        try:
            response = requests.post(url, json=payload, timeout=self.config.timeout)
            response.raise_for_status()
            data = response.json()
            return np.array(data["embedding"], dtype=np.float32)
        except Exception as e:
            print(f"[Ollama] Embedding error: {e}")
            # 返回零向量作为 fallback
            return np.zeros(3072, dtype=np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        批量获取 embedding

        Args:
            texts: 文本列表

        Returns:
            embedding 矩阵 [N, D]
        """
        embeddings = []
        for text in texts:
            emb = self.embed(text)
            embeddings.append(emb)
        return np.array(embeddings)

    def generate(self,
                 prompt: str,
                 system: str = None,
                 max_tokens: int = 256,
                 temperature: float = 0.7,
                 stream: bool = False) -> str:
        """
        生成文本

        Args:
            prompt: 输入提示
            system: 系统提示（可选）
            max_tokens: 最大生成 token 数
            temperature: 温度参数
            stream: 是否流式输出

        Returns:
            生成的文本
        """
        url = f"{self.config.host}/api/generate"
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature
            }
        }

        if system:
            payload["system"] = system

        try:
            response = requests.post(url, json=payload, timeout=self.config.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except Exception as e:
            print(f"[Ollama] Generate error: {e}")
            return ""

    def chat(self,
             messages: List[Dict[str, str]],
             max_tokens: int = 256,
             temperature: float = 0.7) -> str:
        """
        对话生成

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}, ...]
            max_tokens: 最大生成 token 数
            temperature: 温度参数

        Returns:
            助手的回复
        """
        url = f"{self.config.host}/api/chat"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=self.config.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except Exception as e:
            print(f"[Ollama] Chat error: {e}")
            return ""

    def score_response(self,
                       user_input: str,
                       candidate_response: str,
                       context: str = None) -> float:
        """
        使用 LLM 评估回复质量

        Args:
            user_input: 用户输入
            candidate_response: 候选回复
            context: 对话上下文（可选）

        Returns:
            质量分数 [0, 1]
        """
        prompt = f"""评估以下对话回复的质量。请只输出一个0到1之间的数字，不要有其他文字。

用户输入: {user_input}
候选回复: {candidate_response}

评估标准:
- 相关性: 回复是否与用户输入相关
- 连贯性: 回复是否自然流畅
- 信息量: 回复是否有价值

分数 (0-1):"""

        try:
            response = self.generate(prompt, max_tokens=10, temperature=0.1)
            # 解析分数
            response = response.strip()
            # 尝试提取数字
            import re
            numbers = re.findall(r'[0-9]*\.?[0-9]+', response)
            if numbers:
                score = float(numbers[0])
                return min(1.0, max(0.0, score))
            return 0.5
        except:
            return 0.5

    def compute_surprise(self,
                        text: str,
                        context: str = None) -> float:
        """
        计算 surprise (预测误差)

        使用 LLM 判断文本的意外程度

        Args:
            text: 目标文本
            context: 上下文

        Returns:
            surprise 分数 [0, 1]
        """
        prompt = f"""判断以下内容在给定上下文中的意外程度。请只输出一个0到1之间的数字，0表示完全预期，1表示非常意外。

上下文: {context or "无"}
内容: {text}

意外程度 (0-1):"""

        try:
            response = self.generate(prompt, max_tokens=10, temperature=0.1)
            import re
            numbers = re.findall(r'[0-9]*\.?[0-9]+', response)
            if numbers:
                score = float(numbers[0])
                return min(1.0, max(0.0, score))
            return 0.5
        except:
            return 0.5

    def check_connection(self) -> bool:
        """检查连接是否正常"""
        try:
            response = requests.get(f"{self.config.host}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def list_models(self) -> List[str]:
        """列出可用模型"""
        try:
            response = requests.get(f"{self.config.host}/api/tags", timeout=5)
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except:
            return []


class OllamaEncoder:
    """
    使用 Ollama 的 Embedding 接口作为编码器

    兼容 BaseEncoder 接口
    """

    def __init__(self, client: OllamaClient = None, model: str = "llama3.2:3b"):
        if client is None:
            client = OllamaClient(model=model)
        self.client = client
        self._embed_dim = None

    @property
    def embed_dim(self) -> int:
        if self._embed_dim is None:
            self._embed_dim = self.client.embed_dim
        return self._embed_dim

    def encode(self, text: str) -> np.ndarray:
        return self.client.embed(text)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        return self.client.embed_batch(texts)


# 便捷函数
def get_ollama_client(model: str = "llama3.2:3b") -> OllamaClient:
    """获取 Ollama 客户端"""
    return OllamaClient(model=model)


def get_ollama_encoder(model: str = "llama3.2:3b") -> OllamaEncoder:
    """获取 Ollama 编码器"""
    return OllamaEncoder(model=model)
