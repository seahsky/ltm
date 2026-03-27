"""
对话记忆系统 (Dialogue Memory System)

基于分层长期记忆架构的对话系统实现

核心组件:
- STM: 短期记忆，暂存当前对话上下文
- LTM: 分层长期记忆 (Fine/Mid/Coarse 三层)
- Consolidation: 记忆巩固，筛选关键经验
- PatternCluster: Mid 层聚类 + success_rate 追踪
- Reranking: 回复重排序机制
- Encoder: 文本编码器 (支持 Ollama, SentenceTransformer, OpenAI)
- Training: 预测模型 + 重要性评分器训练
- Agent: 对话 Agent
- MSC Loader: MSC 数据集加载器
"""

from .stm import ShortTermMemory, DialogueTurn
from .ltm import HierarchicalLTM, MemoryLayer, MemoryEntry
from .consolidation import DialogueConsolidation, DialogueSegment
from .pattern_cluster import PatternClusterer, PatternCluster, MidLayerMemory
from .reranking import (
    ResponseReranker,
    RerankingResult,
    ScoredResponse,
    Scorer,
    HistorySuccessScorer,
    MemorySimilarityScorer,
    CoherenceScorer,
    CandidateGenerator
)
from .encoder import (
    BaseEncoder,
    SentenceTransformerEncoder,
    OpenAIEncoder,
    MockEncoder,
    OllamaEncoder,
    get_encoder
)
from .ollama_client import (
    OllamaClient,
    OllamaConfig,
    get_ollama_client,
    get_ollama_encoder
)
from .train_predictor import (
    PredictionTrainer,
    PredictionMLP,
    PredictionDataset,
    train_predictor
)
from .train_scorer import (
    ScorerTrainer,
    ImportanceScorer,
    ImportanceDataset,
    TopicExtractor,
    train_scorer
)
from .agent import DialogueAgent, MemoryAugmentedGenerator
from .msc_loader import MSCDataLoader, Session, DialogueGroup, download_msc_dataset
from .msc_benchmark import (
    MSCEvaluator,
    PersonaRetrievalBenchmark,
    CrossSessionMemoryBenchmark,
    DialogueGenerationBenchmark,
    generate_msc_benchmark_report
)
from .perplexity import (
    PerplexityCalculator,
    MSCPerplexityEvaluator,
    compute_perplexity_with_model,
    get_perplexity_calculator,
    compute_ppl
)

__all__ = [
    "ShortTermMemory",
    "DialogueTurn",

    # LTM
    "HierarchicalLTM",
    "MemoryLayer",
    "MemoryEntry",

    # Consolidation
    "DialogueConsolidation",
    "DialogueSegment",

    # Pattern Clustering
    "PatternClusterer",
    "PatternCluster",
    "MidLayerMemory",

    # Reranking
    "ResponseReranker",
    "RerankingResult",
    "ScoredResponse",
    "Scorer",
    "HistorySuccessScorer",
    "MemorySimilarityScorer",
    "CoherenceScorer",
    "CandidateGenerator",

    # Encoder
    "BaseEncoder",
    "SentenceTransformerEncoder",
    "OpenAIEncoder",
    "MockEncoder",
    "OllamaEncoder",
    "get_encoder",

    # Ollama
    "OllamaClient",
    "OllamaConfig",
    "get_ollama_client",
    "get_ollama_encoder",

    # Training
    "PredictionTrainer",
    "PredictionMLP",
    "PredictionDataset",
    "train_predictor",
    "ScorerTrainer",
    "ImportanceScorer",
    "ImportanceDataset",
    "TopicExtractor",
    "train_scorer",

    # Agent
    "DialogueAgent",
    "MemoryAugmentedGenerator",

    # Data
    "MSCDataLoader",
    "Session",
    "DialogueGroup",
    "download_msc_dataset",

    # MSC Benchmark (NEW)
    "MSCEvaluator",
    "PersonaRetrievalBenchmark",
    "CrossSessionMemoryBenchmark",
    "DialogueGenerationBenchmark",
    "generate_msc_benchmark_report",
]

__version__ = "0.5.0"
