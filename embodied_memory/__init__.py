"""
embodied_memory — Habitat / HM3D proof-of-life slice for the LTM-Embodied Agent.

Wires the existing dialogue_memory LTM modules (STM, consolidation, hierarchical
LTM, pattern clustering, reranking) onto a Habitat ObjectNav loop. Independent
of the dialogue / MSC entrypoints.
"""

from .episode_source import EpisodeSource, Step, Episode
from .perception import CLIPKeyframeEncoder, SemanticCaptioner, Keyframe
from .frontier_planner import FrontierPlanner, FrontierCandidate, OccupancyGrid
from .memory_bridge import EmbodiedMemoryBridge, EmbodiedRecord
from .episode_runner import EpisodeRunner, RunSummary

__all__ = [
    "EpisodeSource",
    "Step",
    "Episode",
    "CLIPKeyframeEncoder",
    "SemanticCaptioner",
    "Keyframe",
    "FrontierPlanner",
    "FrontierCandidate",
    "OccupancyGrid",
    "EmbodiedMemoryBridge",
    "EmbodiedRecord",
    "EpisodeRunner",
    "RunSummary",
]

__version__ = "0.1.0-pol"
