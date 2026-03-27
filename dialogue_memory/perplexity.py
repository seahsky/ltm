"""
困惑度计算工具
用于评估对话生成质量

MSC 原始评估方式:
- 使用生成模型计算 PPL(response | persona + history)
- 在验证集上计算平均困惑度

我们的实现:
- 使用 GPT-2 作为评估模型
- 计算条件困惑度
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from transformers import GPT2LMHeadModel, GPT2Tokenizer


@dataclass
class PerplexityResult:
    """困惑度计算结果"""
    text: str
    perplexity: float
    token_count: int


class PerplexityCalculator:
    """
    困惑度计算器

    使用 GPT-2 模型计算文本的困惑度

    使用方法:
        calculator = PerplexityCalculator()

        # 计算单段文本
        ppl = calculator.compute("Hello world")

        # 批量计算
        results = calculator.compute_batch(["text1", "text2"])
    """

    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        if self._model is None:
            self._model = GPT2LMHeadModel.from_pretrained(self.model_name)
            self._model.eval()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def compute(self, text: str) -> PerplexityResult:
        """
        计算单段文本的困惑度

        Args:
            text: 输入文本

        Returns:
            PerplexityResult
        """
        inputs = self.tokenizer(text, return_tensors='pt')

        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            ppl = torch.exp(loss).item()

        return PerplexityResult(
            text=text,
            perplexity=ppl,
            token_count=inputs['input_ids'].shape[1]
        )

    def compute_batch(self, texts: List[str]) -> List[PerplexityResult]:
        """
        批量计算困惑度

        Args:
            texts: 文本列表

        Returns:
            PerplexityResult 列表
        """
        results = []
        for text in texts:
            result = self.compute(text)
            results.append(result)
        return results

    def compute_conditional_ppl(self,
                                context: str,
                                target: str) -> float:
        """
        计算条件困惑度

        PPL(target | context)

        Args:
            context: 条件文本 (如 persona + history)
            target: 目标文本 (如生成的回复)

        Returns:
            条件困惑度
        """
        # 合并 context 和 target
        full_text = context + " " + target

        # Tokenize
        inputs = self.tokenizer(full_text, return_tensors='pt')
        context_len = len(self.tokenizer(context)['input_ids'][0])

        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs['input_ids'])

            # 只计算 target 部分的 loss
            logits = outputs.logits[0, context_len-1:-1, :]  # target 部分的 logits
            labels = inputs['input_ids'][0, context_len:]  # target 部分的 labels

            # 计算交叉熵
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            ppl = torch.exp(loss).item()

        return ppl

    def compute_msc_perplexity(self,
                               persona: List[str],
                               history: List[str],
                               response: str) -> float:
        """
        计算 MSC 风格的困惑度

        PPL(response | persona + history)

        Args:
            persona: Persona 句子列表
            history: 历史对话列表
            response: 生成的回复

        Returns:
            困惑度
        """
        # 构建上下文
        context = " ".join(persona) + " " + " ".join(history)

        return self.compute_conditional_ppl(context, response)


class MSCPerplexityEvaluator:
    """
    MSC 困惑度评估器

    评估对话生成质量，支持:
    1. 无条件困惑度
    2. 条件困惑度 (PPL | persona + history)
    3. MSC 风格评估
    """

    def __init__(self, model_name: str = "gpt2"):
        self.calculator = PerplexityCalculator(model_name)

    def evaluate_dialogue(self,
                         persona: List[str],
                         history: List[str],
                         generated_response: str,
                         ground_truth: str = None) -> Dict[str, float]:
        """
        评估单个对话

        Args:
            persona: Persona 句子
            history: 历史对话
            generated_response: 生成的回复
            ground_truth: 真实的回复 (可选)

        Returns:
            评估结果
        """
        results = {}

        # 1. 生成回复的困惑度
        results['generated_ppl'] = self.calculator.compute_msc_perplexity(
            persona, history, generated_response
        )

        # 2. 如果有 ground truth，计算真实回复的困惑度
        if ground_truth:
            results['ground_truth_ppl'] = self.calculator.compute_msc_perplexity(
                persona, history, ground_truth
            )

        # 3. 无条件困惑度
        results['unconditional_ppl'] = self.calculator.compute(generated_response).perplexity

        return results

    def evaluate_batch(self,
                      dialogues: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        批量评估对话

        Args:
            dialogues: 对话列表，每个包含:
                - persona: List[str]
                - history: List[str]
                - response: str
                - ground_truth: str (optional)

        Returns:
            平均评估结果
        """
        all_results = {
            'generated_ppl': [],
            'ground_truth_ppl': [],
            'unconditional_ppl': []
        }

        for dialogue in dialogues:
            result = self.evaluate_dialogue(
                persona=dialogue.get('persona', []),
                history=dialogue.get('history', []),
                generated_response=dialogue['response'],
                ground_truth=dialogue.get('ground_truth')
            )

            for key, value in result.items():
                all_results[key].append(value)

        # 计算平均值
        avg_results = {}
        for key, values in all_results.items():
            if values:
                avg_results[key] = np.mean(values)

        return avg_results


def compute_perplexity_with_model(model, tokenizer, text: str) -> float:
    """
    使用给定模型计算困惑度

    Args:
        model: 语言模型
        tokenizer: tokenizer
        text: 输入文本

    Returns:
        困惑度
    """
    inputs = tokenizer(text, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        ppl = torch.exp(loss).item()

    return ppl


# 便捷函数
def get_perplexity_calculator(model_name: str = "gpt2") -> PerplexityCalculator:
    """获取困惑度计算器"""
    return PerplexityCalculator(model_name)


def compute_ppl(text: str, model_name: str = "gpt2") -> float:
    """快速计算困惑度"""
    calculator = PerplexityCalculator(model_name)
    return calculator.compute(text).perplexity
