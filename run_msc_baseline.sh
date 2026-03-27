#!/bin/bash
# MSC Baseline 评估脚本
# 用于评估 MSC 官方模型，作为我们系统的 baseline

# 激活环境
conda activate msc_eval

# 切换到 ParlAI 目录
cd /tmp/ParlAI

echo "=========================================="
echo "MSC Baseline 评估"
echo "=========================================="

echo ""
echo "【1】评估 MSC 3B 基础模型..."
echo "------------------------------------------"
parlai eval_model \
  -mf zoo:msc/msc3B_1024/model \
  -t msc \
  -dt valid \
  -bs 4 \
  --previous-persona-type raw_history \
  --metrics ppl,f1,token_acc

echo ""
echo "【2】评估 SumMem-MSC 3B (Fid-RAG)..."
echo "------------------------------------------"
parlai eval_model \
  -mf zoo:msc/summsc_fidrag3B/model \
  -t msc \
  -dt valid \
  -bs 4 \
  --previous-persona-type predsum_both \
  --metrics ppl,f1,token_acc

echo ""
echo "=========================================="
echo "评估完成"
echo "=========================================="
