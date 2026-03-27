#!/bin/bash
#===============================================================================
# MSC 综合评估脚本
#
# 支持三种评估配置:
#   1. MSC 3B Baseline     - 原始基础模型
#   2. SumMem-MSC Baseline  - 官方记忆增强模型 (Fid-RAG)
#   3. 我们的 LTM 系统      - 分层长期记忆 + MSC 3B
#
# 评估指标: Perplexity, F1, Token Accuracy
#===============================================================================

# 激活环境
source /home/ec2-user/miniconda3/etc/profile.d/conda.sh
conda activate msc_eval

# 切换到 ParlAI 目录
cd /tmp/ParlAI

# 配置
NUM_SAMPLES=${1:-100}  # 默认评估 100 个样本
DEVICE="cuda"           # GPU 设备

echo "=========================================="
echo "MSC 综合评估"
echo "=========================================="
echo "样本数量: $NUM_SAMPLES"
echo "设备: $DEVICE"
echo ""

# 创建输出目录
mkdir -p /home/ec2-user/studies/ltm_agent/eval_results

#===============================================================================
# 1. MSC 3B Baseline
#===============================================================================
echo ""
echo "【1】评估 MSC 3B Baseline"
echo "------------------------------------------"
echo "配置: 无记忆增强，raw_history"
echo ""

parlai eval_model \
  -mf zoo:msc/msc3B_1024/model \
  -t msc \
  -dt valid \
  -bs 4 \
  --previous-persona-type raw_history \
  --metrics ppl,f1,token_acc \
  -ne $NUM_SAMPLES \
  2>&1 | tee /home/ec2-user/studies/ltm_agent/eval_results/msc3b_baseline.log

echo ""
echo "MSC 3B Baseline 评估完成"
echo "=========================================="

#===============================================================================
# 2. SumMem-MSC Baseline (Fid-RAG)
#===============================================================================
echo ""
echo "【2】评估 SumMem-MSC Baseline (Fid-RAG)"
echo "------------------------------------------"
echo "配置: 官方记忆增强模型，predsum_both"
echo "注意: 此模型较大，可能需要更多 GPU 内存"
echo ""

parlai eval_model \
  -mf zoo:msc/summsc_fidrag3B/model \
  -t msc \
  -dt valid \
  -bs 2 \
  --previous-persona-type predsum_both \
  --metrics ppl,f1,token_acc \
  -ne $NUM_SAMPLES \
  2>&1 | tee /home/ec2-user/studies/ltm_agent/eval_results/summsc_baseline.log

echo ""
echo "SumMem-MSC Baseline 评估完成"
echo "=========================================="

#===============================================================================
# 3. 我们的 LTM 系统 + MSC 3B
#===============================================================================
echo ""
echo "【3】评估我们的 LTM 系统 + MSC 3B"
echo "------------------------------------------"
echo "配置: 分层长期记忆 (Fine/Mid/Coarse)"
echo ""

# 使用 Mutator 注入 LTM 记忆
# 注意: 如果 mutator 有问题，可以回退到原始 MSC 3B
parlai eval_model \
  -mf zoo:msc/msc3B_1024/model \
  -t msc \
  -dt valid \
  -bs 4 \
  --previous-persona-type raw_history \
  --mutators ltm_augment \
  --metrics ppl,f1,token_acc \
  -ne $NUM_SAMPLES \
  2>&1 | tee /home/ec2-user/studies/ltm_agent/eval_results/ltm_ours.log

echo ""
echo "我们的 LTM 系统评估完成"
echo "=========================================="

#===============================================================================
# 结果汇总
#===============================================================================
echo ""
echo "=========================================="
echo "评估完成 - 结果汇总"
echo "=========================================="

# 从日志中提取关键指标
extract_metrics() {
    local log_file=$1
    local model_name=$2

    echo ""
    echo "【$model_name】"
    if [ -f "$log_file" ]; then
        # 提取最后一行指标（汇总结果）
        tail -20 "$log_file" | grep -E "ppl|f1|token_acc" | tail -5
    else
        echo "  日志文件不存在: $log_file"
    fi
}

extract_metrics /home/ec2-user/studies/ltm_agent/eval_results/msc3b_baseline.log "MSC 3B Baseline"
extract_metrics /home/ec2-user/studies/ltm_agent/eval_results/summsc_baseline.log "SumMem-MSC Baseline"
extract_metrics /home/ec2-user/studies/ltm_agent/eval_results/ltm_ours.log "我们的 LTM 系统"

echo ""
echo "=========================================="
echo "详细日志保存在:"
echo "  - MSC 3B:     eval_results/msc3b_baseline.log"
echo "  - SumMem-MSC: eval_results/summsc_baseline.log"
echo "  - 我们的 LTM:  eval_results/ltm_ours.log"
echo "=========================================="
