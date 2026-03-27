#!/bin/bash
#===============================================================================
# MSC 轻量级对比评估脚本
#
# 只对比 MSC 3B baseline vs 我们的 LTM 系统
# 跳过 SumMem-MSC (太占 GPU 内存)
#===============================================================================

source /home/ec2-user/miniconda3/etc/profile.d/conda.sh
conda activate msc_eval
cd /tmp/ParlAI

NUM_SAMPLES=${1:-20}  # 默认 20 个样本快速验证

echo "=========================================="
echo "MSC 对比评估 (快速版)"
echo "样本数: $NUM_SAMPLES"
echo "=========================================="

mkdir -p /home/ec2-user/studies/ltm_agent/eval_results

# 1. MSC 3B Baseline
echo ""
echo "【1】MSC 3B Baseline"
echo "------------------------------------------"
parlai eval_model \
  -mf zoo:msc/msc3B_1024/model \
  -t msc \
  -dt valid \
  -bs 4 \
  --previous-persona-type raw_history \
  --metrics ppl,f1,token_acc \
  -ne $NUM_SAMPLES \
  2>&1 | tee /home/ec2-user/studies/ltm_agent/eval_results/msc3b_baseline.log

# 2. 我们的 LTM + MSC 3B
echo ""
echo "【2】我们的 LTM 系统 + MSC 3B"
echo "------------------------------------------"
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

# 汇总
echo ""
echo "=========================================="
echo "结果汇总"
echo "=========================================="

for log in /home/ec2-user/studies/ltm_agent/eval_results/*.log; do
    name=$(basename "$log" .log)
    echo ""
    echo "【$name】"
    tail -5 "$log" 2>/dev/null | grep -E "(ppl|f1|token_acc|accuracy)" || echo "  查看完整日志: $log"
done

echo ""
echo "=========================================="
