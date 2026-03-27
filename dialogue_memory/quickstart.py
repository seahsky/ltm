"""
快速入门脚本
演示对话记忆系统的基本用法
"""

import sys
sys.path.insert(0, "/home/ec2-user/studies/ltm_agent")

from dialogue_memory import (
    DialogueAgent,
    get_encoder,
    MSCDataLoader,
    MSCEvaluator,
    print_evaluation_report
)


def demo_basic_usage():
    """演示基本用法"""
    print("=" * 60)
    print("对话记忆系统 - 基本用法演示")
    print("=" * 60)

    # 1. 初始化组件
    print("\n1. 初始化编码器和 Agent...")
    encoder = get_encoder("mock", embed_dim=384)  # 使用 mock 编码器进行快速测试
    agent = DialogueAgent(encoder=encoder)

    # 2. 开始对话
    print("\n2. 开始对话...")
    agent.start_dialogue(dialogue_id=1, session_id=0)

    # 3. 处理用户输入
    print("\n3. 处理用户输入...")
    user_inputs = [
        "你好，我喜欢打篮球",
        "你有什么爱好吗？",
        "我最喜欢的球队是湖人队"
    ]

    for user_input in user_inputs:
        print(f"\n用户: {user_input}")
        result = agent.process_user_input(user_input)
        print(f"  STM 轮数: {result['stm_turns']}")
        print(f"  检索上下文:\n{result['retrieval_context'][:200]}..." if result['retrieval_context'] else "  (无检索结果)")

    # 4. 巩固记忆
    print("\n4. 巩固当前 session 的记忆...")
    agent.consolidate_session(persona_info=["喜欢打篮球", "湖人队球迷"])

    # 5. 查看记忆状态
    print("\n5. 记忆系统状态:")
    stats = agent.get_memory_stats()
    print(f"  STM: {stats['stm']['turns']} 轮")
    print(f"  LTM: {stats['ltm']}")

    # 6. 测试检索
    print("\n6. 测试记忆检索...")
    query = "我喜欢什么运动？"
    query_emb = encoder.encode(query)
    results = agent.ltm.multi_scale_search(query_emb, top_k_per_layer=2)

    print(f"  查询: {query}")
    for level, entries in results.items():
        if entries:
            print(f"  {level} 层检索结果:")
            for entry, dist in entries:
                print(f"    - {entry.content[:50]}... (距离: {dist:.4f})")


def demo_msc_dataset():
    """演示 MSC 数据集使用"""
    print("\n" + "=" * 60)
    print("MSC 数据集演示")
    print("=" * 60)

    # 加载数据
    print("\n1. 加载 MSC 数据集...")
    loader = MSCDataLoader()
    stats = loader.stats()
    print(f"  数据集统计:")
    for split, info in stats.items():
        print(f"    {split}: {info['dialogue_groups']} 组对话, {info['total_sessions']} sessions")

    # 查看一个示例
    print("\n2. 查看示例对话...")
    first_group = list(loader.train.values())[0]
    print(f"  对话 ID: {first_group.dialogue_id}")
    print(f"  Session 数量: {len(first_group.sessions)}")

    for session in first_group.sessions:
        print(f"\n  Session {session.session_id}:")
        print(f"    Persona 1: {session.persona1[:2]}...")
        print(f"    对话轮数: {len(session.dialogue)}")
        for i, (speaker, utterance) in enumerate(session.get_turns()):
            if i < 3:  # 只显示前 3 轮
                print(f"    {speaker}: {utterance[:50]}...")


def demo_evaluation():
    """演示评估流程"""
    print("\n" + "=" * 60)
    print("评估演示")
    print("=" * 60)

    print("\n1. 初始化组件...")
    encoder = get_encoder("mock", embed_dim=384)
    agent = DialogueAgent(encoder=encoder)
    loader = MSCDataLoader()

    print("\n2. 运行评估...")
    evaluator = MSCEvaluator(agent, loader, encoder)
    results = evaluator.evaluate_all(split="val", max_dialogues=5)

    print_evaluation_report(results)


def demo_cross_session_memory():
    """演示跨 Session 记忆"""
    print("\n" + "=" * 60)
    print("跨 Session 记忆演示")
    print("=" * 60)

    encoder = get_encoder("mock", embed_dim=384)
    agent = DialogueAgent(encoder=encoder)

    # Session 0
    print("\n【Session 0】")
    agent.start_dialogue(dialogue_id=1, session_id=0)

    inputs_s0 = ["我叫小明", "我喜欢吃苹果", "我是学生"]
    for inp in inputs_s0:
        agent.process_user_input(inp)
        print(f"  用户: {inp}")

    agent.consolidate_session(persona_info=["名字叫小明", "喜欢吃苹果", "是学生"])
    print(f"  记忆状态: {agent.get_memory_stats()['ltm']}")

    # Session 1 (模拟一段时间后)
    print("\n【Session 1】(模拟新的对话)")
    agent.stm.clear()  # 清空 STM
    agent.start_dialogue(dialogue_id=1, session_id=1)

    # 测试是否能回忆起 Session 0 的信息
    print("\n  测试检索 Session 0 的记忆:")
    queries = ["我叫什么名字？", "我喜欢吃什么？", "我的职业是什么？"]

    for query in queries:
        query_emb = encoder.encode(query)
        results = agent.ltm.search("fine", query_emb, top_k=1)
        print(f"    查询: {query}")
        if results:
            print(f"    检索到: {results[0][0].content[:60]}...")
        else:
            print(f"    (未检索到相关记忆)")


if __name__ == "__main__":
    print("\n对话记忆系统 - 快速入门\n")

    # 基本用法
    demo_basic_usage()

    # MSC 数据集
    demo_msc_dataset()

    # 跨 Session 记忆
    demo_cross_session_memory()

    # 评估
    # demo_evaluation()  # 可选：运行完整评估

    print("\n✅ 演示完成!")
