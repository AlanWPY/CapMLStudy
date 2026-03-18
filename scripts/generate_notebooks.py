from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
NOTEBOOK_DIR.mkdir(exist_ok=True)


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def teaching_cells(
    intuition: str,
    architecture: str,
    training: str,
    usage: str,
    pitfalls: str,
):
    return [
        md(
            f"""
            ## 先建立直觉

            {dedent(intuition).strip()}
            """
        ),
        md(
            f"""
            ## 架构是怎么工作的

            {dedent(architecture).strip()}
            """
        ),
        md(
            f"""
            ## 训练时到底发生了什么

            {dedent(training).strip()}
            """
        ),
        md(
            f"""
            ## 什么时候该用它

            {dedent(usage).strip()}
            """
        ),
        md(
            f"""
            ## 最常见的误区

            {dedent(pitfalls).strip()}
            """
        ),
    ]


def professional_cells(
    model_title: str,
    definition: str,
    io_spec: str,
    architecture: str,
    objective: str,
    complexity: str,
    comparison: str,
):
    return [
        md(
            f"""
            ## {model_title}的形式化定义

            {dedent(definition).strip()}
            """
        ),
        md(
            f"""
            ## 输入、输出与参数化方式

            {dedent(io_spec).strip()}
            """
        ),
        md(
            f"""
            ## 结构分解与信息流

            {dedent(architecture).strip()}
            """
        ),
        md(
            f"""
            ## 优化目标与训练机制

            {dedent(objective).strip()}
            """
        ),
        md(
            f"""
            ## 计算复杂度、统计性质与工程代价

            {dedent(complexity).strip()}
            """
        ),
        md(
            f"""
            ## 与相邻模型的差异

            {dedent(comparison).strip()}
            """
        ),
    ]


def architecture_diagram_cell(kind: str):
    diagrams = {
        "linear": r"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)

for idx, x in enumerate([1.0, 1.8, 2.6, 3.4]):
    circle = plt.Circle((x, 2), 0.22, color="#4C78A8")
    ax.add_patch(circle)
    ax.text(x, 1.45, f"x{idx+1}", ha="center", fontsize=12)

rect = plt.Rectangle((4.4, 1.25), 1.4, 1.5, facecolor="#F58518", edgecolor="black")
ax.add_patch(rect)
ax.text(5.1, 2.0, "加权求和\nw^T x + b", ha="center", va="center", fontsize=12)

rect2 = plt.Rectangle((6.6, 1.25), 1.2, 1.5, facecolor="#54A24B", edgecolor="black")
ax.add_patch(rect2)
ax.text(7.2, 2.0, "输出", ha="center", va="center", fontsize=12)
ax.text(7.2, 1.3, "ŷ 或 p(y|x)", ha="center", va="bottom", fontsize=11)

for x in [1.22, 2.02, 2.82, 3.62]:
    ax.annotate("", xy=(4.4, 2), xytext=(x, 2), arrowprops=dict(arrowstyle="->", lw=1.8))
ax.annotate("", xy=(6.6, 2), xytext=(5.8, 2), arrowprops=dict(arrowstyle="->", lw=1.8))

ax.set_title("线性模型结构示意图")
plt.show()
""",
        "tree": r"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

nodes = {
    "root": (5, 8.5, "根节点\nx_j <= t ?"),
    "left": (3, 5.8, "左子树\n继续切分"),
    "right": (7, 5.8, "右子树\n继续切分"),
    "ll": (1.8, 2.5, "叶节点\n预测 A"),
    "lr": (4.2, 2.5, "叶节点\n预测 B"),
    "rl": (5.8, 2.5, "叶节点\n预测 C"),
    "rr": (8.2, 2.5, "叶节点\n预测 D"),
}
for key, (x, y, text) in nodes.items():
    w, h = (1.7, 1.0) if "叶节点" in text else (2.0, 1.1)
    rect = plt.Rectangle((x - w/2, y - h/2), w, h, facecolor="#72B7B2", edgecolor="black")
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=11)

edges = [("root", "left"), ("root", "right"), ("left", "ll"), ("left", "lr"), ("right", "rl"), ("right", "rr")]
for a, b in edges:
    ax.annotate("", xy=nodes[b][:2], xytext=nodes[a][:2], arrowprops=dict(arrowstyle="->", lw=1.6))
ax.set_title("决策树的层次切分结构")
plt.show()
""",
        "mlp": r"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis("off")
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)

layers = {
    "输入层": (1.5, [6.5, 5.0, 3.5, 2.0]),
    "隐藏层1": (4.5, [6.8, 5.6, 4.4, 3.2, 2.0]),
    "隐藏层2": (7.5, [6.2, 4.8, 3.4, 2.0]),
    "输出层": (10.2, [5.6, 4.0, 2.4]),
}
for name, (x, ys) in layers.items():
    for y in ys:
        ax.add_patch(plt.Circle((x, y), 0.18, color="#4C78A8"))
    ax.text(x, 7.4, name, ha="center", fontsize=12)

layer_values = list(layers.values())
for i in range(len(layer_values) - 1):
    x1, ys1 = layer_values[i]
    x2, ys2 = layer_values[i + 1]
    for y1 in ys1:
        for y2 in ys2:
            ax.plot([x1, x2], [y1, y2], color="gray", alpha=0.25, lw=0.8)

ax.text(6.0, 0.8, "每层执行: 线性变换 -> 激活函数 -> 表示更新", ha="center", fontsize=12)
ax.set_title("多层感知机的前馈结构")
plt.show()
""",
        "cnn": r"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(13, 5))
ax.axis("off")
ax.set_xlim(0, 13)
ax.set_ylim(0, 5)

stages = [
    (1.2, 2.5, 1.2, 1.8, "#4C78A8", "输入图像\nH x W"),
    (3.5, 2.5, 1.4, 1.8, "#F58518", "卷积 + ReLU\n局部特征"),
    (5.9, 2.5, 1.4, 1.6, "#E45756", "池化\n降采样"),
    (8.2, 2.5, 1.4, 1.8, "#72B7B2", "卷积块\n高阶特征"),
    (10.4, 2.5, 1.2, 1.6, "#54A24B", "Flatten"),
    (12.0, 2.5, 1.1, 1.5, "#B279A2", "分类头"),
]
for x, y, w, h, color, text in stages:
    rect = plt.Rectangle((x - w/2, y - h/2), w, h, facecolor=color, edgecolor="black")
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=11)
for i in range(len(stages) - 1):
    ax.annotate("", xy=(stages[i+1][0] - stages[i+1][2]/2, 2.5), xytext=(stages[i][0] + stages[i][2]/2, 2.5),
                arrowprops=dict(arrowstyle="->", lw=1.8))
ax.set_title("卷积神经网络的数据流")
plt.show()
""",
        "rnn": r"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(13, 4.8))
ax.axis("off")
ax.set_xlim(0, 13)
ax.set_ylim(0, 5)

xs = [1.5, 4.0, 6.5, 9.0]
for i, x in enumerate(xs, start=1):
    rect = plt.Rectangle((x - 0.6, 2.0), 1.2, 1.0, facecolor="#4C78A8", edgecolor="black")
    ax.add_patch(rect)
    ax.text(x, 2.5, f"RNN Cell\nh{i}", ha="center", va="center", fontsize=11)
    ax.text(x, 1.3, f"x{i}", ha="center", fontsize=11)
    ax.annotate("", xy=(x, 2.0), xytext=(x, 1.55), arrowprops=dict(arrowstyle="->", lw=1.5))
for a, b in zip(xs[:-1], xs[1:]):
    ax.annotate("", xy=(b - 0.62, 2.5), xytext=(a + 0.62, 2.5), arrowprops=dict(arrowstyle="->", lw=1.8))
ax.text(10.8, 3.8, "隐藏状态沿时间递推", fontsize=12)
ax.set_title("RNN / LSTM / GRU 的时间展开视图")
plt.show()
""",
        "transformer": r"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis("off")
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)

blocks = [
    (6, 7.0, 2.4, 0.8, "#4C78A8", "Token Embedding + Positional Encoding"),
    (6, 5.7, 2.2, 0.9, "#F58518", "Multi-Head Self-Attention"),
    (6, 4.4, 2.0, 0.8, "#E45756", "Residual + Norm"),
    (6, 3.1, 2.0, 0.9, "#72B7B2", "Feed-Forward Network"),
    (6, 1.8, 2.0, 0.8, "#54A24B", "Residual + Norm"),
    (6, 0.6, 1.8, 0.8, "#B279A2", "LM Head / Output"),
]
for x, y, w, h, color, text in blocks:
    rect = plt.Rectangle((x - w/2, y - h/2), w, h, facecolor=color, edgecolor="black")
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=11)
for i in range(len(blocks) - 1):
    ax.annotate("", xy=(6, blocks[i+1][1] + blocks[i+1][3]/2), xytext=(6, blocks[i][1] - blocks[i][3]/2),
                arrowprops=dict(arrowstyle="->", lw=1.8))
ax.set_title("Decoder-only Transformer Block 的层级结构")
plt.show()
""",
        "moe": r"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(13, 6))
ax.axis("off")
ax.set_xlim(0, 13)
ax.set_ylim(0, 6)

ax.add_patch(plt.Rectangle((0.8, 2.3), 1.5, 1.2, facecolor="#4C78A8", edgecolor="black"))
ax.text(1.55, 2.9, "Token 表示\nh", ha="center", va="center", fontsize=11)

ax.add_patch(plt.Rectangle((3.2, 2.2), 1.8, 1.4, facecolor="#F58518", edgecolor="black"))
ax.text(4.1, 2.9, "Router\nsoftmax(W_r h)", ha="center", va="center", fontsize=11)

experts = [(7.0, 4.7, "Expert 1"), (7.0, 3.4, "Expert 2"), (7.0, 2.1, "Expert 3"), (7.0, 0.8, "Expert 4")]
for x, y, text in experts:
    ax.add_patch(plt.Rectangle((x - 0.9, y - 0.4), 1.8, 0.8, facecolor="#72B7B2", edgecolor="black"))
    ax.text(x, y, text, ha="center", va="center", fontsize=11)

ax.add_patch(plt.Rectangle((10.7, 2.3), 1.5, 1.2, facecolor="#54A24B", edgecolor="black"))
ax.text(11.45, 2.9, "加权聚合\nΣ p_e E_e(h)", ha="center", va="center", fontsize=11)

ax.annotate("", xy=(3.2, 2.9), xytext=(2.3, 2.9), arrowprops=dict(arrowstyle="->", lw=1.8))
for x, y, _ in experts:
    ax.annotate("", xy=(x - 0.9, y), xytext=(5.0, 2.9), arrowprops=dict(arrowstyle="->", lw=1.2, alpha=0.7))
    ax.annotate("", xy=(10.7, 2.9), xytext=(x + 0.9, y), arrowprops=dict(arrowstyle="->", lw=1.2, alpha=0.7))

ax.text(6.0, 5.6, "Top-k 路由只激活少数专家", fontsize=12)
ax.set_title("MoE 前馈层的稀疏路由结构")
plt.show()
""",
        "lora": r"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis("off")
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)

ax.add_patch(plt.Rectangle((1.0, 1.9), 1.3, 1.2, facecolor="#4C78A8", edgecolor="black"))
ax.text(1.65, 2.5, "输入 x", ha="center", va="center", fontsize=11)

ax.add_patch(plt.Rectangle((4.0, 3.0), 1.8, 1.0, facecolor="#F58518", edgecolor="black"))
ax.text(4.9, 3.5, "冻结主权重 W", ha="center", va="center", fontsize=11)

ax.add_patch(plt.Rectangle((4.0, 1.1), 1.1, 0.8, facecolor="#E45756", edgecolor="black"))
ax.text(4.55, 1.5, "A", ha="center", va="center", fontsize=11)
ax.add_patch(plt.Rectangle((6.0, 1.1), 1.1, 0.8, facecolor="#72B7B2", edgecolor="black"))
ax.text(6.55, 1.5, "B", ha="center", va="center", fontsize=11)
ax.text(5.25, 0.55, "低秩增量 ΔW = BA", ha="center", fontsize=11)

ax.add_patch(plt.Circle((8.4, 2.5), 0.35, color="#54A24B"))
ax.text(8.4, 2.5, "+", ha="center", va="center", fontsize=16, color="white")

ax.add_patch(plt.Rectangle((10.0, 1.9), 1.3, 1.2, facecolor="#B279A2", edgecolor="black"))
ax.text(10.65, 2.5, "输出 y", ha="center", va="center", fontsize=11)

ax.annotate("", xy=(4.0, 3.5), xytext=(2.3, 2.5), arrowprops=dict(arrowstyle="->", lw=1.8))
ax.annotate("", xy=(4.0, 1.5), xytext=(2.3, 2.5), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(6.0, 1.5), xytext=(5.1, 1.5), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(8.05, 2.7), xytext=(5.8, 3.4), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(8.05, 2.3), xytext=(7.1, 1.5), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(10.0, 2.5), xytext=(8.75, 2.5), arrowprops=dict(arrowstyle="->", lw=1.8))
ax.set_title("LoRA 的低秩增量结构")
plt.show()
""",
        "mdp": r"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis("off")
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)

boxes = [
    (1.8, 2.5, 2.1, 1.2, "#4C78A8", "Agent\npolicy / value"),
    (6.0, 2.5, 2.4, 1.3, "#F58518", "Environment\nP(s', r | s, a)"),
    (10.2, 3.6, 1.8, 0.9, "#54A24B", "reward r_t"),
    (10.2, 1.4, 1.8, 0.9, "#E45756", "next state s_{t+1}"),
]
for x, y, w, h, color, text in boxes:
    rect = plt.Rectangle((x - w / 2, y - h / 2), w, h, facecolor=color, edgecolor="black")
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=11)

ax.annotate("", xy=(4.8, 2.5), xytext=(2.85, 2.5), arrowprops=dict(arrowstyle="->", lw=1.8))
ax.text(3.9, 2.82, "action a_t", ha="center", fontsize=11)
ax.annotate("", xy=(2.85, 3.05), xytext=(8.0, 3.6), arrowprops=dict(arrowstyle="->", lw=1.8))
ax.text(5.6, 3.9, "reward feedback", ha="center", fontsize=11)
ax.annotate("", xy=(2.85, 1.95), xytext=(8.0, 1.4), arrowprops=dict(arrowstyle="->", lw=1.8))
ax.text(5.6, 0.95, "state transition", ha="center", fontsize=11)
ax.set_title("Agent-Environment Interaction in Reinforcement Learning")
plt.show()
""",
        "bellman": r"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis("off")
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)

states = [(2.0, 2.5, "state s"), (6.0, 2.5, "next state s'"), (10.0, 2.5, "bootstrap target")]
for x, y, text in states:
    ax.add_patch(plt.Circle((x, y), 0.6, color="#4C78A8"))
    ax.text(x, y, text, ha="center", va="center", color="white", fontsize=11)

ax.annotate("", xy=(5.35, 2.5), xytext=(2.65, 2.5), arrowprops=dict(arrowstyle="->", lw=1.8))
ax.text(4.0, 2.85, "take action a", ha="center", fontsize=11)
ax.annotate("", xy=(9.35, 2.5), xytext=(6.65, 2.5), arrowprops=dict(arrowstyle="->", lw=1.8))
ax.text(8.0, 2.95, "r + gamma * V(s')", ha="center", fontsize=11)
ax.text(6.0, 1.1, "Bellman update: current estimate is corrected by a one-step target", ha="center", fontsize=12)
ax.set_title("Bellman Backup")
plt.show()
""",
        "dqn": r"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(13, 5.5))
ax.axis("off")
ax.set_xlim(0, 13)
ax.set_ylim(0, 5.5)

blocks = [
    (1.6, 2.8, 1.8, 1.0, "#4C78A8", "state s"),
    (4.0, 2.8, 2.2, 1.1, "#F58518", "online Q-net"),
    (7.1, 4.0, 2.4, 1.0, "#54A24B", "replay buffer"),
    (7.1, 1.6, 2.4, 1.0, "#E45756", "target Q-net"),
    (10.8, 2.8, 2.2, 1.2, "#72B7B2", "TD target\nr + gamma max Q_target"),
]
for x, y, w, h, color, text in blocks:
    rect = plt.Rectangle((x - w / 2, y - h / 2), w, h, facecolor=color, edgecolor="black")
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=11)

ax.annotate("", xy=(2.9, 2.8), xytext=(2.5, 2.8), arrowprops=dict(arrowstyle="->", lw=1.8))
ax.annotate("", xy=(5.3, 3.5), xytext=(5.9, 3.8), arrowprops=dict(arrowstyle="->", lw=1.6))
ax.annotate("", xy=(5.3, 2.1), xytext=(5.9, 1.8), arrowprops=dict(arrowstyle="->", lw=1.6))
ax.annotate("", xy=(9.6, 2.8), xytext=(8.3, 2.8), arrowprops=dict(arrowstyle="->", lw=1.8))
ax.text(6.9, 5.0, "sample mini-batch", ha="center", fontsize=11)
ax.text(7.1, 0.55, "periodic / Polyak sync", ha="center", fontsize=11)
ax.set_title("Deep Q-Network with Replay Buffer and Target Network")
plt.show()
""",
        "actor_critic": r"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(13, 5.5))
ax.axis("off")
ax.set_xlim(0, 13)
ax.set_ylim(0, 5.5)

blocks = [
    (1.4, 2.75, 1.6, 1.0, "#4C78A8", "state s"),
    (4.2, 3.7, 2.2, 1.0, "#F58518", "actor\npi(a|s)"),
    (4.2, 1.8, 2.2, 1.0, "#54A24B", "critic\nV(s) / Q(s,a)"),
    (7.2, 2.75, 2.0, 1.0, "#E45756", "action a"),
    (10.2, 2.75, 2.3, 1.2, "#72B7B2", "advantage / TD error"),
]
for x, y, w, h, color, text in blocks:
    rect = plt.Rectangle((x - w / 2, y - h / 2), w, h, facecolor=color, edgecolor="black")
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=11)

ax.annotate("", xy=(3.1, 3.7), xytext=(2.2, 2.95), arrowprops=dict(arrowstyle="->", lw=1.8))
ax.annotate("", xy=(3.1, 1.8), xytext=(2.2, 2.55), arrowprops=dict(arrowstyle="->", lw=1.8))
ax.annotate("", xy=(6.2, 2.75), xytext=(5.3, 3.35), arrowprops=dict(arrowstyle="->", lw=1.8))
ax.annotate("", xy=(9.0, 2.75), xytext=(8.2, 2.75), arrowprops=dict(arrowstyle="->", lw=1.8))
ax.annotate("", xy=(5.3, 2.15), xytext=(9.1, 2.25), arrowprops=dict(arrowstyle="->", lw=1.6))
ax.text(8.2, 4.45, "critic supplies the learning signal for the actor", ha="center", fontsize=11)
ax.set_title("Actor-Critic Information Flow")
plt.show()
""",
        "ppo": r"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis("off")
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)

ax.add_patch(plt.Rectangle((0.8, 1.8), 2.2, 1.4, facecolor="#4C78A8", edgecolor="black"))
ax.text(1.9, 2.5, "old policy rollout\npi_old", ha="center", va="center", fontsize=11)
ax.add_patch(plt.Rectangle((4.0, 1.8), 2.0, 1.4, facecolor="#F58518", edgecolor="black"))
ax.text(5.0, 2.5, "ratio r = pi / pi_old", ha="center", va="center", fontsize=11)
ax.add_patch(plt.Rectangle((7.0, 1.8), 2.2, 1.4, facecolor="#E45756", edgecolor="black"))
ax.text(8.1, 2.5, "clip to [1-eps, 1+eps]", ha="center", va="center", fontsize=11)
ax.add_patch(plt.Rectangle((10.2, 1.8), 2.0, 1.4, facecolor="#54A24B", edgecolor="black"))
ax.text(11.2, 2.5, "stable update", ha="center", va="center", fontsize=11)

for start, end in [(3.0, 4.0), (6.0, 7.0), (9.2, 10.2)]:
    ax.annotate("", xy=(end, 2.5), xytext=(start, 2.5), arrowprops=dict(arrowstyle="->", lw=1.8))

ax.text(6.0, 4.0, "PPO keeps policy improvement but limits destructive update steps", ha="center", fontsize=12)
ax.set_title("Clipped PPO Objective")
plt.show()
""",
        "rlhf_pipeline": r"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(14, 5))
ax.axis("off")
ax.set_xlim(0, 14)
ax.set_ylim(0, 5)

stages = [
    (1.2, 2.5, 1.8, 1.1, "#4C78A8", "pretraining"),
    (4.0, 2.5, 1.8, 1.1, "#F58518", "SFT"),
    (6.8, 2.5, 2.0, 1.1, "#E45756", "preference / reward"),
    (9.8, 2.5, 2.1, 1.1, "#72B7B2", "RL or PO"),
    (12.5, 2.5, 1.8, 1.1, "#54A24B", "aligned model"),
]
for x, y, w, h, color, text in stages:
    rect = plt.Rectangle((x - w / 2, y - h / 2), w, h, facecolor=color, edgecolor="black")
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=11)
for i in range(len(stages) - 1):
    ax.annotate("", xy=(stages[i + 1][0] - stages[i + 1][2] / 2, 2.5), xytext=(stages[i][0] + stages[i][2] / 2, 2.5),
                arrowprops=dict(arrowstyle="->", lw=1.8))
ax.text(6.8, 4.0, "offline PO uses preference pairs; online RL uses sampled rollouts plus reward / rule signals", ha="center", fontsize=11)
ax.set_title("LLM Post-Training Pipeline")
plt.show()
""",
        "group_po": r"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(13, 5.5))
ax.axis("off")
ax.set_xlim(0, 13)
ax.set_ylim(0, 5.5)

ax.add_patch(plt.Rectangle((0.9, 2.2), 1.8, 1.2, facecolor="#4C78A8", edgecolor="black"))
ax.text(1.8, 2.8, "prompt x", ha="center", va="center", fontsize=11)

group_y = [4.4, 3.2, 2.0, 0.8]
for idx, y in enumerate(group_y, start=1):
    ax.add_patch(plt.Rectangle((3.5, y), 2.2, 0.7, facecolor="#F58518", edgecolor="black"))
    ax.text(4.6, y + 0.35, f"sampled response y{idx}", ha="center", va="center", fontsize=10)

ax.add_patch(plt.Rectangle((7.4, 2.2), 2.3, 1.2, facecolor="#E45756", edgecolor="black"))
ax.text(8.55, 2.8, "group reward\nnormalize -> A_i", ha="center", va="center", fontsize=11)

ax.add_patch(plt.Rectangle((11.1, 2.2), 2.0, 1.2, facecolor="#54A24B", edgecolor="black"))
ax.text(12.1, 2.8, "policy update", ha="center", va="center", fontsize=11)

ax.annotate("", xy=(3.5, 2.8), xytext=(2.7, 2.8), arrowprops=dict(arrowstyle="->", lw=1.8))
for y in group_y:
    ax.annotate("", xy=(7.4, 2.8), xytext=(5.7, y + 0.35), arrowprops=dict(arrowstyle="->", lw=1.4))
ax.annotate("", xy=(11.1, 2.8), xytext=(9.7, 2.8), arrowprops=dict(arrowstyle="->", lw=1.8))
ax.text(6.6, 4.9, "GRPO / GSPO / SAPO / GiGPO start from grouped rollouts but differ in credit assignment and ratio control", ha="center", fontsize=10.5)
ax.set_title("Grouped Policy Optimization for LLM Post-Training")
plt.show()
""",
    }
    return code(diagrams[kind])


def write_notebook(filename: str, cells: list):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    }
    with (NOTEBOOK_DIR / filename).open("w", encoding="utf-8") as f:
        nbf.write(nb, f)


def common_style_code(include_torch: bool = False) -> str:
    torch_imports = """
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
""" if include_torch else ""

    return (
        f"""
# 兼容当前 Windows 环境：把临时目录固定到用户目录下的 ASCII 路径，
# 避免 scipy / sklearn 在中文工作目录下寻找临时文件时报错。
from pathlib import Path
import os
import warnings

temp_root = Path(os.environ.get("ML_DL_TMP", str(Path.home() / ".ml_dl_notebook_tmp")))
temp_root.mkdir(exist_ok=True)
os.environ["TMP"] = str(temp_root)
os.environ["TEMP"] = str(temp_root)

warnings.filterwarnings("ignore")

import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
random.seed(42)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]

{torch_imports}

print("临时目录:", temp_root)
"""
    ).strip()


def intro_notebook():
    cells = [
        md(
            r"""
            # 00. 课程导航与学习方法

            这组 notebook 面向“系统学习机器学习与深度学习”的目标设计。整体原则是：

            1. 先从**建模假设**理解模型为什么成立。
            2. 再从**数学公式**理解模型究竟在优化什么。
            3. 接着用**代码实验**观察参数、损失和预测结果如何变化。
            4. 最后通过**可视化与对比实验**建立直觉，而不是只背结论。
            """
        ),
        md(
            r"""
            ## Notebook 顺序

            | 顺序 | 文件 | 核心内容 | 建议关注点 |
            | --- | --- | --- | --- |
            | 00 | `00_课程导航与学习方法.ipynb` | 学习路线、评价指标、使用建议 | 如何高效使用整套材料 |
            | 01 | `01_线性模型体系.ipynb` | 线性回归、Ridge、Lasso、Logistic 回归 | 损失函数、正则化、可解释性 |
            | 02 | `02_决策树体系.ipynb` | 分类树、回归树、剪枝 | 熵、基尼系数、树深度与过拟合 |
            | 03 | `03_集成学习体系.ipynb` | Bagging、Random Forest、AdaBoost、GBDT | 偏差-方差权衡、弱学习器集成 |
            | 04 | `04_聚类与降维体系.ipynb` | KMeans、层次聚类、DBSCAN、PCA | 无监督学习的目标与局限 |
            | 05 | `05_PyTorch基础与MLP体系.ipynb` | Tensor、Autograd、多层感知机 | 深度学习最基本训练闭环 |
            | 06 | `06_CNN体系.ipynb` | 卷积、池化、BatchNorm、特征图 | 参数共享与局部感受野 |
            | 07 | `07_RNN_LSTM_GRU体系.ipynb` | RNN、LSTM、GRU | 时间依赖、梯度消失、门控机制 |
            | 08 | `08_Transformer体系.ipynb` | Self-Attention、多头注意力、位置编码 | 为什么 Transformer 能替代传统序列模型 |
            | 09 | `09_优化器与损失函数体系.ipynb` | SGD、Momentum、RMSProp、Adam、AdamW 与常见损失函数 | 为什么优化器和损失函数决定训练行为 |
            | 10 | `10_训练稳定性与正则化技巧.ipynb` | 初始化、归一化、Dropout、权重衰减、学习率调度 | 如何让深度网络更稳、更泛化 |
            | 11 | `11_大语言模型架构与MoE体系.ipynb` | Decoder-only LLM、RoPE、KV Cache、GQA/MQA、MoE | 现代开源大模型的核心架构与差异 |
            | 12 | `12_参数高效微调与推理解码体系.ipynb` | LoRA、QLoRA、Prompt Tuning、Greedy / Beam / Top-k / Top-p | 如何以更低成本使用大模型 |
            | 13 | `13_现代开源模型家族与新架构体系.ipynb` | Qwen、Gemma、OLMo、Jamba、Mamba / SSM | 如何理解不同开源家族的设计取向 |
            | 14 | `14_对齐训练与偏好优化体系.ipynb` | SFT、Reward Model、RLHF、DPO、ORPO、KTO | 大模型为什么需要后训练与偏好优化 |
            | 15 | `15_推理系统与部署优化体系.ipynb` | 量化、KV Cache、Paged Attention、Speculative Decoding、Continuous Batching | 为什么系统优化决定大模型落地成本 |
            """
        ),
        md(
            r"""
            ## 推荐学习方法

            每个 notebook 都建议按下面顺序使用：

            1. **先读开头的 Markdown**，明确模型的输入、输出、目标函数和适用场景。
            2. **自己推导关键公式**，至少要能解释每个符号代表什么。
            3. **运行代码并观察图像**，把抽象公式和真实结果对应起来。
            4. **修改超参数**，例如学习率、正则化强度、隐藏层维度、树深度等。
            5. **写总结**：这个模型解决什么问题，为什么有效，何时会失效。
            """
        ),
        md(
            r"""
            ## 常见评价指标

            ### 回归

            - 均方误差（MSE）  
              $$
              \mathrm{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
              $$
            - 平均绝对误差（MAE）  
              $$
              \mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
              $$
            - 决定系数 $R^2$：衡量模型解释方差的能力。

            ### 分类

            - 准确率（Accuracy）
            - 精确率（Precision）
            - 召回率（Recall）
            - F1 分数
            - 交叉熵（Cross Entropy）
            """
        ),
        code(
            """
            from pathlib import Path
            import os
            import warnings

            temp_root = Path(os.environ.get("ML_DL_TMP", str(Path.home() / ".ml_dl_notebook_tmp")))
            temp_root.mkdir(exist_ok=True)
            os.environ["TMP"] = str(temp_root)
            os.environ["TEMP"] = str(temp_root)

            warnings.filterwarnings("ignore")

            import sys
            import numpy as np
            import pandas as pd
            import sklearn
            import matplotlib
            import seaborn as sns
            import torch

            print("Python 版本:", sys.version.split()[0])
            print("NumPy 版本:", np.__version__)
            print("Pandas 版本:", pd.__version__)
            print("sklearn 版本:", sklearn.__version__)
            print("Matplotlib 版本:", matplotlib.__version__)
            print("Seaborn 版本:", sns.__version__)
            print("PyTorch 版本:", torch.__version__)
            """
        ),
        md(
            r"""
            ## 学习节奏建议

            如果你是第一次系统学习，可以按下面的节奏推进：

            - 第 1 周：线性模型 + 决策树
            - 第 2 周：集成学习 + 聚类与降维
            - 第 3 周：PyTorch 基础 + MLP
            - 第 4 周：CNN
            - 第 5 周：RNN / LSTM / GRU
            - 第 6 周：Transformer
            - 第 7 周：优化器、损失函数、初始化与正则化
            - 第 8 周：Decoder-only LLM、MoE 与开源架构对比
            - 第 9 周：参数高效微调与推理解码
            - 第 10 周：现代开源模型家族、SSM / Hybrid 架构
            - 第 11 周：SFT、RLHF、DPO 等对齐训练链路
            - 第 12 周：推理系统、量化与部署优化
            - 第 13 周：自己选一个数据集，完整复现“数据处理 -> 建模 -> 评估 -> 可视化”
            """
        ),
    ]
    write_notebook("00_课程导航与学习方法.ipynb", cells)


def intro_notebook():
    cells = [
        md(
            r"""
            # 00. 课程导航与学习方法
            这套 notebook 的目标不是把模型名称机械罗列，而是把机器学习、深度学习、强化学习与大语言模型后训练整理成一条可以逐层推进的知识主线。整体组织遵循四个原则：

            1. 先明确模型的**问题定义、输入输出与建模假设**。
            2. 再分析模型的**目标函数、优化机制与统计含义**。
            3. 通过**可运行代码与可视化**观察模型行为，而不是只记结论。
            4. 把模型放回方法谱系中，理解它相对上一代方法究竟改进了什么。
            """
        ),
        md(
            r"""
            ## Notebook 顺序

            | 顺序 | 文件 | 核心内容 | 建议关注点 |
            | --- | --- | --- | --- |
            | 00 | `00_课程导航与学习方法.ipynb` | 学习路线、评估指标、使用方式 | 如何高效使用整套课程 |
            | 01 | `01_线性模型体系.ipynb` | 线性回归、Ridge、Lasso、Logistic 回归 | 损失函数、正则化、可解释性 |
            | 02 | `02_决策树体系.ipynb` | 分类树、回归树、剪枝 | 信息增益、基尼系数、树复杂度 |
            | 03 | `03_集成学习体系.ipynb` | Bagging、Random Forest、AdaBoost、GBDT | 偏差-方差权衡与弱学习器集成 |
            | 04 | `04_聚类与降维体系.ipynb` | KMeans、层次聚类、DBSCAN、PCA | 无监督学习的目标与局限 |
            | 05 | `05_PyTorch基础与MLP体系.ipynb` | Tensor、Autograd、MLP | 深度学习训练闭环的最小单元 |
            | 06 | `06_CNN体系.ipynb` | 卷积、池化、BatchNorm、特征图 | 局部连接与参数共享 |
            | 07 | `07_RNN_LSTM_GRU体系.ipynb` | RNN、LSTM、GRU | 时序建模、梯度传播与门控机制 |
            | 08 | `08_Transformer体系.ipynb` | Self-Attention、多头注意力、位置编码 | Transformer 的表示与并行性优势 |
            | 09 | `09_优化器与损失函数体系.ipynb` | SGD、Momentum、RMSProp、Adam、AdamW 与常见损失函数 | 优化动力学与任务匹配 |
            | 10 | `10_训练稳定性与正则化技巧.ipynb` | 初始化、归一化、Dropout、权重衰减、学习率调度 | 如何让深层网络稳定可训练 |
            | 11 | `11_大语言模型架构与MoE体系.ipynb` | Decoder-only LLM、RoPE、KV Cache、GQA/MQA、MoE | 现代 LLM 的核心结构 |
            | 12 | `12_参数高效微调与推理解码体系.ipynb` | LoRA、QLoRA、Prompt Tuning、Greedy / Beam / Top-k / Top-p | 微调成本与推理策略 |
            | 13 | `13_现代开源模型家族与新架构体系.ipynb` | Qwen、Gemma、OLMo、Jamba、Mamba / SSM | 开源模型家族与新架构取向 |
            | 14 | `14_对齐训练与偏好优化体系.ipynb` | SFT、Reward Model、RLHF、DPO、ORPO、KTO、SimPO | 大模型后训练的基本链路 |
            | 15 | `15_推理系统与部署优化体系.ipynb` | 量化、KV Cache、PagedAttention、Speculative Decoding、Continuous Batching | 推理系统级优化 |
            | 16 | `16_强化学习基础与MDP体系.ipynb` | MDP、回报、Bellman 方程、DP / MC / TD | 强化学习的统一问题设置 |
            | 17 | `17_值函数方法与DQN体系.ipynb` | SARSA、Q-learning、DQN、Double DQN、Dueling DQN、PER | 值函数逼近与离策略控制 |
            | 18 | `18_策略梯度与Actor_Critic体系.ipynb` | REINFORCE、Baseline、Advantage、A2C/A3C、TRPO、PPO | 策略优化与稳定更新 |
            | 19 | `19_连续控制与现代深度强化学习体系.ipynb` | DDPG、TD3、SAC、熵正则化、探索噪声 | 连续动作空间的现代方法 |
            | 20 | `20_LLM后训练中的RL与PO体系.ipynb` | PPO、RLOO、GRPO、GDPO、GSPO、SAPO、GiGPO 与 DPO 家族对比 | 大语言模型偏好优化前沿 |
            """
        ),
        md(
            r"""
            ## 推荐学习方法

            每个 notebook 建议按同一条主线阅读：

            1. 先读 Markdown，明确模型的**任务定义、参数化方式、目标函数和适用场景**。
            2. 再看公式，把每一个符号和损失项的作用说清楚，而不是只记“最终公式”。
            3. 运行代码并观察图像，把抽象数学与实际训练现象对应起来。
            4. 主动修改超参数，例如学习率、正则化强度、折扣因子、网络宽度、clip 系数、温度系数等。
            5. 最后写一段总结：该方法解决什么问题、为什么有效、相对前一代方法改进了什么、在什么条件下会失败。
            """
        ),
        md(
            r"""
            ## 常见评价指标

            ### 回归

            - 均方误差（MSE）
              $$
              \mathrm{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
              $$
            - 平均绝对误差（MAE）
              $$
              \mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
              $$
            - 决定系数 $R^2$

            ### 分类

            - Accuracy、Precision、Recall、F1、ROC-AUC、PR-AUC
            - Cross Entropy

            ### 强化学习

            - Episodic Return：单条轨迹累计回报
            - Average Return：多轮平均回报
            - Success Rate：任务完成比例
            - Sample Efficiency：达到给定性能所需样本量
            - Stability：不同随机种子下的方差与最差表现

            ### 大模型后训练

            - Win Rate / Preference Accuracy
            - KL to Reference：与参考策略的偏移程度
            - Response Length / Entropy：回答长度与多样性
            - Reward Hacking Risk：是否通过投机模式刷高奖励
            """
        ),
        code(
            """
            from pathlib import Path
            import os
            import warnings

            temp_root = Path(os.environ.get("ML_DL_TMP", str(Path.home() / ".ml_dl_notebook_tmp")))
            temp_root.mkdir(exist_ok=True)
            os.environ["TMP"] = str(temp_root)
            os.environ["TEMP"] = str(temp_root)

            warnings.filterwarnings("ignore")

            import sys
            import numpy as np
            import pandas as pd
            import sklearn
            import matplotlib
            import seaborn as sns
            import torch

            print("Python version:", sys.version.split()[0])
            print("NumPy version:", np.__version__)
            print("Pandas version:", pd.__version__)
            print("sklearn version:", sklearn.__version__)
            print("Matplotlib version:", matplotlib.__version__)
            print("Seaborn version:", sns.__version__)
            print("PyTorch version:", torch.__version__)
            """
        ),
        md(
            r"""
            ## 建议学习节奏

            如果是第一次系统学习，可以参考下面的推进顺序：

            - 第 1 周：线性模型、决策树
            - 第 2 周：集成学习、聚类与降维
            - 第 3 周：PyTorch 基础与 MLP
            - 第 4 周：CNN
            - 第 5 周：RNN / LSTM / GRU
            - 第 6 周：Transformer
            - 第 7 周：优化器、损失函数、正则化与训练稳定性
            - 第 8 周：LLM 基础架构与 MoE
            - 第 9 周：参数高效微调与推理解码
            - 第 10 周：开源模型家族、SSM 与 Hybrid 架构
            - 第 11 周：SFT、RLHF、DPO 家族的基础后训练链路
            - 第 12 周：推理系统、量化与部署优化
            - 第 13 周：强化学习基础与值函数方法
            - 第 14 周：策略梯度、Actor-Critic、PPO
            - 第 15 周：连续控制、DDPG / TD3 / SAC
            - 第 16 周：LLM 后训练中的 RL 与现代 PO 体系
            """
        ),
    ]
    write_notebook("00_课程导航与学习方法.ipynb", cells)


def linear_models_notebook():
    cells = [
        md(
            r"""
            # 01. 线性模型体系

            线性模型是机器学习中最重要的一类基线模型。虽然形式简单，但它们几乎覆盖了：

            - 回归问题中的基础建模思路
            - 分类问题中的概率建模思路
            - 正则化、可解释性、凸优化等核心概念
            """
        ),
        *professional_cells(
            model_title="线性模型",
            definition="""
            线性模型是一类以输入特征的线性组合为核心假设的统计学习模型。其基本形式可统一写为：

            $$
            f(x) = w^T x + b
            $$

            在线性回归中，$f(x)$ 直接作为实值预测；在 Logistic 回归中，$f(x)$ 作为 logit，再通过非线性链接函数映射到概率空间。线性模型的关键价值在于：模型假设清晰、目标函数通常是凸的、参数具有直接解释性。
            """,
            io_spec="""
            输入通常为定长特征向量 $x \in \mathbb{R}^d$，参数为权重向量 $w \in \mathbb{R}^d$ 与偏置标量 $b$。

            - 回归任务输出 $\hat{y} \in \mathbb{R}$
            - 二分类任务输出 $P(y=1|x)\in[0,1]$
            - 多分类任务则输出各类别的概率分布

            对于高维稀疏特征，线性模型仍然具有很强的适配能力，这也是它在工业 CTR、风控、文本稀疏特征场景长期被使用的重要原因。
            """,
            architecture="""
            线性模型的结构极简，但其统计含义非常完整：

            1. 每个输入特征对应一个权重
            2. 权重与特征相乘，表示该特征对预测的边际贡献
            3. 所有贡献相加，再叠加偏置项
            4. 根据任务类型，直接输出数值或再经由链接函数输出概率

            在这种结构下，模型不会自动学习复杂层级表示，因此它更依赖输入特征的表达质量，而不是网络深度。
            """,
            objective="""
            线性回归通常最小化均方误差，Logistic 回归通常最小化负对数似然或交叉熵。加入 L1 / L2 正则化后，优化目标不仅关注拟合误差，还会对参数复杂度施加约束。

            这使得线性模型天然适合作为“偏差-方差权衡”的教学起点：

            - 无正则化：偏差较低，但更容易过拟合
            - L2 正则化：参数整体收缩，模型更稳定
            - L1 正则化：得到稀疏解，兼具拟合和特征选择作用
            """,
            complexity="""
            在线性模型中，单次预测复杂度通常为 $O(d)$；若采用闭式解，训练复杂度与矩阵求逆相关；若采用梯度法，则与样本数、特征数和迭代轮数线性相关。

            从统计角度看，线性模型假设较强，因此在关系接近线性的场景中高效且稳健；在复杂非线性场景中则可能出现明显欠拟合。
            """,
            comparison="""
            与决策树相比，线性模型的边界更平滑、可解释性更直接，但对复杂非线性关系表达不足。
            与神经网络相比，线性模型的表示能力有限，但优化更稳定、训练更便宜、结果更容易解释。
            因此在线上系统中，线性模型经常承担两个角色：一是强基线，二是高可解释性模型。
            """,
        ),
        architecture_diagram_cell("linear"),
        md(
            r"""
            ## 学习目标

            学完本 notebook 后，你应该能够：

            1. 写出线性回归、Ridge、Lasso、Logistic 回归的目标函数。
            2. 解释正则化为什么能降低过拟合。
            3. 使用 `sklearn` 完成回归和分类任务。
            4. 通过系数图、残差图、混淆矩阵理解模型行为。
            """
        ),
        *teaching_cells(
            intuition="""
            可以把线性模型理解成“给每个特征一个分数，然后把这些分数加起来做决定”。

            例如预测房价时：

            - 面积可能加分
            - 房龄可能减分
            - 地段可能强烈加分

            这些“加分”和“减分”的力度，就是模型学出来的参数。

            线性模型最重要的价值不是“永远最强”，而是：

            - 它是很多复杂模型的出发点
            - 它的每个参数都容易解释
            - 你很容易看懂模型为什么做出某个预测
            """,
            architecture="""
            对回归问题，模型输出是一个实数；对分类问题，模型先算线性分数，再把分数转成概率。

            从结构上看，线性模型几乎没有“深层结构”，它本质上只有一层：

            1. 接收输入特征
            2. 每个特征乘以自己的权重
            3. 把结果加总，再加偏置
            4. 输出预测值或类别概率

            这也是为什么线性模型速度快、结构清晰、可解释性强。

            但代价也很明显：

            - 如果真实关系是弯曲的、复杂的、存在高阶交互，线性模型很难直接表达
            - 它更依赖好的特征工程
            """,
            training="""
            训练线性模型，其实就是在问：

            “怎样选择一组权重，让模型预测结果和真实答案尽量接近？”

            对回归来说，通常最小化平方误差；对分类来说，通常最小化交叉熵。

            正则化的意义也很重要：

            - Ridge 会限制权重不要过大，让模型更稳
            - Lasso 会主动把一部分权重压到 0，起到特征选择作用

            所以训练线性模型并不只是拟合数据，更是在做“拟合能力”和“泛化能力”的平衡。
            """,
            usage="""
            优先考虑线性模型的场景：

            - 你需要一个强基线
            - 你希望看到每个特征对结果的影响方向和强度
            - 数据规模不算大，但想先快速判断问题难度
            - 你要做商业、金融、医学等需要解释性的任务

            不适合的场景：

            - 图像、语音、长文本等高维复杂结构数据
            - 明显存在复杂非线性边界的问题
            """,
            pitfalls="""
            初学者最容易有三个误解：

            1. **误以为“线性模型简单，所以没有学习价值”**
               实际上，很多优化、正则化、概率建模思想都要先从这里理解。

            2. **误以为 Logistic 回归是回归模型**
               它名字里有“回归”，但最常见用途是分类，因为它输出的是类别概率。

            3. **误以为线性模型一定只能拟合直线**
               只要你做了非线性特征变换，例如多项式特征，线性模型也能拟合复杂函数。
            """,
        ),
        md(
            r"""
            ## 1. 线性回归的公式与原理

            对于输入向量 $x \in \mathbb{R}^d$，线性回归假设：

            $$
            \hat{y} = w^T x + b
            $$

            其中：

            - $w$ 是权重向量
            - $b$ 是偏置项
            - $\hat{y}$ 是模型预测值

            在线性回归中，我们通常最小化均方误差：

            $$
            J(w, b) = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
            $$
            """
        ),
        code(common_style_code(include_torch=False)),
        code(
            """
            from sklearn.datasets import load_diabetes
            from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
            from sklearn.metrics import (
                ConfusionMatrixDisplay,
                accuracy_score,
                classification_report,
                f1_score,
                mean_absolute_error,
                mean_squared_error,
                r2_score,
            )
            from sklearn.model_selection import train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler

            diabetes = load_diabetes(as_frame=True)
            X_reg = diabetes.data
            y_reg = diabetes.target

            X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
                X_reg, y_reg, test_size=0.2, random_state=42
            )

            reg_models = {
                "LinearRegression": LinearRegression(),
                "Ridge(alpha=1.0)": Ridge(alpha=1.0),
                "Lasso(alpha=0.05)": Lasso(alpha=0.05),
            }

            reg_results = []
            fitted_reg_models = {}

            for name, model in reg_models.items():
                pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
                pipe.fit(X_train_reg, y_train_reg)
                preds = pipe.predict(X_test_reg)
                fitted_reg_models[name] = pipe
                reg_results.append(
                    {
                        "模型": name,
                        "MSE": mean_squared_error(y_test_reg, preds),
                        "MAE": mean_absolute_error(y_test_reg, preds),
                        "R^2": r2_score(y_test_reg, preds),
                    }
                )

            reg_results_df = pd.DataFrame(reg_results).sort_values("MSE")
            reg_results_df
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            sns.barplot(data=reg_results_df, x="模型", y="MSE", ax=axes[0], palette="Blues_d")
            axes[0].set_title("不同线性回归模型的 MSE")
            axes[0].tick_params(axis="x", rotation=15)

            sns.barplot(data=reg_results_df, x="模型", y="R^2", ax=axes[1], palette="Greens_d")
            axes[1].set_title("不同线性回归模型的 R^2")
            axes[1].tick_params(axis="x", rotation=15)

            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            r"""
            ## 2. 正则化：Ridge 与 Lasso

            ### Ridge（L2 正则化）

            $$
            J(w, b) = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \|w\|_2^2
            $$

            ### Lasso（L1 正则化）

            $$
            J(w, b) = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \|w\|_1
            $$

            Ridge 更倾向于整体收缩参数；Lasso 更容易得到稀疏解。
            """
        ),
        code(
            """
            coef_df = pd.DataFrame({"feature": X_reg.columns})
            for name, pipe in fitted_reg_models.items():
                coef_df[name] = pipe.named_steps["model"].coef_

            coef_long = coef_df.melt(id_vars="feature", var_name="模型", value_name="系数")

            plt.figure(figsize=(14, 7))
            sns.barplot(data=coef_long, x="feature", y="系数", hue="模型")
            plt.xticks(rotation=45, ha="right")
            plt.title("不同线性模型的特征系数对比")
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            alphas = np.logspace(-3, 1, 20)
            coef_path = []

            for alpha in alphas:
                pipe = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", Lasso(alpha=alpha, max_iter=10000)),
                    ]
                )
                pipe.fit(X_train_reg, y_train_reg)
                coef_path.append(pipe.named_steps["model"].coef_)

            coef_path = np.array(coef_path)

            plt.figure(figsize=(12, 7))
            for idx, feature_name in enumerate(X_reg.columns):
                plt.plot(alphas, coef_path[:, idx], label=feature_name)

            plt.xscale("log")
            plt.xlabel("alpha (log scale)")
            plt.ylabel("coefficient value")
            plt.title("Lasso 系数路径图")
            plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            r"""
            ## 3. Logistic 回归

            Logistic 回归先计算线性得分：

            $$
            z = w^T x + b
            $$

            再通过 Sigmoid 转成概率：

            $$
            \sigma(z) = \frac{1}{1 + e^{-z}}
            $$

            最终最小化的常见目标是二元交叉熵。
            """
        ),
        code(
            """
            from sklearn.datasets import load_breast_cancer, make_classification

            X_syn, y_syn = make_classification(
                n_samples=400,
                n_features=2,
                n_redundant=0,
                n_informative=2,
                n_clusters_per_class=1,
                class_sep=1.8,
                random_state=42,
            )

            syn_model = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])
            syn_model.fit(X_syn, y_syn)

            xx, yy = np.meshgrid(
                np.linspace(X_syn[:, 0].min() - 1, X_syn[:, 0].max() + 1, 300),
                np.linspace(X_syn[:, 1].min() - 1, X_syn[:, 1].max() + 1, 300),
            )
            grid = np.c_[xx.ravel(), yy.ravel()]
            prob = syn_model.predict_proba(grid)[:, 1].reshape(xx.shape)

            plt.figure(figsize=(10, 8))
            plt.contourf(xx, yy, prob, levels=20, cmap="RdBu", alpha=0.35)
            plt.contour(xx, yy, prob, levels=[0.5], colors="black", linewidths=2)
            plt.scatter(X_syn[:, 0], X_syn[:, 1], c=y_syn, cmap="Set1", edgecolor="k", s=40)
            plt.title("Logistic 回归的决策边界与概率等高线")
            plt.show()
            """
        ),
        code(
            """
            cancer = load_breast_cancer(as_frame=True)
            X_cls = cancer.data
            y_cls = cancer.target

            X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
                X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
            )

            logreg = Pipeline(
                [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=5000))]
            )
            logreg.fit(X_train_cls, y_train_cls)
            y_pred_cls = logreg.predict(X_test_cls)

            print(classification_report(y_test_cls, y_pred_cls, target_names=cancer.target_names))
            print("Accuracy:", round(accuracy_score(y_test_cls, y_pred_cls), 4))
            print("F1 score:", round(f1_score(y_test_cls, y_pred_cls), 4))
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(18, 7))

            ConfusionMatrixDisplay.from_predictions(
                y_test_cls, y_pred_cls, display_labels=cancer.target_names, cmap="Blues", ax=axes[0]
            )
            axes[0].set_title("Breast Cancer: 混淆矩阵")

            coef_series = pd.Series(
                logreg.named_steps["model"].coef_[0], index=X_cls.columns
            ).sort_values(key=np.abs, ascending=False).head(12)
            sns.barplot(x=coef_series.values, y=coef_series.index, palette="vlag", ax=axes[1])
            axes[1].set_title("Logistic 回归中最重要的 12 个特征")
            axes[1].set_xlabel("系数")

            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            r"""
            ## 4. 线性模型的用法总结

            - 需要速度快、可解释、强基线时，优先考虑线性模型。
            - 需要控制过拟合时，优先考虑正则化版本。
            - 数据关系明显非线性时，线性模型往往不如树模型与神经网络。
            """
        ),
    ]
    write_notebook("01_线性模型体系.ipynb", cells)


def decision_tree_notebook():
    cells = [
        md(
            r"""
            # 02. 决策树体系

            决策树的核心思想是：不断提问“某个特征是否满足某个条件”，把样本一步步切分到更纯净的子区域中。
            """
        ),
        *professional_cells(
            model_title="决策树",
            definition="""
            决策树是一类递归分割特征空间的非参数模型。它通过一系列条件判断将输入空间划分为若干互不重叠的区域，并在每个区域上输出类别标签或实值预测。

            对分类树而言，叶节点输出类别分布；对回归树而言，叶节点输出该区域内目标值的统计汇总，通常是均值。
            """,
            io_spec="""
            输入为结构化特征向量，内部参数不是连续权重矩阵，而是一组离散的切分规则：

            - 选择哪个特征
            - 在哪个阈值处切分
            - 左右子树分别如何继续分裂

            输出是由输入样本落入的叶节点决定的。
            """,
            architecture="""
            决策树的结构是层次化的条件分裂图。

            根节点负责全局第一次分割；中间节点持续在局部区域上做进一步切分；叶节点则对应最终决策区域。每一次切分都会改变样本所在的子空间，因此决策树本质上是一个分段常数模型。

            由于每次切分一般沿单一特征轴进行，标准树模型形成的边界往往是轴对齐的分块区域。
            """,
            objective="""
            训练过程是离散搜索问题。分类树选择能最大化纯度提升的切分，回归树选择能最大化误差下降的切分。

            递归分裂会不断降低训练误差，但也会逐步提高模型方差。因此剪枝、最大深度限制、最小叶节点样本数等机制并不是附属参数，而是决策树泛化控制的核心。
            """,
            complexity="""
            决策树训练通常需要在多个特征和阈值组合上搜索最优分裂，代价高于单次线性模型拟合，但推理速度很快，只需沿树路径向下遍历。

            从统计性质看，单棵深树偏差低、方差高；浅树偏差高、方差低。这也是后续随机森林和 Boosting 出现的理论背景。
            """,
            comparison="""
            与线性模型相比，决策树可以天然表达非线性和特征交互，但稳定性更弱。
            与神经网络相比，单棵树在表格数据上通常更容易解释，但表达边界不够平滑，且对数据扰动敏感。
            决策树最重要的理论地位在于：它既是一个独立模型，也是绝大多数集成树模型的基本构件。
            """,
        ),
        architecture_diagram_cell("tree"),
        *teaching_cells(
            intuition="""
            决策树最像人类做判断的过程。

            比如你在判断“要不要买一台电脑”，可能会这样思考：

            - 预算高吗？
            - 如果预算高，是不是经常打游戏？
            - 如果不打游戏，是不是经常跑深度学习？

            这其实就是一棵树：每问一个问题，就把样本分到不同分支，直到得到最终结论。

            所以决策树的核心不是复杂数学，而是“不断提问，把复杂问题拆成一连串简单判断”。
            """,
            architecture="""
            从结构上看，决策树由三部分组成：

            - 根节点：第一个切分问题
            - 中间节点：继续切分
            - 叶子节点：最终预测结果

            分类树在叶子节点输出类别；回归树在叶子节点输出数值。

            决策树的本质是把特征空间切成很多区域。每次切分都在问：

            “如果按这个特征、这个阈值切开，左右两边会不会更纯？”

            这就是为什么树模型的决策边界常常是块状、阶梯状的。
            """,
            training="""
            训练决策树并不是“调一个连续公式”，而是在大量候选切分中搜索最优切分。

            每一层都会做两件事：

            1. 枚举某个特征和阈值
            2. 计算切完之后纯度是否提升

            分类任务看的是熵或基尼系数；回归任务看的是平方误差能否下降。

            树越深，通常训练集拟合越好，但也更容易把噪声当规律学进去，这就是过拟合。
            因此剪枝、限制树深、限制叶子样本数，都是在防止树长得过于复杂。
            """,
            usage="""
            决策树适合：

            - 表格数据
            - 特征之间有明显条件分支关系的问题
            - 需要可解释性的问题
            - 想快速做出可视化和业务解释的问题

            如果你面对的是结构化表格任务，树模型通常比神经网络更值得优先尝试。
            """,
            pitfalls="""
            常见误区：

            1. **误以为树越深越强**
               深树往往只是训练集更强，不代表测试集更强。

            2. **误以为树模型不需要调参**
               `max_depth`、`min_samples_leaf`、`ccp_alpha` 都会显著影响泛化。

            3. **误以为决策树一定稳定**
               单棵树对数据扰动其实很敏感，这也是后面集成学习要解决的问题。
            """,
        ),
        md(
            r"""
            ## 1. 分类树中的纯度指标

            ### 熵（Entropy）

            $$
            H(S) = - \sum_k p_k \log_2 p_k
            $$

            ### 基尼系数（Gini Impurity）

            $$
            G(S) = 1 - \sum_k p_k^2
            $$
            """
        ),
        code(common_style_code(include_torch=False)),
        code(
            """
            from sklearn.datasets import load_diabetes, load_iris, make_moons
            from sklearn.metrics import ConfusionMatrixDisplay, mean_squared_error, r2_score
            from sklearn.model_selection import train_test_split
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

            X_moon, y_moon = make_moons(n_samples=400, noise=0.25, random_state=42)

            moon_models = {
                "max_depth=1": DecisionTreeClassifier(max_depth=1, random_state=42),
                "max_depth=3": DecisionTreeClassifier(max_depth=3, random_state=42),
                "max_depth=6": DecisionTreeClassifier(max_depth=6, random_state=42),
            }

            for model in moon_models.values():
                model.fit(X_moon, y_moon)
            """
        ),
        code(
            """
            def plot_boundary(ax, model, X, y, title):
                xx, yy = np.meshgrid(
                    np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300),
                    np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300),
                )
                grid = np.c_[xx.ravel(), yy.ravel()]
                prob = model.predict(grid).reshape(xx.shape)
                ax.contourf(xx, yy, prob, alpha=0.3, cmap="coolwarm")
                ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=35, edgecolor="k")
                ax.set_title(title)

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            for ax, (name, model) in zip(axes, moon_models.items()):
                plot_boundary(ax, model, X_moon, y_moon, f"决策边界: {name}")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            r"""
            ## 2. Iris 分类树

            决策树分类模型会递归切分，直到达到最大深度、节点过小或者纯度足够高。
            """
        ),
        code(
            """
            iris = load_iris(as_frame=True)
            X_iris = iris.data
            y_iris = iris.target

            X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
                X_iris, y_iris, test_size=0.25, random_state=42, stratify=y_iris
            )

            iris_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
            iris_tree.fit(X_train_iris, y_train_iris)

            fig, ax = plt.subplots(figsize=(18, 10))
            plot_tree(
                iris_tree,
                feature_names=iris.feature_names,
                class_names=iris.target_names,
                filled=True,
                rounded=True,
                fontsize=10,
                ax=ax,
            )
            ax.set_title("Iris 分类树结构图")
            plt.show()
            """
        ),
        code(
            """
            path = DecisionTreeClassifier(random_state=42).cost_complexity_pruning_path(
                X_train_iris, y_train_iris
            )
            ccp_alphas = path.ccp_alphas

            records = []
            for alpha in ccp_alphas[:-1]:
                clf = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
                clf.fit(X_train_iris, y_train_iris)
                records.append(
                    {
                        "ccp_alpha": alpha,
                        "depth": clf.get_depth(),
                        "leaves": clf.get_n_leaves(),
                        "train_score": clf.score(X_train_iris, y_train_iris),
                        "test_score": clf.score(X_test_iris, y_test_iris),
                    }
                )

            prune_df = pd.DataFrame(records)
            prune_df.head()
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            axes[0].plot(prune_df["ccp_alpha"], prune_df["depth"], marker="o")
            axes[0].set_xscale("log")
            axes[0].set_title("剪枝参数与树深度")

            axes[1].plot(prune_df["ccp_alpha"], prune_df["train_score"], label="train", marker="o")
            axes[1].plot(prune_df["ccp_alpha"], prune_df["test_score"], label="test", marker="o")
            axes[1].set_xscale("log")
            axes[1].set_title("剪枝参数与准确率")
            axes[1].legend()

            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            y_pred_iris = iris_tree.predict(X_test_iris)

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            ConfusionMatrixDisplay.from_predictions(
                y_test_iris, y_pred_iris, display_labels=iris.target_names, cmap="Blues", ax=axes[0]
            )
            axes[0].set_title("Iris 分类树的混淆矩阵")

            importance = pd.Series(iris_tree.feature_importances_, index=iris.feature_names).sort_values(
                ascending=True
            )
            importance.plot(kind="barh", ax=axes[1], color="teal")
            axes[1].set_title("特征重要性")

            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            r"""
            ## 3. 回归树

            对于回归问题，叶子节点输出的是该区域内目标值的平均值。
            """
        ),
        code(
            """
            diabetes = load_diabetes(as_frame=True)
            X_reg = diabetes.data
            y_reg = diabetes.target

            X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
                X_reg, y_reg, test_size=0.2, random_state=42
            )

            reg_depths = [2, 4, 6, None]
            reg_records = []
            reg_models = {}

            for depth in reg_depths:
                model = DecisionTreeRegressor(max_depth=depth, random_state=42)
                model.fit(X_train_reg, y_train_reg)
                preds = model.predict(X_test_reg)
                key = f"max_depth={depth}"
                reg_models[key] = model
                reg_records.append(
                    {"模型": key, "MSE": mean_squared_error(y_test_reg, preds), "R^2": r2_score(y_test_reg, preds)}
                )

            reg_tree_df = pd.DataFrame(reg_records).sort_values("MSE")
            reg_tree_df
            """
        ),
        code(
            """
            best_reg_key = reg_tree_df.iloc[0]["模型"]
            best_reg_model = reg_models[best_reg_key]
            best_reg_preds = best_reg_model.predict(X_test_reg)

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            sns.barplot(data=reg_tree_df, x="模型", y="MSE", ax=axes[0], palette="Oranges_d")
            axes[0].tick_params(axis="x", rotation=15)
            axes[0].set_title("不同回归树深度的 MSE")

            axes[1].scatter(y_test_reg, best_reg_preds, alpha=0.75, color="darkorange")
            min_v = min(y_test_reg.min(), best_reg_preds.min())
            max_v = max(y_test_reg.max(), best_reg_preds.max())
            axes[1].plot([min_v, max_v], [min_v, max_v], linestyle="--", color="black")
            axes[1].set_title(f"最佳回归树 ({best_reg_key}) 的预测效果")
            axes[1].set_xlabel("真实值")
            axes[1].set_ylabel("预测值")

            plt.tight_layout()
            plt.show()
            """
        ),
    ]
    write_notebook("02_决策树体系.ipynb", cells)


def ensemble_notebook():
    cells = [
        md(
            r"""
            # 03. 集成学习体系

            集成学习的思想很直接：单个模型不稳定，那就把多个模型组合起来。
            """
        ),
        *teaching_cells(
            intuition="""
            集成学习可以理解成“不要只听一个专家的意见”。

            单个模型可能会：

            - 看走眼
            - 对训练数据过于敏感
            - 在某些局部模式上犯系统性错误

            如果把多个模型组合起来，就有机会让它们互相纠错。

            所以集成学习最核心的思想是：

            - 要么让多个模型从不同角度看问题，然后投票
            - 要么让后面的模型专门修正前面的错误
            """,
            architecture="""
            集成学习主要有两条路线：

            - Bagging：多个模型并行训练，再做平均或投票
            - Boosting：多个模型串行训练，后一个接着修前一个

            随机森林属于 Bagging 路线，本质上是“很多棵不一样的树一起投票”。
            AdaBoost 和 GBDT 属于 Boosting 路线，本质上是“逐步叠加弱学习器，让整体越来越强”。
            """,
            training="""
            Bagging 训练时，会给每个子模型喂不同抽样的数据，目的是制造差异性。
            只有子模型之间有差异，投票才有意义。

            Boosting 则更像一条纠错流水线：

            - 第一个模型先学
            - 第二个模型重点关注第一个模型没学好的部分
            - 第三个模型继续修正前面模型的残差或错误

            所以 Bagging 更偏向“降方差”，Boosting 更偏向“降偏差”。
            """,
            usage="""
            如果你做的是表格数据任务，集成学习往往是极强的基线。

            - 随机森林：稳、好调、鲁棒
            - AdaBoost：结构简单时有优势
            - GBDT：通常精度更高，但调参更敏感

            在很多工业表格场景里，集成模型的表现非常强。
            """,
            pitfalls="""
            常见误区：

            1. **误以为模型越多越好**
               如果弱学习器几乎一样，增加数量收益会递减。

            2. **误以为 Boosting 一定优于 Bagging**
               有噪声、数据量不大、场景不稳定时，Boosting 也可能更容易过拟合。

            3. **误以为集成模型就不需要解释**
               虽然比单棵树复杂，但仍然可以通过特征重要性、SHAP 等方法做解释。
            """,
        ),
        md(
            r"""
            ## 1. 集成学习的核心直觉

            - Bagging：并行训练多个模型，再做投票或平均，主要用于**降方差**
            - Boosting：串行训练多个模型，后一个模型重点纠正前一个模型的错误
            """
        ),
        code(common_style_code(include_torch=False)),
        code(
            """
            from sklearn.datasets import load_breast_cancer, make_moons
            from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
            from sklearn.metrics import accuracy_score, f1_score
            from sklearn.model_selection import train_test_split
            from sklearn.tree import DecisionTreeClassifier

            X_moon, y_moon = make_moons(n_samples=500, noise=0.28, random_state=42)

            moon_models = {
                "单棵决策树": DecisionTreeClassifier(max_depth=3, random_state=42),
                "随机森林": RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42),
                "AdaBoost": AdaBoostClassifier(
                    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
                    n_estimators=150,
                    learning_rate=0.8,
                    random_state=42,
                ),
                "GradientBoosting": GradientBoostingClassifier(random_state=42),
            }

            for model in moon_models.values():
                model.fit(X_moon, y_moon)
            """
        ),
        code(
            """
            def plot_cls_boundary(ax, model, X, y, title):
                xx, yy = np.meshgrid(
                    np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300),
                    np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300),
                )
                grid = np.c_[xx.ravel(), yy.ravel()]
                pred = model.predict(grid).reshape(xx.shape)
                ax.contourf(xx, yy, pred, alpha=0.30, cmap="coolwarm")
                ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=30, edgecolor="k")
                ax.set_title(title)

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            for ax, (name, model) in zip(axes.ravel(), moon_models.items()):
                plot_cls_boundary(ax, model, X_moon, y_moon, name)
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            cancer = load_breast_cancer(as_frame=True)
            X = cancer.data
            y = cancer.target

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            models = {
                "DecisionTree": DecisionTreeClassifier(max_depth=4, random_state=42),
                "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42),
                "AdaBoost": AdaBoostClassifier(
                    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
                    n_estimators=200,
                    learning_rate=0.8,
                    random_state=42,
                ),
                "GradientBoosting": GradientBoostingClassifier(random_state=42),
            }

            results = []
            fitted_models = {}

            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                fitted_models[name] = model
                results.append(
                    {"模型": name, "Accuracy": accuracy_score(y_test, preds), "F1": f1_score(y_test, preds)}
                )

            ensemble_df = pd.DataFrame(results).sort_values("F1", ascending=False)
            ensemble_df
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            sns.barplot(data=ensemble_df, x="模型", y="Accuracy", ax=axes[0], palette="Blues")
            axes[0].set_title("集成模型 Accuracy 对比")

            sns.barplot(data=ensemble_df, x="模型", y="F1", ax=axes[1], palette="Greens")
            axes[1].set_title("集成模型 F1 对比")

            for ax in axes:
                ax.tick_params(axis="x", rotation=15)

            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            best_model_name = ensemble_df.iloc[0]["模型"]
            best_model = fitted_models[best_model_name]

            importance = pd.Series(best_model.feature_importances_, index=X.columns)
            top_importance = importance.sort_values(ascending=False).head(12).sort_values()

            plt.figure(figsize=(10, 7))
            plt.barh(top_importance.index, top_importance.values, color="slateblue")
            plt.title(f"{best_model_name} 的 Top-12 特征重要性")
            plt.xlabel("importance")
            plt.show()
            """
        ),
    ]
    write_notebook("03_集成学习体系.ipynb", cells)


def clustering_dimensionality_notebook():
    cells = [
        md(
            r"""
            # 04. 聚类与降维体系

            无监督学习不依赖标签，更强调数据内部结构的发现。
            """
        ),
        *teaching_cells(
            intuition="""
            无监督学习最像“看一堆没有标注的东西，试着自己找规律”。

            聚类在问的是：

            - 哪些样本彼此更像？
            - 数据里是不是天然存在几类群体？

            降维在问的是：

            - 这些高维特征里，真正有用的信息是不是只集中在少数方向上？

            所以无监督学习的核心，不是预测标签，而是理解数据结构本身。
            """,
            architecture="""
            这一节会出现三种完全不同的思路：

            - KMeans：假设数据围绕若干中心聚集
            - DBSCAN：假设高密度区域形成簇
            - PCA：假设数据主要沿少数几个方向变化

            它们不是同一种算法的不同版本，而是在回答完全不同的问题。
            理解这一点，比背公式更重要。
            """,
            training="""
            KMeans 通过“分配样本 -> 更新中心”反复迭代；
            DBSCAN 通过“局部密度是否足够高”扩展簇；
            PCA 则是直接从数据协方差结构中找最重要的投影方向。

            这意味着：

            - KMeans 很依赖距离定义和特征缩放
            - DBSCAN 很依赖 `eps` 和 `min_samples`
            - PCA 更像线性代数问题，而不是传统意义上的迭代分类器训练
            """,
            usage="""
            使用建议：

            - 想做用户分群、样本分群：优先考虑聚类
            - 想做高维可视化、压缩特征：优先考虑 PCA
            - 想找异常点、非球形簇：优先考虑 DBSCAN
            """,
            pitfalls="""
            常见误区：

            1. **误以为聚类一定能找到“真实类别”**
               聚类找到的是“结构上的相似性”，不一定和人工标签完全一致。

            2. **误以为降维后信息不会损失**
               降维本质就是压缩，只能尽量保留主信息，不可能无损。

            3. **误以为 PCA 是聚类算法**
               PCA 是降维工具，不负责分组。
            """,
        ),
        md(
            r"""
            ## 1. KMeans 的目标函数

            $$
            J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2
            $$
            """
        ),
        code(common_style_code(include_torch=False)),
        code(
            """
            from contextlib import nullcontext
            from scipy.cluster.hierarchy import dendrogram, linkage
            from sklearn.cluster import DBSCAN, KMeans
            from sklearn.datasets import load_digits, load_wine, make_moons
            from sklearn.decomposition import PCA
            from sklearn.metrics import silhouette_score
            from sklearn.preprocessing import StandardScaler
            import sklearn.cluster._kmeans as kmeans_backend

            # 当前 Windows 受限环境中 threadpoolctl 会在 KMeans 探测 BLAS 线程库时出错。
            # 这里把线程池限制逻辑替换成空上下文，保证 notebook 可稳定执行。
            kmeans_backend.threadpool_info = lambda: []
            kmeans_backend.threadpool_limits = lambda *args, **kwargs: nullcontext()

            wine = load_wine(as_frame=True)
            X_wine = wine.data
            y_wine = wine.target

            scaler = StandardScaler()
            X_wine_scaled = scaler.fit_transform(X_wine)

            ks = range(2, 9)
            kmeans_records = []
            for k in ks:
                model = KMeans(n_clusters=k, random_state=42, n_init=20)
                labels = model.fit_predict(X_wine_scaled)
                kmeans_records.append(
                    {"k": k, "inertia": model.inertia_, "silhouette": silhouette_score(X_wine_scaled, labels)}
                )

            kmeans_eval_df = pd.DataFrame(kmeans_records)
            kmeans_eval_df
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            axes[0].plot(kmeans_eval_df["k"], kmeans_eval_df["inertia"], marker="o")
            axes[0].set_title("Elbow Method")

            axes[1].plot(kmeans_eval_df["k"], kmeans_eval_df["silhouette"], marker="o", color="darkgreen")
            axes[1].set_title("Silhouette Score")

            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            pca_wine = PCA(n_components=2, random_state=42)
            wine_2d = pca_wine.fit_transform(X_wine_scaled)

            kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
            cluster_labels = kmeans.fit_predict(X_wine_scaled)

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            sns.scatterplot(x=wine_2d[:, 0], y=wine_2d[:, 1], hue=y_wine, palette="Set2", s=90, ax=axes[0])
            axes[0].set_title("Wine 真实类别（PCA 2D）")

            sns.scatterplot(x=wine_2d[:, 0], y=wine_2d[:, 1], hue=cluster_labels, palette="Set1", s=90, ax=axes[1])
            axes[1].set_title("KMeans 聚类结果（PCA 2D）")

            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            subset_idx = np.random.choice(len(X_wine_scaled), size=35, replace=False)
            linkage_matrix = linkage(X_wine_scaled[subset_idx], method="ward")

            plt.figure(figsize=(16, 6))
            dendrogram(linkage_matrix, leaf_rotation=90, leaf_font_size=10)
            plt.title("Wine 子集的层次聚类树状图")
            plt.show()
            """
        ),
        code(
            """
            X_moon, y_moon = make_moons(n_samples=500, noise=0.08, random_state=42)
            X_moon_scaled = StandardScaler().fit_transform(X_moon)

            dbscan = DBSCAN(eps=0.22, min_samples=5)
            db_labels = dbscan.fit_predict(X_moon_scaled)

            plt.figure(figsize=(10, 7))
            sns.scatterplot(x=X_moon_scaled[:, 0], y=X_moon_scaled[:, 1], hue=db_labels, palette="tab10", s=70)
            plt.title("DBSCAN 在双月数据上的聚类结果")
            plt.show()
            """
        ),
        code(
            """
            digits = load_digits()
            X_digits = digits.data
            y_digits = digits.target

            X_digits_scaled = StandardScaler().fit_transform(X_digits)
            pca_digits = PCA(n_components=20, random_state=42)
            X_digits_pca = pca_digits.fit_transform(X_digits_scaled)

            cumulative_ratio = np.cumsum(pca_digits.explained_variance_ratio_)

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            axes[0].plot(range(1, 21), pca_digits.explained_variance_ratio_, marker="o")
            axes[0].set_title("每个主成分的解释方差比")

            axes[1].plot(range(1, 21), cumulative_ratio, marker="o", color="darkred")
            axes[1].set_title("累计解释方差比")

            plt.tight_layout()
            plt.show()

            pca_2d = PCA(n_components=2, random_state=42)
            digits_2d = pca_2d.fit_transform(X_digits_scaled)

            plt.figure(figsize=(10, 8))
            sns.scatterplot(x=digits_2d[:, 0], y=digits_2d[:, 1], hue=y_digits, palette="tab10", s=55, alpha=0.8)
            plt.title("Digits 数据的 PCA 二维投影")
            plt.show()
            """
        ),
    ]
    write_notebook("04_聚类与降维体系.ipynb", cells)


def mlp_notebook():
    cells = [
        md(
            r"""
            # 05. PyTorch 基础与 MLP 体系

            这一节开始进入深度学习。我们会用 PyTorch 建立最基础的训练闭环。
            """
        ),
        *professional_cells(
            model_title="多层感知机（MLP）",
            definition="""
            多层感知机是由多个全连接仿射变换与非线性激活函数交替堆叠而成的前馈神经网络。它通过逐层特征变换，将输入映射到更适合当前任务的表示空间中。

            在函数逼近视角下，MLP 是一类通用非线性近似器；在表示学习视角下，MLP 通过深度组合逐层构造中间特征。
            """,
            io_spec="""
            输入为定长向量，网络参数包括每一层的权重矩阵和偏置向量。输出层根据任务类型不同而变化：

            - 回归：线性输出
            - 二分类：单 logit / sigmoid
            - 多分类：多 logit / softmax

            与卷积网络和序列网络不同，MLP 不假设局部结构或时间结构，因此最适合处理已向量化的特征输入。
            """,
            architecture="""
            一个标准 MLP 层块由三部分组成：

            1. 线性层：完成特征重组合
            2. 激活函数：引入非线性，使多层堆叠不再退化为单一线性映射
            3. 可选正则化组件：如 Dropout、Norm

            MLP 的关键不是“层数多”，而是“层间表示逐步重构”。因此在分析 MLP 时，应关注隐藏维度、激活函数与深度共同决定的表示容量。
            """,
            objective="""
            MLP 的训练依赖标准反向传播。前向传播产生预测，损失函数定义目标，反向传播计算梯度，优化器执行参数更新。

            在这一过程中，激活函数形状、初始化尺度、学习率和正则化强度都会显著影响训练动态。对于 MLP 这类全连接网络，过拟合和梯度不稳定是两个最常见的问题。
            """,
            complexity="""
            全连接层的参数量与输入维度和隐藏维度乘积成正比，因此在高维输入场景下参数增长很快。

            与 CNN 相比，MLP 不进行参数共享，因此对图像等高维结构化输入的参数效率较低；但在中小规模表格与向量任务中，MLP 仍然是非常重要的基线结构。
            """,
            comparison="""
            与线性模型相比，MLP 通过非线性激活获得更强表达能力。
            与 CNN 相比，MLP 不利用空间局部性。
            与 Transformer 相比，MLP 没有显式的 token 间交互机制。
            因此，MLP 更像深度学习的通用基础单元，而不是针对某类结构化数据的专用架构。
            """,
        ),
        architecture_diagram_cell("mlp"),
        *teaching_cells(
            intuition="""
            MLP 可以理解成“把很多线性变换和非线性激活堆起来”。

            如果只有一层线性变换，模型只能学到比较简单的关系。
            但如果你让模型经过多次“线性变换 + 激活函数”，它就能逐层提炼更复杂的表示。

            一个很实用的直觉是：

            - 前面层学的是比较粗糙的模式
            - 后面层学的是更抽象、更任务相关的模式

            这就是“深度”真正带来的价值。
            """,
            architecture="""
            MLP 的每一层都做同样的事：

            1. 接收上一层表示
            2. 做线性变换
            3. 通过激活函数引入非线性

            如果没有激活函数，多层线性层叠起来仍然等价于一层线性变换，深度就没有意义。

            所以：

            - 线性层负责学习“怎么组合特征”
            - 激活函数负责让模型能表达弯曲、复杂的关系
            - Dropout 等模块负责防止模型记死训练集
            """,
            training="""
            深度学习训练的最基本闭环是：

            1. 前向传播：输入经过网络得到预测
            2. 计算损失：预测和真实答案差多少
            3. 反向传播：自动求每个参数的梯度
            4. 参数更新：优化器根据梯度调整参数

            这一套流程一旦真正理解，后面 CNN、RNN、Transformer 只是“网络结构不同”，训练逻辑并没有变。
            """,
            usage="""
            MLP 适合：

            - 向量化后的中小规模数据
            - 作为深度学习入门模型
            - 想理解 PyTorch 基本训练范式时

            它不擅长显式利用空间结构和时序结构，所以在图像和序列任务上往往不是最优结构。
            """,
            pitfalls="""
            常见误区：

            1. **误以为层数越多一定越强**
               更深的网络更难训练，也更容易过拟合。

            2. **误以为只要调大 hidden_dim 就行**
               更大的宽度会增加参数量和过拟合风险。

            3. **误以为 MLP 学不好图像是因为数据不够**
               更核心的原因是它没有利用图像的局部空间结构。
            """,
        ),
        md(
            r"""
            ## 1. 感知机与多层感知机

            单个神经元可以写成：

            $$
            z = w^T x + b,\qquad a = \phi(z)
            $$
            """
        ),
        code(common_style_code(include_torch=True)),
        code(
            """
            x = torch.linspace(-2, 2, 50).unsqueeze(1)
            y = 3 * x + 1

            w = torch.randn(1, requires_grad=True)
            b = torch.randn(1, requires_grad=True)

            lr = 0.1
            losses = []
            for _ in range(100):
                pred = x * w + b
                loss = ((pred - y) ** 2).mean()
                loss.backward()
                with torch.no_grad():
                    w -= lr * w.grad
                    b -= lr * b.grad
                w.grad.zero_()
                b.grad.zero_()
                losses.append(loss.item())

            print("学习到的 w:", round(w.item(), 4))
            print("学习到的 b:", round(b.item(), 4))
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            axes[0].plot(losses, color="darkorange")
            axes[0].set_title("Autograd 示例的损失下降曲线")

            with torch.no_grad():
                pred_line = x * w + b
            axes[1].scatter(x.numpy(), y.numpy(), label="真实数据", alpha=0.7)
            axes[1].plot(x.numpy(), pred_line.numpy(), color="red", label="拟合直线")
            axes[1].set_title("Autograd 拟合结果")
            axes[1].legend()
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            from sklearn.datasets import load_digits
            from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler

            digits = load_digits()
            X = digits.data.astype(np.float32)
            y = digits.target.astype(np.int64)

            scaler = StandardScaler()
            X = scaler.fit_transform(X).astype(np.float32)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            X_train_tensor = torch.tensor(X_train)
            y_train_tensor = torch.tensor(y_train)
            X_test_tensor = torch.tensor(X_test)
            y_test_tensor = torch.tensor(y_test)

            train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
            test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=256, shuffle=False)
            """
        ),
        code(
            """
            class ShallowMLP(nn.Module):
                def __init__(self, input_dim=64, hidden_dim=64, num_classes=10):
                    super().__init__()
                    self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes))

                def forward(self, x):
                    return self.net(x)


            class DeepMLP(nn.Module):
                def __init__(self, input_dim=64, hidden_dims=(128, 64), num_classes=10, dropout=0.2):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(input_dim, hidden_dims[0]),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dims[0], hidden_dims[1]),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dims[1], num_classes),
                    )

                def forward(self, x):
                    return self.net(x)


            def train_classifier(model, train_loader, test_loader, epochs=25, lr=1e-3):
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

                for _ in range(epochs):
                    model.train()
                    running_loss, running_correct, total = 0.0, 0, 0
                    for batch_x, batch_y in train_loader:
                        optimizer.zero_grad()
                        logits = model(batch_x)
                        loss = criterion(logits, batch_y)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item() * batch_x.size(0)
                        running_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                        total += batch_x.size(0)

                    model.eval()
                    test_loss, test_correct, test_total = 0.0, 0, 0
                    with torch.no_grad():
                        for batch_x, batch_y in test_loader:
                            logits = model(batch_x)
                            loss = criterion(logits, batch_y)
                            test_loss += loss.item() * batch_x.size(0)
                            test_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                            test_total += batch_x.size(0)

                    history["train_loss"].append(running_loss / total)
                    history["train_acc"].append(running_correct / total)
                    history["test_loss"].append(test_loss / test_total)
                    history["test_acc"].append(test_correct / test_total)

                return history
            """
        ),
        code(
            """
            mlp_models = {"ShallowMLP": ShallowMLP(), "DeepMLP": DeepMLP()}
            histories = {}
            trained_models = {}

            for name, model in mlp_models.items():
                history = train_classifier(model, train_loader, test_loader, epochs=25, lr=1e-3)
                histories[name] = history
                trained_models[name] = model
                print(name, "最终测试准确率:", round(history["test_acc"][-1], 4))
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            for name, history in histories.items():
                axes[0].plot(history["train_loss"], label=f"{name} train")
                axes[0].plot(history["test_loss"], linestyle="--", label=f"{name} test")
                axes[1].plot(history["test_acc"], label=name)
            axes[0].set_title("MLP 训练 / 测试损失曲线")
            axes[0].legend()
            axes[1].set_title("MLP 测试准确率曲线")
            axes[1].legend()
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            best_name = max(histories, key=lambda name: histories[name]["test_acc"][-1])
            best_model = trained_models[best_name]

            best_model.eval()
            with torch.no_grad():
                logits = best_model(X_test_tensor)
                preds = logits.argmax(dim=1).numpy()

            print("最佳模型:", best_name)
            print("测试准确率:", round(accuracy_score(y_test, preds), 4))

            plt.figure(figsize=(8, 7))
            ConfusionMatrixDisplay.from_predictions(y_test, preds, cmap="Blues")
            plt.title(f"{best_name} 的混淆矩阵")
            plt.show()
            """
        ),
    ]
    write_notebook("05_PyTorch基础与MLP体系.ipynb", cells)


def cnn_notebook():
    cells = [
        md(
            r"""
            # 06. CNN 体系

            卷积神经网络（CNN）解决了 MLP 在图像任务上的两个关键问题：参数太多、没有利用局部空间结构。
            """
        ),
        *professional_cells(
            model_title="卷积神经网络（CNN）",
            definition="""
            卷积神经网络是一类面向网格结构数据的深度模型，其核心操作是局部感受野上的卷积变换与跨空间位置的参数共享。该结构能够有效提取局部空间模式，并以较低参数代价构造层级特征表示。
            """,
            io_spec="""
            输入通常是二维或三维张量，例如图像张量 $X \in \mathbb{R}^{C \times H \times W}$。卷积层参数以卷积核形式存在，其输出是新的特征图张量。

            经过多层卷积和下采样之后，网络把低层局部纹理逐步变换为高层语义特征，最终由分类头、检测头或分割头完成任务输出。
            """,
            architecture="""
            CNN 的结构可分为特征提取主干与任务头两部分：

            - 特征提取主干：卷积、激活、归一化、池化
            - 任务头：全连接层或其他预测模块

            参数共享与局部连接是 CNN 的关键归纳偏置。前者显著降低参数量，后者使网络天然适应空间邻域模式。
            """,
            objective="""
            CNN 的训练方式与其他深度网络一致，但其表示学习行为具有明显层次性。浅层卷积核通常对边缘、纹理等局部低级模式敏感，深层卷积块则更关注任务相关的高层语义模式。

            在训练中，卷积核、归一化参数和分类头都会通过反向传播共同优化。
            """,
            complexity="""
            CNN 在视觉任务中的优势来自较好的参数效率和空间归纳偏置。相比全连接网络，它的参数量通常显著更低；相比纯注意力结构，它在中小视觉任务上常常更具计算性价比。

            其局限在于：长距离依赖需要依靠更深网络或更大感受野间接建模。
            """,
            comparison="""
            与 MLP 相比，CNN 显式利用空间结构。
            与 Vision Transformer 相比，CNN 具有更强的局部先验，但全局建模通常不如注意力直接。
            在资源受限或样本量不大的视觉任务中，CNN 仍然具有很强的现实优势。
            """,
        ),
        architecture_diagram_cell("cnn"),
        *teaching_cells(
            intuition="""
            CNN 的核心直觉很简单：

            图像不是普通向量，而是有空间结构的网格。

            当你看到一张数字图片时，你不会把 64 个像素点当成 64 个彼此独立的数字；
            你会注意局部笔画、边缘、拐角，再逐步组合成完整数字。

            CNN 正是在做这件事：

            - 先识别局部模式
            - 再把局部模式组合成更高级特征
            """,
            architecture="""
            CNN 的关键组件有三个：

            - 卷积层：用共享卷积核扫过图像，提取局部特征
            - 激活函数：让表示变得非线性
            - 池化层：压缩空间尺寸，增强鲁棒性

            前层卷积核常学到边缘、方向、局部纹理；
            后层会逐渐学到更抽象的笔画组合或局部形状。

            这就是为什么 CNN 对图像比 MLP 更自然。
            """,
            training="""
            训练 CNN 和训练 MLP 的流程本质上完全一致，
            不同点只在于前向传播时使用了卷积和池化。

            真正需要理解的是：卷积核不是手工写好的，它们也是参数，会通过反向传播自动学出来。

            所以 CNN 的强大之处，不是“卷积这个运算本身很神奇”，
            而是“它给模型施加了更适合图像的结构归纳偏置”。
            """,
            usage="""
            CNN 适合：

            - 图像分类
            - 图像检测与分割的主干特征提取
            - 局部模式很重要的二维或一维信号任务

            即使在 Transformer 很流行之后，CNN 仍然是视觉任务里非常重要的基础。
            """,
            pitfalls="""
            常见误区：

            1. **误以为卷积核是在做固定滤波**
               在深度学习里，卷积核是可学习参数，不是手工滤波器。

            2. **误以为池化一定越多越好**
               池化太强会丢失细节。

            3. **误以为 CNN 已经过时**
               它在很多中小视觉任务、边缘设备、轻量模型中依然非常强。
            """,
        ),
        md(
            r"""
            ## 1. 卷积的公式

            $$
            Y(i, j) = \sum_m \sum_n X(i+m, j+n) K(m, n)
            $$
            """
        ),
        code(common_style_code(include_torch=True)),
        code(
            """
            from sklearn.datasets import load_digits
            from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
            from sklearn.model_selection import train_test_split

            digits = load_digits()
            X = digits.images.astype(np.float32) / 16.0
            y = digits.target.astype(np.int64)
            X = X[:, None, :, :]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=64, shuffle=True)
            test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=256, shuffle=False)
            """
        ),
        code(
            """
            class LinearImageBaseline(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(nn.Flatten(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))

                def forward(self, x):
                    return self.net(x)


            class SimpleCNN(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(1, 16, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(16, 32, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                    )
                    self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32 * 2 * 2, 64), nn.ReLU(), nn.Linear(64, 10))

                def forward(self, x):
                    return self.classifier(self.features(x))


            class CNNWithBatchNorm(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(1, 16, kernel_size=3, padding=1),
                        nn.BatchNorm2d(16),
                        nn.ReLU(),
                        nn.Conv2d(16, 32, kernel_size=3, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                    )
                    self.classifier = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(64 * 2 * 2, 64),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, 10),
                    )

                def forward(self, x):
                    return self.classifier(self.features(x))


            def train_model(model, train_loader, test_loader, epochs=20, lr=1e-3):
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                history = {"test_loss": [], "test_acc": []}

                for _ in range(epochs):
                    model.train()
                    for batch_x, batch_y in train_loader:
                        optimizer.zero_grad()
                        logits = model(batch_x)
                        loss = criterion(logits, batch_y)
                        loss.backward()
                        optimizer.step()

                    model.eval()
                    test_loss, test_correct, test_total = 0.0, 0, 0
                    with torch.no_grad():
                        for batch_x, batch_y in test_loader:
                            logits = model(batch_x)
                            loss = criterion(logits, batch_y)
                            test_loss += loss.item() * batch_x.size(0)
                            test_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                            test_total += batch_x.size(0)

                    history["test_loss"].append(test_loss / test_total)
                    history["test_acc"].append(test_correct / test_total)
                return history
            """
        ),
        code(
            """
            cnn_models = {
                "LinearBaseline": LinearImageBaseline(),
                "SimpleCNN": SimpleCNN(),
                "CNN+BatchNorm": CNNWithBatchNorm(),
            }

            cnn_histories = {}
            trained_cnn_models = {}
            for name, model in cnn_models.items():
                history = train_model(model, train_loader, test_loader, epochs=20, lr=1e-3)
                cnn_histories[name] = history
                trained_cnn_models[name] = model
                print(name, "最终测试准确率:", round(history["test_acc"][-1], 4))
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            for name, history in cnn_histories.items():
                axes[0].plot(history["test_loss"], label=name)
                axes[1].plot(history["test_acc"], label=name)
            axes[0].set_title("CNN 测试损失曲线")
            axes[1].set_title("CNN 测试准确率曲线")
            axes[0].legend()
            axes[1].legend()
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            best_name = max(cnn_histories, key=lambda name: cnn_histories[name]["test_acc"][-1])
            best_model = trained_cnn_models[best_name]

            best_model.eval()
            with torch.no_grad():
                logits = best_model(torch.tensor(X_test))
                preds = logits.argmax(dim=1).numpy()

            print("最佳模型:", best_name)
            print("测试准确率:", round(accuracy_score(y_test, preds), 4))

            plt.figure(figsize=(8, 7))
            ConfusionMatrixDisplay.from_predictions(y_test, preds, cmap="Blues")
            plt.title(f"{best_name} 的混淆矩阵")
            plt.show()
            """
        ),
    ]
    write_notebook("06_CNN体系.ipynb", cells)


def rnn_notebook():
    cells = [
        md(
            r"""
            # 07. RNN / LSTM / GRU 体系

            序列模型的核心问题是：当前输出不仅依赖当前输入，还依赖过去的信息。
            """
        ),
        *professional_cells(
            model_title="RNN / LSTM / GRU",
            definition="""
            循环神经网络是一类通过隐状态递推处理序列数据的模型。其核心思想是使用历史隐藏状态作为当前时刻计算的一部分，从而实现对时间依赖或序列依赖的建模。

            LSTM 与 GRU 是为缓解标准 RNN 长依赖建模困难而提出的门控循环结构。
            """,
            io_spec="""
            输入通常为序列张量 $X = (x_1, x_2, \dots, x_T)$。模型维护随时间变化的隐藏状态 $h_t$，并在需要时输出逐步预测或最终时刻预测。

            LSTM 额外维护细胞状态 $c_t$；GRU 则通过更紧凑的门控机制简化状态更新。
            """,
            architecture="""
            标准 RNN 单元通过当前输入与上一时刻隐藏状态生成新的隐藏状态。LSTM 在此基础上引入输入门、遗忘门、输出门，从而对信息写入、保留与暴露进行细粒度控制；GRU 则使用更新门和重置门实现更简洁的门控递推。

            这些门控机制并非附加修饰，而是解决长距离梯度传播困难的核心结构设计。
            """,
            objective="""
            训练通过时间反向传播进行。误差信号不仅沿网络深度传播，还会沿时间维度传播，因此优化难度通常高于前馈网络。

            梯度消失和梯度爆炸是该类模型的典型问题，这也是门控结构、梯度裁剪与学习率控制的重要性所在。
            """,
            complexity="""
            RNN 家族在时间步上具有天然递归依赖，因此并行性弱于 Transformer。其单步状态更新开销较低，但长序列吞吐通常受限。

            在中短序列预测、小规模时间序列和资源有限场景中，RNN / LSTM / GRU 仍然具有实际价值。
            """,
            comparison="""
            与 MLP 相比，RNN 显式建模时间依赖。
            与 Transformer 相比，RNN 的长程依赖建模和并行能力较弱，但状态递推结构更紧凑、在小模型场景下更经济。
            与 SSM 相比，RNN 家族更传统，也更容易用门控视角理解序列记忆机制。
            """,
        ),
        architecture_diagram_cell("rnn"),
        *teaching_cells(
            intuition="""
            序列问题和图像不同。图像是“同时出现”的空间结构，序列是“按时间展开”的结构。

            例如一句话“我今天很开心”，你读到“开心”时，含义依赖前面“我今天很”。

            RNN 家族的核心目标就是：

            - 每处理一个时间步，就把当前信息和历史记忆融合起来
            - 让模型带着“上下文状态”继续往后读
            """,
            architecture="""
            普通 RNN 会维护一个隐藏状态 `h_t`。
            每到新时间步，它都会用当前输入 `x_t` 和上一步隐藏状态 `h_{t-1}` 共同计算新的 `h_t`。

            LSTM 和 GRU 的区别在于：它们发现“直接更新隐藏状态太粗暴”，于是引入门控机制：

            - 哪些旧信息该忘
            - 哪些新信息该写入
            - 哪些信息该输出

            所以门控结构本质上是在做“记忆管理”。
            """,
            training="""
            序列模型训练时会把误差沿时间反向传播，这叫做 BPTT（Backpropagation Through Time）。

            问题也正出在这里：

            - 序列太长时，梯度会越来越小，模型记不住远处信息
            - 或者梯度会变得很大，训练不稳定

            这就是为什么 LSTM / GRU 和梯度裁剪如此重要。
            """,
            usage="""
            RNN 家族适合：

            - 中短序列时间序列预测
            - 小规模序列建模任务
            - 需要理解“状态递推”思想时

            现代大模型里，RNN 已经不是主角，但它仍然是理解序列建模历史和基本问题的关键。
            """,
            pitfalls="""
            常见误区：

            1. **误以为 LSTM 只是“更深的 RNN”**
               它的关键不是更深，而是门控记忆机制。

            2. **误以为序列长就一定要用 RNN**
               长序列场景现在常常优先考虑 Transformer 或 SSM。

            3. **误以为损失下降慢就是模型不行**
               序列模型往往训练更敏感，超参数和梯度稳定性非常关键。
            """,
        ),
        md(
            r"""
            ## 1. RNN 的递推公式

            $$
            h_t = \tanh(W_x x_t + W_h h_{t-1} + b)
            $$
            """
        ),
        code(common_style_code(include_torch=True)),
        code(
            """
            time = np.linspace(0, 80, 1600)
            series = np.sin(time) + 0.35 * np.sin(3 * time) + 0.10 * np.random.randn(len(time))

            plt.figure(figsize=(14, 5))
            plt.plot(time[:300], series[:300], color="steelblue")
            plt.title("合成时间序列（局部）")
            plt.show()
            """
        ),
        code(
            """
            def create_sequences(values, seq_len=24):
                X, y = [], []
                for i in range(len(values) - seq_len):
                    X.append(values[i : i + seq_len])
                    y.append(values[i + seq_len])
                return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

            seq_len = 24
            X, y = create_sequences(series, seq_len=seq_len)
            X = X[:, :, None]

            n_train = int(len(X) * 0.7)
            n_val = int(len(X) * 0.15)

            X_train, y_train = X[:n_train], y[:n_train]
            X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
            X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]

            train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=64, shuffle=True)
            val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=128, shuffle=False)
            test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=128, shuffle=False)
            """
        ),
        code(
            """
            class SequenceRegressor(nn.Module):
                def __init__(self, cell_type="RNN", input_size=1, hidden_size=32):
                    super().__init__()
                    if cell_type == "RNN":
                        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
                    elif cell_type == "LSTM":
                        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
                    elif cell_type == "GRU":
                        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
                    else:
                        raise ValueError(cell_type)
                    self.head = nn.Linear(hidden_size, 1)

                def forward(self, x):
                    out, _ = self.rnn(x)
                    return self.head(out[:, -1, :]).squeeze(-1)


            def train_sequence_model(model, train_loader, val_loader, epochs=35, lr=1e-3):
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                history = {"val_loss": []}

                for _ in range(epochs):
                    model.train()
                    for batch_x, batch_y in train_loader:
                        optimizer.zero_grad()
                        pred = model(batch_x)
                        loss = criterion(pred, batch_y)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                    model.eval()
                    val_loss, val_total = 0.0, 0
                    with torch.no_grad():
                        for batch_x, batch_y in val_loader:
                            pred = model(batch_x)
                            loss = criterion(pred, batch_y)
                            val_loss += loss.item() * batch_x.size(0)
                            val_total += batch_x.size(0)
                    history["val_loss"].append(val_loss / val_total)
                return history
            """
        ),
        code(
            """
            seq_models = {"RNN": SequenceRegressor("RNN"), "LSTM": SequenceRegressor("LSTM"), "GRU": SequenceRegressor("GRU")}
            seq_histories = {}
            trained_seq_models = {}

            for name, model in seq_models.items():
                history = train_sequence_model(model, train_loader, val_loader, epochs=35, lr=1e-3)
                seq_histories[name] = history
                trained_seq_models[name] = model
                print(name, "最终验证损失:", round(history["val_loss"][-1], 6))
            """
        ),
        code(
            """
            plt.figure(figsize=(12, 6))
            for name, history in seq_histories.items():
                plt.plot(history["val_loss"], label=name)
            plt.title("不同序列模型的验证损失曲线")
            plt.legend()
            plt.show()
            """
        ),
        code(
            """
            def predict_series(model, loader):
                model.eval()
                preds, targets = [], []
                with torch.no_grad():
                    for batch_x, batch_y in loader:
                        preds.append(model(batch_x).numpy())
                        targets.append(batch_y.numpy())
                return np.concatenate(preds), np.concatenate(targets)

            test_records = []
            test_predictions = {}
            targets = None
            for name, model in trained_seq_models.items():
                preds, current_targets = predict_series(model, test_loader)
                mse = np.mean((preds - current_targets) ** 2)
                test_records.append({"模型": name, "Test MSE": mse})
                test_predictions[name] = preds
                targets = current_targets

            test_df = pd.DataFrame(test_records).sort_values("Test MSE")
            test_df
            """
        ),
        code(
            """
            best_name = test_df.iloc[0]["模型"]
            best_preds = test_predictions[best_name]

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            sns.barplot(data=test_df, x="模型", y="Test MSE", ax=axes[0], palette="mako")
            axes[0].set_title("RNN / LSTM / GRU 的测试误差对比")

            axes[1].plot(targets[:200], label="真实值", color="black")
            axes[1].plot(best_preds[:200], label=f"{best_name} 预测值", color="crimson", alpha=0.8)
            axes[1].set_title(f"{best_name} 在测试集上的预测效果")
            axes[1].legend()

            plt.tight_layout()
            plt.show()
            """
        ),
    ]
    write_notebook("07_RNN_LSTM_GRU体系.ipynb", cells)


def transformer_notebook():
    cells = [
        md(
            r"""
            # 08. Transformer 体系

            Transformer 的关键在于让序列中的每个位置直接和其他位置建立联系。
            """
        ),
        *professional_cells(
            model_title="Transformer",
            definition="""
            Transformer 是一种基于注意力机制的序列建模架构，其核心在于通过可学习的 Query、Key、Value 交互直接建模任意位置之间的依赖关系。与依赖递推的循环网络不同，Transformer 可以在单层内部实现全局上下文交互。
            """,
            io_spec="""
            输入为 token 序列的嵌入表示及位置信息。每个 block 内部包含注意力子层和前馈子层，输出仍为逐 token 表示。对于 Decoder-only 模型，最终表示再映射到词表 logits，用于下一 token 预测。
            """,
            architecture="""
            典型 Transformer Block 包含：

            - 多头自注意力：负责 token 间的信息交换
            - 前馈网络：负责 token 内部特征变换
            - 残差连接：保证深层优化稳定
            - 归一化层：稳定训练动力学

            多头机制使模型能够在同一层并行学习不同类型的依赖模式，而位置编码则保证模型保留序列顺序信息。
            """,
            objective="""
            语言建模场景中，Transformer 常以 next-token prediction 为目标，通过交叉熵训练。由于注意力机制使每个 token 都能直接访问上下文，因此模型更容易学习长程依赖和复杂语法语义结构。

            在大规模预训练中，优化器、学习率调度、归一化形式与注意力实现会共同决定训练稳定性和扩展上限。
            """,
            complexity="""
            标准自注意力在序列长度维度上的复杂度为 $O(n^2)$，这既带来了强大的全局建模能力，也带来了长上下文代价高昂的问题。

            因此，现代 Transformer 演化出多种工程改进，包括 RoPE、KV Cache、GQA、FlashAttention 等，用于降低推理成本和改善长上下文效率。
            """,
            comparison="""
            与 RNN 相比，Transformer 更易并行、长依赖建模更强。
            与 CNN 相比，Transformer 缺乏强局部先验，但更擅长全局关系建模。
            与 MoE 结合后，Transformer 又可以进一步扩展为稀疏激活的大模型架构。
            """,
        ),
        architecture_diagram_cell("transformer"),
        *teaching_cells(
            intuition="""
            Transformer 最重要的突破，是它不再要求信息必须一步一步传递。

            在 RNN 里，如果第 1 个词想影响第 50 个词，信息必须经过 49 次传递；
            但在 Transformer 里，第 50 个词可以直接“看”第 1 个词。

            所以它最核心的优点是：

            - 更容易建模长距离依赖
            - 更适合并行计算
            - 对现代硬件更友好
            """,
            architecture="""
            Transformer Block 通常包含两大部分：

            - Attention：让 token 之间彼此交流
            - FFN：让每个 token 在自己的特征空间里进一步变换

            多头注意力的含义也很重要：

            - 一个头可能更关注位置关系
            - 一个头可能更关注语义关联
            - 一个头可能更关注局部搭配

            你可以把多头理解成“同时用多种角度看同一句话”。
            """,
            training="""
            Transformer 的训练流程本身并不神秘，还是前向、损失、反向传播、更新参数。

            真正需要理解的是：

            - 它的表示能力来自 token 间的全局交互
            - 它的代价也来自 token 间的全局交互

            也就是说，Transformer 强的地方和贵的地方，其实是同一件事：
            注意力机制。
            """,
            usage="""
            Transformer 适合：

            - 文本建模
            - 多模态任务
            - 长依赖序列建模
            - 需要大规模预训练的任务

            它之所以成为大语言模型主干，不是偶然，而是因为它在表达能力、并行性和扩展性之间取得了很好的平衡。
            """,
            pitfalls="""
            常见误区：

            1. **误以为 Transformer 只有 Attention**
               FFN、Norm、Residual 也同样关键。

            2. **误以为注意力权重就是完整解释**
               注意力可以提供线索，但不等于完整因果解释。

            3. **误以为 Transformer 不需要位置信息**
               如果没有位置编码，它根本不知道 token 的顺序。
            """,
        ),
        md(
            r"""
            ## 1. Self-Attention 的核心公式

            $$
            \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
            $$
            """
        ),
        code(common_style_code(include_torch=True)),
        code(
            """
            from collections import Counter
            from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
            from sklearn.model_selection import train_test_split

            positive_words = ["good", "great", "excellent", "helpful", "enjoyable"]
            negative_words = ["bad", "terrible", "boring", "awful", "annoying"]
            nouns = ["movie", "course", "service", "book", "game", "app"]

            def build_samples():
                samples = []
                for noun in nouns:
                    for word in positive_words:
                        samples.append((f"{noun} is {word}", 1))
                        samples.append((f"{noun} is very {word}", 1))
                        samples.append((f"{noun} is not {word}", 0))
                    for word in negative_words:
                        samples.append((f"{noun} is {word}", 0))
                        samples.append((f"{noun} is really {word}", 0))
                        samples.append((f"{noun} is not {word}", 1))
                random.shuffle(samples)
                return samples

            samples = build_samples()
            texts = [text for text, label in samples]
            labels = [label for text, label in samples]
            """
        ),
        code(
            """
            def tokenize(text):
                return text.split()

            counter = Counter()
            for text in texts:
                counter.update(tokenize(text))

            vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2}
            for token, _ in counter.most_common():
                vocab[token] = len(vocab)

            max_len = max(len(tokenize(text)) for text in texts) + 1

            def encode_text(text):
                tokens = ["[CLS]"] + tokenize(text)
                ids = [vocab.get(token, vocab["[UNK]"]) for token in tokens]
                pad_len = max_len - len(ids)
                ids = ids + [vocab["[PAD]"]] * pad_len
                mask = [1] * len(tokens) + [0] * pad_len
                return ids, mask

            encoded = [encode_text(text) for text in texts]
            input_ids = np.array([item[0] for item in encoded], dtype=np.int64)
            attention_mask = np.array([item[1] for item in encoded], dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)

            X_train_ids, X_test_ids, X_train_mask, X_test_mask, y_train, y_test, train_texts, test_texts = train_test_split(
                input_ids, attention_mask, labels, texts, test_size=0.2, random_state=42, stratify=labels
            )

            train_loader = DataLoader(
                TensorDataset(torch.tensor(X_train_ids), torch.tensor(X_train_mask), torch.tensor(y_train)),
                batch_size=32,
                shuffle=True,
            )
            test_loader = DataLoader(
                TensorDataset(torch.tensor(X_test_ids), torch.tensor(X_test_mask), torch.tensor(y_test)),
                batch_size=64,
                shuffle=False,
            )
            """
        ),
        code(
            """
            class PositionalEncoding(nn.Module):
                def __init__(self, d_model, max_len=64):
                    super().__init__()
                    pe = torch.zeros(max_len, d_model)
                    position = torch.arange(0, max_len).unsqueeze(1).float()
                    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                    pe[:, 0::2] = torch.sin(position * div_term)
                    pe[:, 1::2] = torch.cos(position * div_term)
                    self.register_buffer("pe", pe.unsqueeze(0))

                def forward(self, x):
                    return x + self.pe[:, : x.size(1)]


            class TransformerClassifier(nn.Module):
                def __init__(self, vocab_size, d_model=64, num_heads=4, ff_dim=128, num_classes=2):
                    super().__init__()
                    self.embedding = nn.Embedding(vocab_size, d_model)
                    self.pos_encoding = PositionalEncoding(d_model, max_len=max_len + 2)
                    self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=0.1, batch_first=True)
                    self.norm1 = nn.LayerNorm(d_model)
                    self.ffn = nn.Sequential(nn.Linear(d_model, ff_dim), nn.ReLU(), nn.Linear(ff_dim, d_model))
                    self.norm2 = nn.LayerNorm(d_model)
                    self.classifier = nn.Linear(d_model, num_classes)

                def forward(self, input_ids, attention_mask):
                    x = self.embedding(input_ids)
                    x = self.pos_encoding(x)
                    key_padding_mask = attention_mask == 0
                    attn_out, attn_weights = self.attention(
                        x, x, x, key_padding_mask=key_padding_mask, need_weights=True, average_attn_weights=False
                    )
                    x = self.norm1(x + attn_out)
                    x = self.norm2(x + self.ffn(x))
                    return self.classifier(x[:, 0, :]), attn_weights


            def train_transformer(model, train_loader, test_loader, epochs=25, lr=2e-3):
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

                for _ in range(epochs):
                    model.train()
                    train_loss, train_correct, train_total = 0.0, 0, 0
                    for batch_ids, batch_mask, batch_y in train_loader:
                        optimizer.zero_grad()
                        logits, _ = model(batch_ids, batch_mask)
                        loss = criterion(logits, batch_y)
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item() * batch_ids.size(0)
                        train_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                        train_total += batch_ids.size(0)

                    model.eval()
                    test_loss, test_correct, test_total = 0.0, 0, 0
                    with torch.no_grad():
                        for batch_ids, batch_mask, batch_y in test_loader:
                            logits, _ = model(batch_ids, batch_mask)
                            loss = criterion(logits, batch_y)
                            test_loss += loss.item() * batch_ids.size(0)
                            test_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                            test_total += batch_ids.size(0)

                    history["train_loss"].append(train_loss / train_total)
                    history["train_acc"].append(train_correct / train_total)
                    history["test_loss"].append(test_loss / test_total)
                    history["test_acc"].append(test_correct / test_total)
                return history
            """
        ),
        code(
            """
            transformer = TransformerClassifier(vocab_size=len(vocab))
            transformer_history = train_transformer(transformer, train_loader, test_loader, epochs=25, lr=2e-3)
            print("最终测试准确率:", round(transformer_history["test_acc"][-1], 4))
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            axes[0].plot(transformer_history["train_loss"], label="train loss")
            axes[0].plot(transformer_history["test_loss"], label="test loss")
            axes[0].set_title("Transformer 损失曲线")
            axes[0].legend()

            axes[1].plot(transformer_history["train_acc"], label="train acc")
            axes[1].plot(transformer_history["test_acc"], label="test acc")
            axes[1].set_title("Transformer 准确率曲线")
            axes[1].legend()
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            transformer.eval()
            test_ids_tensor = torch.tensor(X_test_ids)
            test_mask_tensor = torch.tensor(X_test_mask)
            with torch.no_grad():
                logits, attn_weights = transformer(test_ids_tensor, test_mask_tensor)
                preds = logits.argmax(dim=1).numpy()

            print("测试准确率:", round(accuracy_score(y_test, preds), 4))
            plt.figure(figsize=(8, 7))
            ConfusionMatrixDisplay.from_predictions(y_test, preds, cmap="Blues")
            plt.title("Transformer 文本分类混淆矩阵")
            plt.show()
            """
        ),
        code(
            """
            candidate_idx = next(i for i, text in enumerate(test_texts) if " not " in text)
            sample_text = test_texts[candidate_idx]
            sample_ids = torch.tensor(X_test_ids[candidate_idx : candidate_idx + 1])
            sample_mask = torch.tensor(X_test_mask[candidate_idx : candidate_idx + 1])

            with torch.no_grad():
                sample_logits, sample_attn = transformer(sample_ids, sample_mask)
                sample_pred = sample_logits.argmax(dim=1).item()

            valid_len = int(sample_mask.sum().item())
            tokens = ["[CLS]"] + sample_text.split()
            cls_attention = sample_attn[0, 0, 0, :valid_len].numpy()

            plt.figure(figsize=(10, 4))
            sns.heatmap(
                cls_attention.reshape(1, -1),
                annot=np.round(cls_attention.reshape(1, -1), 2),
                cmap="YlGnBu",
                xticklabels=tokens,
                yticklabels=["[CLS] query"],
            )
            plt.title(f"样本文本: '{sample_text}' | 预测标签: {sample_pred}")
            plt.show()
            """
        ),
    ]
    write_notebook("08_Transformer体系.ipynb", cells)


def optimizer_loss_notebook():
    cells = [
        md(
            r"""
            # 09. 优化器与损失函数体系

            很多初学者把“模型结构”当成深度学习的全部，但训练效果往往同样取决于两件事：

            1. **损失函数**决定模型究竟在优化什么。
            2. **优化器**决定模型如何沿着梯度更新参数。

            同一个网络结构，换一个损失函数或优化器，训练行为可能完全不同。
            """
        ),
        *teaching_cells(
            intuition="""
            优化器和损失函数可以分别理解成：

            - 损失函数：告诉模型“什么叫做学得好”
            - 优化器：告诉模型“应该怎么改参数才能变好”

            很多人把模型结构当成主角，但在真实训练里，这两者几乎和结构同样重要。

            例如：

            - 你可以有一个很强的网络，但损失函数不适合，模型会学偏
            - 你可以有一个合理的损失函数，但优化器不稳定，模型会收敛很慢甚至发散
            """,
            architecture="""
            这一节虽然不讲“网络层结构”，但其实讲的是训练系统的底层架构。

            训练闭环里有四个连续步骤：

            1. 前向传播得到预测
            2. 损失函数把预测变成一个可优化的标量
            3. 反向传播得到梯度
            4. 优化器根据梯度更新参数

            这四步里，优化器和损失函数正好占了中间两环。
            """,
            training="""
            损失函数本质上定义了梯度长什么样；优化器本质上决定了你怎么使用这些梯度。

            例如：

            - MSE 会对大误差施加更强惩罚
            - MAE 对异常值更稳
            - Cross Entropy 会强烈推动正确类别概率上升
            - Adam 会为不同参数自适应调整步长

            所以“学得快不快、稳不稳、会不会过拟合”，并不只由网络结构决定。
            """,
            usage="""
            经验上可以这样选：

            - 小中规模任务：Adam / AdamW 往往是很好的默认选项
            - 视觉大模型后期微调：SGD + Momentum 仍很常见
            - 回归含异常值：优先考虑 Huber
            - 类别不平衡检测：可以考虑 Focal Loss
            """,
            pitfalls="""
            常见误区：

            1. **误以为 Adam 永远最好**
               它通常收敛更快，但最终泛化不一定总赢。

            2. **误以为损失下降更快就一定更好**
               有时只是优化得更快，不代表最终测试集更优。

            3. **误以为 Cross Entropy 和 Accuracy 是同一件事**
               一个是训练目标，一个是评价指标，它们相关但不等价。
            """,
        ),
        md(
            r"""
            ## 1. 常见优化器公式

            ### SGD

            $$
            \theta_{t+1} = \theta_t - \eta g_t
            $$

            其中 $g_t = \nabla_\theta \mathcal{L}(\theta_t)$，$\eta$ 是学习率。

            ### Momentum

            $$
            v_{t+1} = \beta v_t + g_t,\qquad \theta_{t+1} = \theta_t - \eta v_{t+1}
            $$

            作用：在一致方向上累积速度，降低震荡。

            ### RMSProp

            $$
            s_{t+1} = \rho s_t + (1-\rho) g_t^2,\qquad
            \theta_{t+1} = \theta_t - \eta \frac{g_t}{\sqrt{s_{t+1}} + \epsilon}
            $$

            作用：为不同参数分配自适应步长。

            ### Adam

            $$
            m_{t+1} = \beta_1 m_t + (1-\beta_1) g_t
            $$

            $$
            v_{t+1} = \beta_2 v_t + (1-\beta_2) g_t^2
            $$

            $$
            \hat{m}_{t+1} = \frac{m_{t+1}}{1-\beta_1^{t+1}},\qquad
            \hat{v}_{t+1} = \frac{v_{t+1}}{1-\beta_2^{t+1}}
            $$

            $$
            \theta_{t+1} = \theta_t - \eta \frac{\hat{m}_{t+1}}{\sqrt{\hat{v}_{t+1}} + \epsilon}
            $$

            ### AdamW

            AdamW 和 Adam 的关键区别是：**权重衰减从梯度中解耦**。

            $$
            \theta_{t+1} = \theta_t - \eta \frac{\hat{m}_{t+1}}{\sqrt{\hat{v}_{t+1}} + \epsilon} - \eta \lambda \theta_t
            $$

            在现代 Transformer 训练中，AdamW 是非常常见的默认选择。
            """
        ),
        md(
            r"""
            ## 2. 常见损失函数公式

            ### 回归损失

            - MSE
              $$
              \mathcal{L}_{\mathrm{MSE}} = \frac{1}{n}\sum_i (y_i - \hat{y}_i)^2
              $$
            - MAE
              $$
              \mathcal{L}_{\mathrm{MAE}} = \frac{1}{n}\sum_i |y_i - \hat{y}_i|
              $$
            - Huber
              $$
              \mathcal{L}_{\delta}(r)=
              \begin{cases}
              \frac{1}{2}r^2,& |r|\le \delta \\
              \delta(|r|-\frac{1}{2}\delta),& |r|>\delta
              \end{cases}
              $$

            其中 $r = y - \hat{y}$。

            ### 分类损失

            - Cross Entropy
              $$
              \mathcal{L}_{\mathrm{CE}} = -\sum_k y_k \log p_k
              $$
            - Binary Cross Entropy
              $$
              \mathcal{L}_{\mathrm{BCE}} = -\left[y\log p + (1-y)\log(1-p)\right]
              $$
            - Focal Loss
              $$
              \mathcal{L}_{\mathrm{Focal}} = -(1-p_t)^\gamma \log p_t
              $$

            Focal Loss 会降低“简单样本”的权重，强调困难样本，因此常用于类别不平衡检测任务。
            """
        ),
        code(common_style_code(include_torch=True)),
        code(
            """
            # 可视化不同回归损失对残差的惩罚形状。
            residuals = np.linspace(-3, 3, 400)
            mse = residuals ** 2
            mae = np.abs(residuals)
            delta = 1.0
            huber = np.where(
                np.abs(residuals) <= delta,
                0.5 * residuals ** 2,
                delta * (np.abs(residuals) - 0.5 * delta),
            )

            plt.figure(figsize=(11, 6))
            plt.plot(residuals, mse, label="MSE", linewidth=3)
            plt.plot(residuals, mae, label="MAE", linewidth=3)
            plt.plot(residuals, huber, label="Huber(delta=1.0)", linewidth=3)
            plt.title("不同回归损失对残差的惩罚曲线")
            plt.xlabel("残差 r = y - y_hat")
            plt.ylabel("loss")
            plt.legend()
            plt.show()
            """
        ),
        code(
            """
            # 分类场景：比较交叉熵与 Focal Loss 对“置信度”的响应。
            p = np.linspace(1e-4, 0.9999, 300)
            ce = -np.log(p)
            gamma = 2.0
            focal = -(1 - p) ** gamma * np.log(p)

            plt.figure(figsize=(11, 6))
            plt.plot(p, ce, label="Cross Entropy", linewidth=3)
            plt.plot(p, focal, label="Focal Loss (gamma=2)", linewidth=3)
            plt.title("正样本预测概率 p 下的 CE / Focal 损失")
            plt.xlabel("模型对正确类别的预测概率 p")
            plt.ylabel("loss")
            plt.legend()
            plt.show()
            """
        ),
        md(
            r"""
            ## 3. 在同一任务上比较优化器

            下面我们在 `digits` 数据集上训练相同的 MLP，仅仅替换优化器：

            - SGD
            - SGD + Momentum
            - RMSprop
            - Adam
            - AdamW

            这样可以更直观地看到“收敛速度”和“最终精度”的差异。
            """
        ),
        code(
            """
            from sklearn.datasets import load_digits
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler

            digits = load_digits()
            X = digits.data.astype(np.float32)
            y = digits.target.astype(np.int64)

            X = StandardScaler().fit_transform(X).astype(np.float32)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            train_loader = DataLoader(
                TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                batch_size=64,
                shuffle=True,
            )
            test_loader = DataLoader(
                TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
                batch_size=256,
                shuffle=False,
            )


            class OptimizerMLP(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(64, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 10),
                    )

                def forward(self, x):
                    return self.net(x)


            def build_optimizer(name, params):
                if name == "SGD":
                    return torch.optim.SGD(params, lr=0.05)
                if name == "Momentum":
                    return torch.optim.SGD(params, lr=0.03, momentum=0.9)
                if name == "RMSprop":
                    return torch.optim.RMSprop(params, lr=0.001)
                if name == "Adam":
                    return torch.optim.Adam(params, lr=0.001)
                if name == "AdamW":
                    return torch.optim.AdamW(params, lr=0.001, weight_decay=1e-2)
                raise ValueError(name)


            def train_with_optimizer(optimizer_name, epochs=15):
                model = OptimizerMLP()
                optimizer = build_optimizer(optimizer_name, model.parameters())
                criterion = nn.CrossEntropyLoss()
                history = {"train_loss": [], "test_acc": []}

                for _ in range(epochs):
                    model.train()
                    total_loss, total_count = 0.0, 0
                    for batch_x, batch_y in train_loader:
                        optimizer.zero_grad()
                        logits = model(batch_x)
                        loss = criterion(logits, batch_y)
                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item() * batch_x.size(0)
                        total_count += batch_x.size(0)

                    model.eval()
                    all_preds = []
                    with torch.no_grad():
                        for batch_x, _ in test_loader:
                            all_preds.append(model(batch_x).argmax(dim=1))

                    preds = torch.cat(all_preds).numpy()
                    history["train_loss"].append(total_loss / total_count)
                    history["test_acc"].append(accuracy_score(y_test, preds))

                return model, history
            """
        ),
        code(
            """
            optimizer_names = ["SGD", "Momentum", "RMSprop", "Adam", "AdamW"]
            opt_histories = {}
            trained_opt_models = {}

            for name in optimizer_names:
                model, history = train_with_optimizer(name, epochs=15)
                trained_opt_models[name] = model
                opt_histories[name] = history
                print(name, "最终测试准确率:", round(history["test_acc"][-1], 4))
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            for name, history in opt_histories.items():
                axes[0].plot(history["train_loss"], label=name)
                axes[1].plot(history["test_acc"], label=name)

            axes[0].set_title("不同优化器的训练损失曲线")
            axes[0].set_xlabel("epoch")
            axes[0].set_ylabel("loss")
            axes[0].legend()

            axes[1].set_title("不同优化器的测试准确率曲线")
            axes[1].set_xlabel("epoch")
            axes[1].set_ylabel("accuracy")
            axes[1].legend()

            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            optimizer_summary = pd.DataFrame(
                [
                    {
                        "优化器": name,
                        "最终训练损失": history["train_loss"][-1],
                        "最终测试准确率": history["test_acc"][-1],
                    }
                    for name, history in opt_histories.items()
                ]
            ).sort_values("最终测试准确率", ascending=False)

            optimizer_summary
            """
        ),
        md(
            r"""
            ## 4. 什么时候用什么优化器 / 损失函数

            ### 优化器选择建议

            - `SGD`：适合理解优化本质，也常见于大规模视觉训练的后期精调。
            - `Momentum`：当损失面震荡明显时，比纯 SGD 更稳。
            - `RMSprop`：历史上常用于 RNN 等训练不稳定场景。
            - `Adam`：中小规模任务、快速原型、默认强基线。
            - `AdamW`：现代 Transformer / ViT / LLM 训练中最常用。

            ### 损失函数选择建议

            - `MSE`：误差服从高斯噪声、希望大误差被更强惩罚。
            - `MAE`：对异常值更鲁棒，但梯度不如 MSE 平滑。
            - `Huber`：想兼顾稳定性和鲁棒性时常用。
            - `CrossEntropy`：多分类默认首选。
            - `BCEWithLogitsLoss`：二分类、多标签分类默认首选。
            - `Focal Loss`：类别极不平衡、困难样本稀少时有价值。
            """
        ),
    ]
    write_notebook("09_优化器与损失函数体系.ipynb", cells)


def training_stability_notebook():
    cells = [
        md(
            r"""
            # 10. 训练稳定性与正则化技巧

            深度学习中很多“模型效果不好”的问题，本质上不是模型结构不够强，而是训练不稳定：

            - 梯度爆炸 / 梯度消失
            - 参数初始化不合理
            - 学习率不合适
            - 过拟合
            - 归一化与正则化缺失

            这一节把这些工程上极其关键的细节系统化。
            """
        ),
        *teaching_cells(
            intuition="""
            很多时候，模型学不好不是因为“模型不够先进”，而是因为训练过程本身不稳定。

            你可以把训练想成在山谷里下山：

            - 初始化决定你从哪里出发
            - 学习率决定你步子迈多大
            - 归一化决定地面是否平整
            - 正则化决定你会不会为了贴合训练集而走偏

            所以这节内容虽然不花哨，但是真正决定工程成败。
            """,
            architecture="""
            这里讲的不是一个单独模型，而是任何深度网络都会用到的训练组件：

            - 初始化
            - 归一化
            - Dropout
            - Weight Decay
            - Gradient Clipping
            - Learning Rate Scheduler

            它们共同组成了“训练稳定性工具箱”。
            """,
            training="""
            训练稳定性问题通常有几个典型症状：

            - 损失一开始就爆炸
            - 损失下降极慢
            - 训练集很好但测试集很差
            - 梯度非常大或非常小

            解决这类问题时，优秀工程师往往不是盲目换模型，而是先排查：

            1. 初始化是否合理
            2. 学习率是否合理
            3. 归一化是否缺失
            4. 是否需要正则化
            """,
            usage="""
            实践建议非常直接：

            - ReLU 网络优先 He 初始化
            - 深层网络优先加 Norm
            - 易过拟合任务优先考虑 Dropout / Weight Decay
            - 序列模型优先加 Gradient Clipping
            - 学习率不确定时，先从更小值开始
            """,
            pitfalls="""
            常见误区：

            1. **误以为正则化越强越好**
               太强会让模型根本学不动。

            2. **误以为 BatchNorm 只是“做标准化”**
               它还会改变优化动力学，提高训练稳定性。

            3. **误以为训练不稳定就一定是代码 bug**
               很多时候只是初始化、学习率、归一化配置不合理。
            """,
        ),
        md(
            r"""
            ## 1. 参数初始化

            设线性层为 $y = Wx$。

            如果初始化尺度过大，激活值和梯度会逐层放大；
            如果尺度过小，激活值和梯度会逐层衰减。

            常见初始化：

            - Xavier / Glorot：适合 `tanh` 等对称激活
              $$
              \mathrm{Var}(W) \approx \frac{2}{n_{\mathrm{in}} + n_{\mathrm{out}}}
              $$
            - He Initialization：适合 `ReLU`
              $$
              \mathrm{Var}(W) \approx \frac{2}{n_{\mathrm{in}}}
              $$
            """
        ),
        md(
            r"""
            ## 2. 归一化与正则化

            ### BatchNorm

            $$
            \hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}},\qquad
            y = \gamma \hat{x} + \beta
            $$

            ### LayerNorm

            LayerNorm 不是在 batch 维度上统计，而是在特征维度上统计，因此在 Transformer 中非常常见。

            ### Dropout

            Dropout 会随机把部分神经元置零：

            $$
            \tilde{h} = m \odot h,\qquad m_i \sim \mathrm{Bernoulli}(1-p)
            $$

            ### Weight Decay

            通过约束权重大小抑制过拟合。

            ### Gradient Clipping

            当梯度范数过大时，进行裁剪：

            $$
            g \leftarrow g \cdot \min\left(1,\frac{\tau}{\|g\|}\right)
            $$
            """
        ),
        code(common_style_code(include_torch=True)),
        code(
            """
            # 观察不同初始化方式下，多层 ReLU 网络的激活统计量。
            class DeepProbeNet(nn.Module):
                def __init__(self, width=256, depth=8):
                    super().__init__()
                    layers = []
                    for _ in range(depth):
                        layers.append(nn.Linear(width, width))
                        layers.append(nn.ReLU())
                    self.net = nn.Sequential(*layers)

                def forward(self, x):
                    activations = []
                    for layer in self.net:
                        x = layer(x)
                        if isinstance(layer, nn.ReLU):
                            activations.append(x.detach().std().item())
                    return activations


            def init_model(model, mode):
                for module in model.modules():
                    if isinstance(module, nn.Linear):
                        if mode == "small":
                            nn.init.normal_(module.weight, mean=0.0, std=0.01)
                        elif mode == "xavier":
                            nn.init.xavier_uniform_(module.weight)
                        elif mode == "he":
                            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                        else:
                            raise ValueError(mode)
                        nn.init.zeros_(module.bias)


            x = torch.randn(512, 256)
            init_records = []
            for init_name in ["small", "xavier", "he"]:
                probe = DeepProbeNet()
                init_model(probe, init_name)
                stds = probe(x)
                for layer_id, std in enumerate(stds, start=1):
                    init_records.append({"初始化": init_name, "层": layer_id, "激活标准差": std})

            init_df = pd.DataFrame(init_records)
            """
        ),
        code(
            """
            plt.figure(figsize=(11, 6))
            sns.lineplot(data=init_df, x="层", y="激活标准差", hue="初始化", marker="o")
            plt.title("不同初始化方式下，多层网络的激活尺度变化")
            plt.show()
            """
        ),
        code(
            """
            from sklearn.datasets import load_digits
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler

            digits = load_digits()
            X = StandardScaler().fit_transform(digits.data.astype(np.float32)).astype(np.float32)
            y = digits.target.astype(np.int64)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            train_loader = DataLoader(
                TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                batch_size=64,
                shuffle=True,
            )
            test_loader = DataLoader(
                TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
                batch_size=256,
                shuffle=False,
            )


            class PlainMLP(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(64, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 10),
                    )

                def forward(self, x):
                    return self.net(x)


            class StableMLP(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(64, 128),
                        nn.BatchNorm1d(128),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(128, 128),
                        nn.BatchNorm1d(128),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(128, 10),
                    )

                def forward(self, x):
                    return self.net(x)


            def train_regularized_model(model, weight_decay=0.0, epochs=20):
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

                history = {"train_loss": [], "test_acc": [], "lr": []}

                for _ in range(epochs):
                    model.train()
                    total_loss, total_count = 0.0, 0
                    for batch_x, batch_y in train_loader:
                        optimizer.zero_grad()
                        logits = model(batch_x)
                        loss = criterion(logits, batch_y)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item() * batch_x.size(0)
                        total_count += batch_x.size(0)

                    model.eval()
                    preds = []
                    with torch.no_grad():
                        for batch_x, _ in test_loader:
                            preds.append(model(batch_x).argmax(dim=1))
                    preds = torch.cat(preds).numpy()

                    history["train_loss"].append(total_loss / total_count)
                    history["test_acc"].append(accuracy_score(y_test, preds))
                    history["lr"].append(optimizer.param_groups[0]["lr"])
                    scheduler.step()

                return history
            """
        ),
        code(
            """
            regularization_setups = {
                "PlainMLP": (PlainMLP(), 0.0),
                "StableMLP": (StableMLP(), 1e-2),
            }

            reg_histories = {}
            for name, (model, wd) in regularization_setups.items():
                history = train_regularized_model(model, weight_decay=wd, epochs=20)
                reg_histories[name] = history
                print(name, "最终测试准确率:", round(history["test_acc"][-1], 4))
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            for name, history in reg_histories.items():
                axes[0].plot(history["train_loss"], label=name)
                axes[1].plot(history["test_acc"], label=name)
                axes[2].plot(history["lr"], label=name)

            axes[0].set_title("训练损失")
            axes[1].set_title("测试准确率")
            axes[2].set_title("学习率调度曲线")

            for ax in axes:
                ax.legend()

            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            # 梯度裁剪的直观示例：同一个梯度，在裁剪前后范数如何变化。
            toy_model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))
            toy_x = torch.randn(128, 16)
            toy_y = 20 * torch.randn(128, 1)

            pred = toy_model(toy_x)
            loss = ((pred - toy_y) ** 2).mean() * 50
            loss.backward()

            before_clip = torch.sqrt(
                sum((param.grad.detach() ** 2).sum() for param in toy_model.parameters())
            ).item()

            nn.utils.clip_grad_norm_(toy_model.parameters(), max_norm=1.0)

            after_clip = torch.sqrt(
                sum((param.grad.detach() ** 2).sum() for param in toy_model.parameters())
            ).item()

            print("裁剪前梯度范数:", round(before_clip, 4))
            print("裁剪后梯度范数:", round(after_clip, 4))
            """
        ),
        md(
            r"""
            ## 3. 实战建议

            - `ReLU` 网络优先尝试 He 初始化。
            - CNN 中常见 `BatchNorm`，Transformer 中更常见 `LayerNorm` / `RMSNorm`。
            - 如果训练集精度很高、测试集明显偏低，优先考虑 Dropout / Weight Decay / 数据增强。
            - 如果训练一开始就震荡或发散，先降学习率，再检查初始化和归一化。
            - 对 RNN / LLM 长序列训练，梯度裁剪通常是必需项。
            """
        ),
    ]
    write_notebook("10_训练稳定性与正则化技巧.ipynb", cells)


def llm_moe_notebook():
    cells = [
        md(
            r"""
            # 11. 大语言模型架构与 MoE 体系

            这一节的目标不是“会调用 API”，而是系统理解现代大语言模型的内部结构：

            1. 现代 LLM 为什么大多采用 **Decoder-only Transformer**
            2. RoPE、KV Cache、GQA / MQA、SwiGLU、RMSNorm 等结构为什么流行
            3. **MoE（Mixture of Experts）** 为什么能让模型在总参数很大时仍保持较低激活成本
            4. 当前代表性的开源 / 开放权重大模型家族在架构上到底有什么差异
            """
        ),
        *professional_cells(
            model_title="现代大语言模型与 MoE",
            definition="""
            现代大语言模型通常指基于大规模自回归预训练的 Decoder-only Transformer 及其变体。其目标是在大规模文本分布上学习条件概率模型：

            $$
            P(x_1, \dots, x_T)=\prod_{t=1}^{T} P(x_t \mid x_{<t})
            $$

            MoE 则是在 Transformer 前馈层中引入稀疏专家路由机制的结构变体，其核心目的是在扩大总参数规模的同时控制单 token 的激活计算量。
            """,
            io_spec="""
            输入为离散 token 序列，经嵌入和位置编码后进入多层 Transformer Block。每层的注意力子层负责上下文交互，前馈子层负责逐 token 非线性映射。

            在 MoE 版本中，标准稠密 FFN 被替换为“路由器 + 多专家前馈网络”的组合。每个 token 只进入少量专家，因此总参数与激活参数分离。
            """,
            architecture="""
            现代 LLM 的关键结构组件通常包括：

            - Token Embedding
            - Rotary Position Embedding
            - Multi-Head / Grouped-Query Attention
            - RMSNorm
            - SwiGLU 或其他门控前馈层
            - KV Cache 友好的推理设计

            在 MoE 架构中，注意力层一般保持稠密，而前馈层采用稀疏专家结构，这是因为前馈层通常占据了主要参数量，最适合做稀疏扩容。
            """,
            objective="""
            LLM 的核心训练目标仍是最大化序列似然或最小化 next-token cross entropy。MoE 在此基础上额外需要解决负载均衡问题，因此通常引入 auxiliary balancing loss，避免专家使用高度失衡。

            这意味着 MoE 的优化并不只是“把总参数做大”，而是同时优化语言建模能力、路由稳定性与分布式训练效率。
            """,
            complexity="""
            Dense Transformer 的计算和显存成本对参数和上下文长度都十分敏感。MoE 通过稀疏激活降低了每个 token 的实际计算量，但通信成本上升，系统复杂度也更高。

            因此，在评估现代 LLM 架构时，不能只看总参数量，还必须区分：

            - 总参数
            - 激活参数
            - KV Cache 占用
            - 训练通信代价
            - 推理延迟
            """,
            comparison="""
            Dense LLM 的结构更直接、推理路径更稳定；MoE LLM 在相同激活预算下可容纳更大容量，但训练和部署更复杂。
            相比早期 Transformer，现代 LLM 还增加了大量面向长上下文和推理效率的结构细节，因此“现代 LLM 架构”已经是一整套系统设计，而非单一模块。
            """,
        ),
        architecture_diagram_cell("moe"),
        *teaching_cells(
            intuition="""
            现代大语言模型可以先粗略理解成“非常非常大的 Transformer”，
            但真正重要的是：它们不是简单把参数变大，而是在架构上不断为两个目标服务：

            - 提高能力
            - 降低推理与训练成本

            MoE 的出现正是为了回答一个现实问题：

            “如果我想让总参数变大，能不能不要让每个 token 都把所有参数算一遍？”
            """,
            architecture="""
            现代 LLM 主干大多仍是 Decoder-only Transformer，但细节已经发生了很多变化：

            - 位置编码从绝对位置编码转向 RoPE
            - 归一化常用 RMSNorm
            - FFN 常用 SwiGLU / GeGLU 变体
            - 推理时大量依赖 KV Cache
            - 为了降低 KV Cache 成本，引入 GQA / MQA
            - 为了在总参数更大时保持稀疏计算，引入 MoE

            也就是说，现代 LLM 不是“一个点子”，而是一整套为大规模训练与推理服务的架构组合。
            """,
            training="""
            在 dense 模型里，每个 token 每层都会经过同样的 FFN；
            在 MoE 里，每个 token 会先被路由到少数专家，再只经过这些专家。

            这样做的好处是：

            - 总参数可以很大
            - 单次前向的激活参数量却不必同样大

            但训练也更复杂：

            - 路由是否均衡
            - 专家是否都被充分训练
            - 分布式通信是否高效

            这些问题都不是附属问题，而是 MoE 真正的工程难点。
            """,
            usage="""
            如果你的目标是理解现代 LLM，必须先区分几个概念：

            - Dense 和 MoE 的区别
            - 总参数和激活参数的区别
            - 训练效率和推理效率并不完全相同

            学这节时，不要只盯着“参数多少”，更要理解“为什么这样设计”。
            """,
            pitfalls="""
            常见误区：

            1. **误以为 MoE 就一定比 dense 更强**
               它更像一种容量 / 成本折中方案，不是自动加分项。

            2. **误以为总参数越大，推理成本就一定越大**
               对 MoE 来说，关键要看激活参数量。

            3. **误以为开源大模型架构差异只是尺寸不同**
               其实在注意力、归一化、专家路由、上下文扩展、推理优化上都有实质差别。
            """,
        ),
        md(
            r"""
            ## 1. Decoder-only LLM 的主干结构

            现代自回归大语言模型通常按以下顺序堆叠：

            1. Token Embedding
            2. 位置编码（如 RoPE）
            3. 多层 Transformer Block
            4. 输出线性层 / LM Head

            单个 Transformer Block 常见结构：

            $$
            h' = h + \mathrm{Attention}(\mathrm{Norm}(h))
            $$

            $$
            h_{out} = h' + \mathrm{FFN}(\mathrm{Norm}(h'))
            $$

            这就是典型的 **Pre-Norm + Residual** 结构。
            """
        ),
        md(
            r"""
            ## 2. Self-Attention、RoPE、KV Cache、GQA

            ### Self-Attention

            $$
            \mathrm{Attention}(Q, K, V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
            $$

            ### RoPE（Rotary Position Embedding）

            RoPE 的核心思想是：把位置信息编码成对 Query / Key 向量的旋转，使注意力分数天然带有相对位置信息。

            ### KV Cache

            自回归生成时，前面 token 的 Key / Value 不需要每一步都重算，所以会缓存起来：

            - 降低推理延迟
            - 提高长上下文生成效率

            ### MQA / GQA

            - `MQA`：多个 Query 头共享一组 K/V
            - `GQA`：多个 Query 头分组共享 K/V

            它们的主要目标是**减少 KV Cache 占用并加快推理**。
            """
        ),
        md(
            r"""
            ## 3. MoE 的核心公式

            对于输入 token 表示 $h$，路由器先计算每个专家的分数：

            $$
            r = W_r h
            $$

            得到路由概率：

            $$
            p = \mathrm{softmax}(r)
            $$

            选择 Top-$k$ 专家后，输出为：

            $$
            y = \sum_{e \in \mathrm{TopK}(p)} p_e \, E_e(h)
            $$

            其中：

            - $E_e(\cdot)$ 表示第 $e$ 个专家网络
            - 每个 token 只激活少数专家，因此**总参数很大，但单次计算成本更低**

            这也是 MoE 最核心的“稀疏激活”思想。
            """
        ),
        md(
            r"""
            ## 4. MoE 的关键工程问题

            ### 路由不均衡

            如果大部分 token 都涌向少数专家，会导致：

            - 某些专家过载
            - 某些专家几乎学不到东西

            因此 MoE 训练常加入 **load balancing loss**。

            ### 容量限制（capacity）

            每个专家每个 batch 只接收有限 token，超过容量的 token 需要：

            - 丢弃
            - 回退到其他专家
            - 或者采用更复杂的调度机制

            ### 通信开销

            MoE 在分布式训练中常常比 dense 模型更依赖高效的 All-to-All 通信。
            """
        ),
        code(common_style_code(include_torch=True)),
        code(
            """
            # 用一个玩具 MoE 层展示“路由到不同专家”的过程。
            class ToyMoE(nn.Module):
                def __init__(self, hidden_dim=8, expert_dim=16, num_experts=4, top_k=2):
                    super().__init__()
                    self.hidden_dim = hidden_dim
                    self.num_experts = num_experts
                    self.top_k = top_k
                    self.router = nn.Linear(hidden_dim, num_experts)
                    self.experts = nn.ModuleList(
                        [
                            nn.Sequential(
                                nn.Linear(hidden_dim, expert_dim),
                                nn.ReLU(),
                                nn.Linear(expert_dim, hidden_dim),
                            )
                            for _ in range(num_experts)
                        ]
                    )

                def forward(self, x):
                    # x: [tokens, hidden_dim]
                    logits = self.router(x)
                    probs = torch.softmax(logits, dim=-1)
                    topk_prob, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)

                    output = torch.zeros_like(x)
                    for token_id in range(x.size(0)):
                        token_out = 0.0
                        for prob, expert_id in zip(topk_prob[token_id], topk_idx[token_id]):
                            token_out = token_out + prob * self.experts[int(expert_id)](x[token_id : token_id + 1])
                        output[token_id : token_id + 1] = token_out
                    return output, probs, topk_idx


            toy_moe = ToyMoE(hidden_dim=8, expert_dim=16, num_experts=4, top_k=2)
            token_batch = torch.randn(24, 8)
            output, route_prob, topk_idx = toy_moe(token_batch)

            print("输出张量形状:", tuple(output.shape))
            print("路由概率形状:", tuple(route_prob.shape))
            """
        ),
        code(
            """
            # 统计每个专家被选中的次数。
            route_counts = []
            for expert_id in range(route_prob.size(1)):
                count = int((topk_idx == expert_id).sum().item())
                route_counts.append({"专家": f"Expert {expert_id}", "被选中次数": count})

            route_df = pd.DataFrame(route_counts)

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            sns.heatmap(
                route_prob.detach().numpy(),
                cmap="YlGnBu",
                ax=axes[0],
            )
            axes[0].set_title("每个 token 对各专家的路由概率")
            axes[0].set_xlabel("expert")
            axes[0].set_ylabel("token")

            sns.barplot(data=route_df, x="专家", y="被选中次数", palette="crest", ax=axes[1])
            axes[1].set_title("Top-k 路由下各专家的负载")

            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            r"""
            ## 5. 代表性开源 / 开放权重大模型架构对比

            下表基于我核对到的官方公开资料，选取了几类**具有代表性的前沿架构**：

            - Dense Decoder-only：Mistral 7B、Llama 3.1 405B
            - Sparse MoE：Mixtral 8x7B、DeepSeek-V3、Llama 4 Scout
            - 稀疏之外的额外结构创新：例如 DeepSeek-V3 的 MLA、Llama 4 的原生多模态

            这些资料对应的官方页面日期包括：

            - `2023-12-11`：Mixtral 8x7B 官方发布页
            - `2024-12-26`：DeepSeek-V3 官方模型卡
            - `2025-04-05`：Llama 4 官方发布页
            """
        ),
        code(
            """
            llm_df = pd.DataFrame(
                [
                    {
                        "模型": "Mistral 7B v0.3",
                        "发布时间": "2024-05-22",
                        "架构": "Dense Decoder-only Transformer",
                        "注意力 / 推理优化": "GQA + Sliding Window Attention",
                        "FFN / 专家": "Dense MLP",
                        "归一化": "RMSNorm",
                        "上下文": "32k",
                        "总参数": 7,
                        "激活参数": 7,
                        "是否 MoE": "否",
                    },
                    {
                        "模型": "Mixtral 8x7B v0.1",
                        "发布时间": "2023-12-11",
                        "架构": "SMoE Decoder-only Transformer",
                        "注意力 / 推理优化": "Grouped-Query Attention",
                        "FFN / 专家": "8 experts, top-2 routing",
                        "归一化": "RMSNorm",
                        "上下文": "32k",
                        "总参数": 47,
                        "激活参数": 13,
                        "是否 MoE": "是",
                    },
                    {
                        "模型": "Llama 3.1 405B",
                        "发布时间": "2024-07-23",
                        "架构": "Dense Decoder-only Transformer",
                        "注意力 / 推理优化": "GQA + RoPE",
                        "FFN / 专家": "Dense MLP / SwiGLU",
                        "归一化": "RMSNorm",
                        "上下文": "128k",
                        "总参数": 405,
                        "激活参数": 405,
                        "是否 MoE": "否",
                    },
                    {
                        "模型": "DeepSeek-V3",
                        "发布时间": "2024-12-26",
                        "架构": "MoE Decoder-only Transformer",
                        "注意力 / 推理优化": "MLA + RoPE",
                        "FFN / 专家": "671B total, 37B activated",
                        "归一化": "RMSNorm",
                        "上下文": "128k",
                        "总参数": 671,
                        "激活参数": 37,
                        "是否 MoE": "是",
                    },
                    {
                        "模型": "Llama 4 Scout",
                        "发布时间": "2025-04-05",
                        "架构": "Native Multimodal MoE",
                        "注意力 / 推理优化": "MoE + 长上下文设计",
                        "FFN / 专家": "16 experts, 17B active / 109B total",
                        "归一化": "官方未强调变化，保留 Transformer 族设计",
                        "上下文": "10M",
                        "总参数": 109,
                        "激活参数": 17,
                        "是否 MoE": "是",
                    },
                ]
            )

            llm_df
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(18, 6))

            plot_df = llm_df.copy()
            sns.scatterplot(
                data=plot_df,
                x="总参数",
                y="激活参数",
                hue="是否 MoE",
                style="是否 MoE",
                s=220,
                ax=axes[0],
            )
            for _, row in plot_df.iterrows():
                axes[0].text(row["总参数"] + 4, row["激活参数"] + 2, row["模型"], fontsize=10)
            axes[0].set_title("总参数 vs 激活参数")

            moe_ratio = plot_df.assign(激活占比=plot_df["激活参数"] / plot_df["总参数"])
            sns.barplot(data=moe_ratio, x="模型", y="激活占比", palette="viridis", ax=axes[1])
            axes[1].set_title("不同模型的激活参数占比")
            axes[1].tick_params(axis="x", rotation=20)

            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            r"""
            ## 6. 架构差异该如何理解

            ### Dense 模型

            - 优点：训练和推理实现更直接，通信模式更简单
            - 缺点：如果参数继续增大，推理成本线性增加

            ### MoE 模型

            - 优点：在保持较低激活成本的前提下扩大总参数容量
            - 缺点：路由、负载均衡、专家并行和通信实现更复杂

            ### 为什么现代 LLM 越来越重视 GQA / KV Cache / 长上下文

            原因很直接：真实应用的瓶颈往往不是训练 FLOPs，而是**推理延迟、显存占用和长上下文效率**。
            """
        ),
        md(
            r"""
            ## 7. 官方资料链接

            - Llama 4 官方发布页: https://ai.meta.com/blog/llama-4-multimodal-intelligence/
            - Llama 3.1 官方发布页: https://ai.meta.com/blog/meta-llama-3-1/
            - Mixtral 8x7B 官方发布页: https://mistral.ai/news/mixtral-of-experts/
            - Mistral 7B 官方文档: https://docs.mistral.ai/getting-started/models/models_overview/
            - DeepSeek-V3 官方模型卡: https://huggingface.co/deepseek-ai/DeepSeek-V3
            """
        ),
    ]
    write_notebook("11_大语言模型架构与MoE体系.ipynb", cells)


def peft_decoding_notebook():
    cells = [
        md(
            r"""
            # 12. 参数高效微调与推理解码体系

            学完模型结构之后，下一个现实问题是：

            - 模型太大，怎样低成本微调？
            - 推理阶段，怎样控制输出质量、多样性和稳定性？

            这就是 PEFT（参数高效微调）和 decoding（解码策略）要解决的问题。
            """
        ),
        *professional_cells(
            model_title="参数高效微调与推理解码",
            definition="""
            参数高效微调（PEFT）是一类仅更新少量附加参数而非全量底座参数的适配方法，目标是在显著降低显存和计算成本的同时保留大模型迁移能力。推理解码则是在模型给出下一 token 概率分布后，对候选 token 进行搜索或采样的策略集合。
            """,
            io_spec="""
            在 LoRA 中，输入仍经过原始线性层，但额外叠加一个低秩增量分支；在推理解码中，输入是模型生成的 logits 或概率分布，输出是下一个 token 的选取结果及整个序列展开路径。
            """,
            architecture="""
            PEFT 关注的是“如何改模型”；解码关注的是“如何使用模型输出分布”。

            LoRA 的结构是在冻结的权重矩阵旁引入 $BA$ 低秩分支；QLoRA 在此基础上进一步压缩底座参数表示。解码策略则包括 greedy、beam search、temperature sampling、top-k 与 top-p 等，它们并不改变模型参数，只改变输出路径搜索方式。
            """,
            objective="""
            PEFT 训练目标与原任务目标保持一致，区别仅在于可训练参数集合更小。LoRA 通过低秩约束假设“任务适配所需的权重更新位于一个低维子空间中”。

            解码阶段的目标则是平衡：

            - 确定性
            - 多样性
            - 流畅性
            - 可控性
            """,
            complexity="""
            LoRA 的参数量与秩 $r$ 成正比，远小于全量微调；QLoRA 进一步降低显存开销，因此成为消费级 GPU 上微调大模型的重要路线。

            解码复杂度方面，greedy 最便宜，beam search 更昂贵，采样类方法则在质量与多样性间做折中。
            """,
            comparison="""
            与全量微调相比，PEFT 更经济、更灵活，适合多任务并存。
            与 prompt engineering 相比，PEFT 能真正改变参数。
            在解码层面，beam search 更偏搜索最优，top-p / temperature 更偏自然生成。它们不是互相替代关系，而是面向不同输出目标的配置工具。
            """,
        ),
        architecture_diagram_cell("lora"),
        *teaching_cells(
            intuition="""
            大模型真正落地时，最现实的问题不是“它能不能回答”，而是：

            - 我有没有足够算力去训练或微调它？
            - 我能不能控制它输出得更稳定、更自然？

            参数高效微调解决的是“怎么少花资源改模型”；
            解码策略解决的是“同一个模型，怎么让它按你想要的风格输出”。
            """,
            architecture="""
            LoRA 不是换掉原模型，而是在原有线性层旁边加一个很小的低秩分支。

            这意味着：

            - 原模型大部分参数冻结
            - 只训练很少一部分增量参数
            - 推理时再把增量作用叠加回去

            解码策略则发生在模型输出概率之后：

            - 模型先给每个 token 一个概率
            - 你再决定怎么从这些概率里选下一个 token
            """,
            training="""
            PEFT 训练时，真正更新的参数很少，因此：

            - 显存需求更低
            - 训练更快
            - 更适合多任务、多 LoRA 并存

            解码不属于训练，但它极大影响用户看到的效果。
            同一个模型，使用 Greedy、Top-p、Temperature=0.7、Beam Search，输出风格会明显不同。
            """,
            usage="""
            如果你想做业务定制：

            - 数据量不大：优先 LoRA
            - 显存更紧：优先 QLoRA
            - 任务偏格式化：解码可偏保守
            - 任务偏创作：解码可更随机
            """,
            pitfalls="""
            常见误区：

            1. **误以为 LoRA 可以替代所有全量微调**
               对大多数任务够用，但不是所有场景都能完全替代。

            2. **误以为 Temperature 越高越“聪明”**
               它只是更随机，不是更聪明。

            3. **误以为 Beam Search 一定比采样好**
               开放式生成里，Beam Search 反而可能更僵硬。
            """,
        ),
        md(
            r"""
            ## 1. LoRA 的核心思想

            假设原始权重矩阵为 $W \in \mathbb{R}^{d_{out}\times d_{in}}$。

            LoRA 不直接训练整个 $W$，而是只训练一个低秩增量：

            $$
            \Delta W = BA
            $$

            其中：

            - $A \in \mathbb{R}^{r \times d_{in}}$
            - $B \in \mathbb{R}^{d_{out} \times r}$
            - $r \ll \min(d_{in}, d_{out})$

            最终权重：

            $$
            W' = W + \frac{\alpha}{r} BA
            $$

            这样可以显著减少可训练参数数量。
            """
        ),
        md(
            r"""
            ## 2. 常见参数高效微调方法

            - `LoRA`：最常见，工程生态最好
            - `QLoRA`：量化底座模型，再训练低秩增量，显著节省显存
            - `Prompt Tuning`：只学习少量软提示向量
            - `Prefix Tuning`：为注意力层增加可学习前缀
            - `Adapter`：在主干网络中插入小型瓶颈模块

            一般来说：

            - 资源有限时，优先 LoRA / QLoRA
            - 需要最小侵入时，可考虑 Prompt Tuning
            - 需要更强任务适配时，LoRA 往往更实用
            """
        ),
        md(
            r"""
            ## 3. 推理解码策略

            ### Greedy Decoding

            每一步直接选概率最大的 token：

            $$
            x_t = \arg\max_i p(i \mid x_{<t})
            $$

            ### Temperature Sampling

            $$
            p_i' = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
            $$

            - $T < 1$：更保守
            - $T > 1$：更随机

            ### Top-k

            只在概率最高的 $k$ 个 token 中采样。

            ### Top-p / Nucleus Sampling

            选择累计概率达到阈值 $p$ 的最小 token 集合，再在其中采样。

            ### Beam Search

            保留多个候选序列，适合更偏“搜索”的任务，但在开放式生成中未必自然。
            """
        ),
        code(common_style_code(include_torch=True)),
        code(
            """
            # 低秩近似实验：观察 rank 越低，能否用更少参数逼近原矩阵。
            torch.manual_seed(42)
            W = torch.randn(256, 256)
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)

            rank_records = []
            total_params = W.numel()
            for rank in [2, 4, 8, 16, 32, 64, 128]:
                W_approx = (U[:, :rank] * S[:rank]) @ Vh[:rank, :]
                rel_error = torch.norm(W - W_approx) / torch.norm(W)
                lora_params = rank * (256 + 256)
                rank_records.append(
                    {
                        "rank": rank,
                        "相对重建误差": rel_error.item(),
                        "LoRA 参数量": lora_params,
                        "参数占原矩阵比例": lora_params / total_params,
                    }
                )

            rank_df = pd.DataFrame(rank_records)
            rank_df
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            axes[0].plot(rank_df["rank"], rank_df["相对重建误差"], marker="o")
            axes[0].set_title("低秩近似的 rank 与重建误差")
            axes[0].set_xlabel("rank")
            axes[0].set_ylabel("relative reconstruction error")

            axes[1].plot(rank_df["rank"], rank_df["参数占原矩阵比例"], marker="o", color="darkgreen")
            axes[1].set_title("低秩近似的 rank 与参数占比")
            axes[1].set_xlabel("rank")
            axes[1].set_ylabel("parameter ratio")

            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            # 用一个玩具 token 概率分布展示不同解码策略的差异。
            vocab = ["A", "B", "C", "D", "E", "F", "G", "H"]
            logits = torch.tensor([4.2, 3.9, 2.8, 2.3, 1.6, 1.1, 0.2, -0.5], dtype=torch.float32)


            def softmax_with_temperature(logits, temperature=1.0):
                scaled = logits / temperature
                return torch.softmax(scaled, dim=-1)


            def top_k_sample(probs, k=3):
                values, indices = torch.topk(probs, k=k)
                sampled_local = torch.multinomial(values / values.sum(), num_samples=1).item()
                return indices[sampled_local].item()


            def top_p_sample(probs, p=0.8):
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=0)
                keep_mask = cumulative <= p
                keep_mask[0] = True
                kept_probs = sorted_probs[keep_mask]
                kept_indices = sorted_indices[keep_mask]
                sampled_local = torch.multinomial(kept_probs / kept_probs.sum(), num_samples=1).item()
                return kept_indices[sampled_local].item()


            probs_t1 = softmax_with_temperature(logits, temperature=1.0)
            probs_t07 = softmax_with_temperature(logits, temperature=0.7)
            probs_t13 = softmax_with_temperature(logits, temperature=1.3)

            print("Greedy:", vocab[int(torch.argmax(probs_t1))])
            print("Top-k sample:", vocab[top_k_sample(probs_t1, k=3)])
            print("Top-p sample:", vocab[top_p_sample(probs_t1, p=0.8)])
            """
        ),
        code(
            """
            decode_df = pd.DataFrame(
                {
                    "token": vocab,
                    "T=0.7": probs_t07.numpy(),
                    "T=1.0": probs_t1.numpy(),
                    "T=1.3": probs_t13.numpy(),
                }
            )

            decode_long = decode_df.melt(id_vars="token", var_name="temperature", value_name="prob")

            plt.figure(figsize=(12, 6))
            sns.barplot(data=decode_long, x="token", y="prob", hue="temperature")
            plt.title("不同 temperature 下的 token 概率分布")
            plt.show()
            """
        ),
        code(
            """
            # 用简单的序列得分说明 Beam Search 与 Greedy 的差异。
            # 这里只做一个手工构造的树搜索示意，不依赖大模型。
            transitions = {
                "<BOS>": {"我": 0.55, "今天": 0.45},
                "我": {"喜欢": 0.60, "想": 0.40},
                "今天": {"天气": 0.70, "想": 0.30},
                "喜欢": {"学习": 0.65, "跑步": 0.35},
                "想": {"学习": 0.52, "休息": 0.48},
                "天气": {"很好": 0.90, "一般": 0.10},
            }

            def greedy_path():
                path = ["<BOS>"]
                score = 0.0
                current = "<BOS>"
                while current in transitions:
                    next_token, prob = max(transitions[current].items(), key=lambda x: x[1])
                    path.append(next_token)
                    score += math.log(prob)
                    current = next_token
                return path[1:], score

            def beam_search(beam_width=2, max_steps=3):
                beams = [(["<BOS>"], 0.0)]
                for _ in range(max_steps):
                    new_beams = []
                    for path, score in beams:
                        current = path[-1]
                        if current not in transitions:
                            new_beams.append((path, score))
                            continue
                        for next_token, prob in transitions[current].items():
                            new_beams.append((path + [next_token], score + math.log(prob)))
                    beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
                return [(path[1:], score) for path, score in beams]

            print("Greedy 结果:", greedy_path())
            print("Beam Search 结果:", beam_search(beam_width=2, max_steps=3))
            """
        ),
        md(
            r"""
            ## 4. 使用建议

            ### 微调方法

            - `LoRA`：绝大多数私有数据定制任务的第一选择
            - `QLoRA`：显存紧张、只有消费级 GPU 时很有价值
            - `全量微调`：只有在资源足够、任务偏移很大时才考虑

            ### 解码策略

            - `Greedy`：稳定、可重复，适合分类、结构化抽取
            - `Beam Search`：适合机器翻译、摘要等更偏搜索的任务
            - `Top-k / Top-p + Temperature`：适合开放式生成和创作

            实际部署中，很多高质量对话系统都会把：

            - system prompt
            - 采样参数
            - repetition penalty
            - stop tokens

            一起当作“推理配置”的一部分来调优。
            """
        ),
    ]
    write_notebook("12_参数高效微调与推理解码体系.ipynb", cells)


def model_families_notebook():
    cells = [
        md(
            r"""
            # 13. 现代开源模型家族与新架构体系

            现代大模型已经不再是“只有 Transformer 一条路”。

            当前值得重点理解的路线至少包括：

            - **Dense Transformer 家族**：Llama、Qwen、Gemma、OLMo
            - **Sparse MoE 家族**：Mixtral、DeepSeek-V3、Qwen3 MoE
            - **Hybrid SSM-Transformer 家族**：Jamba
            - **State Space Models / Mamba 家族**：Mamba、Mamba-2

            这一节的重点是建立“模型家族视角”，理解它们为什么这样设计。
            """
        ),
        *professional_cells(
            model_title="现代开源模型家族与新架构",
            definition="""
            “模型家族”不是单一 checkpoint，而是一组共享设计哲学、训练路线和部署目标的模型集合。现代开源模型家族的差异主要不再体现在“是否使用 Transformer”，而更多体现在：结构细节、稀疏策略、长上下文能力、生态定位与开放程度。
            """,
            io_spec="""
            不同家族的输入输出接口往往相似，通常都服务于 token 级自回归预测；真正不同的是其内部状态更新机制和系统目标。例如 Dense Transformer 家族更强调通用性，MoE 家族更强调容量扩展，SSM 家族更强调长序列效率。
            """,
            architecture="""
            本节从架构路线而非 benchmark 名次出发，将代表性家族分为：

            - Dense Transformer 家族
            - Sparse MoE Transformer 家族
            - Hybrid SSM-Transformer 家族
            - 纯 SSM / Mamba 家族

            这种划分更有助于理解模型演化逻辑，因为它直接对应了不同架构对“容量、长序列、效率、开放性”的不同回答方式。
            """,
            objective="""
            不同家族虽然共享语言建模目标，但训练和设计目标并不相同：

            - 有的优先追求通用能力与生态覆盖
            - 有的优先追求单位推理成本下的容量
            - 有的优先追求长上下文和吞吐
            - 有的优先追求 fully-open 研究复现

            因此，“为何采用该架构”必须和“该家族想优化什么目标”一起理解。
            """,
            complexity="""
            Dense、MoE、Hybrid、SSM 在复杂度、实现成熟度和系统生态上各有代价。

            - Dense：成熟、稳定，但参数和推理成本耦合更强
            - MoE：容量大，但路由和通信复杂
            - Hybrid：长序列效率更好，但工程链路更复杂
            - SSM：理论效率优越，但生态尚在发展
            """,
            comparison="""
            家族对比的正确方法不是只比较某个 benchmark 数字，而是比较：

            - 核心架构假设
            - 扩展路径
            - 系统代价
            - 开放程度
            - 适用场景

            这也是为什么同样是“开源大模型”，Qwen、Gemma、OLMo、Jamba、Mamba 的学习价值并不相同。
            """,
        ),
        *teaching_cells(
            intuition="""
            学大模型不能只记模型名和参数量。

            更有价值的方式是把模型看成“技术路线”：

            - 有的路线优先做通用 dense 模型
            - 有的路线优先做 MoE 扩容
            - 有的路线试图摆脱标准注意力
            - 有的路线把研究透明度放在首位

            一旦你建立起“家族视角”，以后看到新模型就不会只剩下“又多了一个名字”。
            """,
            architecture="""
            这一节不是讲一个具体网络，而是讲一类模型的设计取向。

            例如：

            - Qwen3 更像完整产品矩阵
            - Gemma 3 更强调轻量部署与开发者友好
            - OLMo 2 更强调 fully-open 科研复现
            - Jamba 试图把 SSM 和 Transformer 融合
            - Mamba 代表非注意力路线的重要探索

            所以这节最重要的问题不是“谁更强”，而是“为什么它们走了不同路”。
            """,
            training="""
            不同架构路线会把训练难点放在不同地方：

            - Dense Transformer：规模化训练、数据与推理优化
            - MoE：路由、通信、专家均衡
            - SSM：算子实现、序列建模能力和生态适配
            - Hybrid：如何同时兼顾两类结构的优势

            学会从训练难点倒推架构目标，是理解模型演化非常重要的方法。
            """,
            usage="""
            这节最适合在你已经学过 Transformer 和 MoE 之后再读。

            读的时候可以重点问自己：

            - 这个家族要解决什么现实问题？
            - 它在哪个维度做了取舍？
            - 它更适合科研、产品、长上下文还是轻量部署？
            """,
            pitfalls="""
            常见误区：

            1. **误以为参数量就是全部**
               很多架构差异体现在训练配方、推理优化和系统设计上。

            2. **误以为新架构一出现就会替代 Transformer**
               现实里新架构通常需要很长时间才能建立完整生态。

            3. **误以为模型家族对比就是 benchmark 对比**
               benchmark 只能说明某个切面，不能代替架构理解。
            """,
        ),
        md(
            r"""
            ## 1. 从架构目标看不同家族

            ### Dense Transformer

            目标：在通用语言建模能力上取得稳定上限，生态最成熟。

            ### MoE Transformer

            目标：在不让单次推理成本线性爆炸的情况下扩大总参数容量。

            ### SSM / Mamba

            目标：缓解注意力的二次复杂度问题，用线性时间建模长序列。

            ### Hybrid SSM-Transformer

            目标：同时保留 Transformer 的高质量建模能力与 SSM 的长序列效率。
            """
        ),
        md(
            r"""
            ## 2. Mamba / SSM 的核心递推公式

            经典状态空间模型可写成：

            $$
            h_t = A h_{t-1} + B x_t
            $$

            $$
            y_t = C h_t + D x_t
            $$

            Mamba 的关键创新之一在于：让参数对输入具有选择性（selective），从而增强表达能力。

            相比标准注意力，SSM 的一个核心卖点是：

            - 计算和显存更接近线性随序列长度增长
            - 更适合超长序列或吞吐优先场景

            但它也面临：

            - 与注意力机制不同的归纳偏置
            - 训练和生态成熟度不如 Transformer 家族
            """
        ),
        code(common_style_code(include_torch=False)),
        code(
            """
            # 对比不同架构在理论复杂度上的增长趋势。
            seq_lengths = np.array([1_000, 2_000, 4_000, 8_000, 16_000, 32_000])
            attn_complexity = seq_lengths ** 2
            linear_complexity = seq_lengths

            complexity_df = pd.DataFrame(
                {
                    "序列长度": seq_lengths,
                    "Attention 复杂度(归一化)": attn_complexity / attn_complexity[0],
                    "Linear/SSM 复杂度(归一化)": linear_complexity / linear_complexity[0],
                }
            )
            complexity_df
            """
        ),
        code(
            """
            plt.figure(figsize=(11, 6))
            plt.plot(complexity_df["序列长度"], complexity_df["Attention 复杂度(归一化)"], marker="o", label="Attention O(n^2)")
            plt.plot(complexity_df["序列长度"], complexity_df["Linear/SSM 复杂度(归一化)"], marker="o", label="SSM / Linear O(n)")
            plt.xscale("log")
            plt.yscale("log")
            plt.title("不同序列建模架构的复杂度增长趋势")
            plt.xlabel("sequence length")
            plt.ylabel("normalized complexity")
            plt.legend()
            plt.show()
            """
        ),
        md(
            r"""
            ## 3. 代表性模型家族对比

            下面这张表整理了几类非常有代表性的现代开源 / 开放权重模型。

            我这里特别用了**绝对日期**来避免“最新”理解偏差：

            - `2024-11-26`：OLMo 2
            - `2025-03-12`：Gemma 3
            - `2025-04-29`：Qwen3
            - `2024-08-22`：Jamba 1.5
            - `2024-05-31`：Mamba-2 论文公开（ICML 2024 版本）
            """
        ),
        code(
            """
            family_df = pd.DataFrame(
                [
                    {
                        "模型家族": "Qwen3",
                        "日期": "2025-04-29",
                        "代表模型": "Qwen3-235B-A22B / Qwen3-32B",
                        "核心架构": "Dense + MoE 并存的 Decoder-only Transformer",
                        "关键机制": "Hybrid thinking, GQA, Dense+MoE lineup",
                        "上下文": "32K-128K",
                        "开源特征": "多尺寸覆盖极广，推理与训练路线都完整",
                        "类别": "Transformer",
                    },
                    {
                        "模型家族": "Gemma 3",
                        "日期": "2025-03-12",
                        "代表模型": "Gemma 3 27B",
                        "核心架构": "轻量 Dense Transformer / 多模态版本",
                        "关键机制": "128K context, vision-language, quantized variants",
                        "上下文": "128K",
                        "开源特征": "单卡友好，强调开发者落地与移动端生态",
                        "类别": "Transformer",
                    },
                    {
                        "模型家族": "OLMo 2",
                        "日期": "2024-11-26",
                        "代表模型": "OLMo-2-13B",
                        "核心架构": "Fully Open Dense Transformer",
                        "关键机制": "开放数据、代码、recipe、checkpoint",
                        "上下文": "官方重点不在超长上下文，而在 fully-open 研究",
                        "开源特征": "最强调 fully-open science",
                        "类别": "Transformer",
                    },
                    {
                        "模型家族": "Jamba 1.5",
                        "日期": "2024-08-22",
                        "代表模型": "Jamba 1.5 Large",
                        "核心架构": "Hybrid SSM-Transformer",
                        "关键机制": "长上下文、高吞吐、SSM + Transformer",
                        "上下文": "长上下文能力是核心卖点",
                        "开源特征": "兼顾长上下文与吞吐",
                        "类别": "Hybrid",
                    },
                    {
                        "模型家族": "Mamba / Mamba-2",
                        "日期": "2023-12-05 / 2024-05-31",
                        "代表模型": "Mamba-2.8B / Mamba2-2.7B",
                        "核心架构": "Selective State Space Models",
                        "关键机制": "线性时间序列建模、SSD duality",
                        "上下文": "理论上更适合长序列伸展",
                        "开源特征": "代表非注意力路线的重要探索",
                        "类别": "SSM",
                    },
                ]
            )

            family_df
            """
        ),
        code(
            """
            category_counts = family_df["类别"].value_counts().reset_index()
            category_counts.columns = ["类别", "数量"]

            plt.figure(figsize=(8, 5))
            sns.barplot(data=category_counts, x="类别", y="数量", palette="Set2")
            plt.title("课程中覆盖的现代模型架构类别")
            plt.show()
            """
        ),
        code(
            """
            # 定性打分，只为帮助建立结构化直觉，不代表官方 benchmark。
            radar_df = pd.DataFrame(
                [
                    {"模型家族": "Qwen3", "长上下文": 4, "效率": 4, "开放程度": 4, "生态成熟度": 5, "新架构探索": 4},
                    {"模型家族": "Gemma 3", "长上下文": 4, "效率": 5, "开放程度": 4, "生态成熟度": 4, "新架构探索": 3},
                    {"模型家族": "OLMo 2", "长上下文": 3, "效率": 3, "开放程度": 5, "生态成熟度": 3, "新架构探索": 3},
                    {"模型家族": "Jamba 1.5", "长上下文": 5, "效率": 5, "开放程度": 4, "生态成熟度": 2, "新架构探索": 5},
                    {"模型家族": "Mamba-2", "长上下文": 5, "效率": 5, "开放程度": 4, "生态成熟度": 2, "新架构探索": 5},
                ]
            )

            score_long = radar_df.melt(id_vars="模型家族", var_name="维度", value_name="分数")
            plt.figure(figsize=(12, 6))
            sns.barplot(data=score_long, x="维度", y="分数", hue="模型家族")
            plt.title("不同模型家族的定性对比")
            plt.ylim(0, 5.5)
            plt.show()
            """
        ),
        md(
            r"""
            ## 4. 如何理解这些家族的技术取向

            ### Qwen3

            更像一个“完整产品矩阵”：

            - 有 dense
            - 有 MoE
            - 有 reasoning / non-thinking 控制
            - 有 agentic coder 路线

            ### Gemma 3

            更强调：

            - 单卡 / 单加速器友好
            - 轻量、高效
            - 方便开发者实际部署

            ### OLMo 2

            最值得学习的不是单点 benchmark，而是：

            - 真正公开 recipe
            - 真正可复现实验路径
            - 更适合科研与教学

            ### Jamba / Mamba

            最值得学习的是它们在问一个更基础的问题：

            “长序列建模是否必须依赖标准注意力？”
            """
        ),
        md(
            r"""
            ## 5. 官方资料链接

            - Qwen3 官方博客（2025-04-29）: https://qwenlm.github.io/blog/qwen3/
            - Gemma 3 官方博客（2025-03-12）: https://blog.google/technology/developers/gemma-3/
            - OLMo 2 官方博客（2024-11-26）: https://allenai.org/blog/olmo2
            - Jamba 1.5 官方博客（2024-08-22）: https://www.ai21.com/blog/announcing-jamba-model-family/
            - Jamba 官方博客（2024-03-28）: https://www.ai21.com/blog/announcing-jamba
            - Mamba 官方仓库: https://github.com/state-spaces/mamba
            - Mamba 论文: https://arxiv.org/abs/2312.00752
            - Mamba-2 / SSD 论文: https://arxiv.org/abs/2405.21060
            """
        ),
    ]
    write_notebook("13_现代开源模型家族与新架构体系.ipynb", cells)


def alignment_notebook():
    cells = [
        md(
            r"""
            # 14. 对齐训练与偏好优化体系

            预训练模型学会了“像互联网文本那样继续写”，但这不等于它已经：

            - 听懂用户意图
            - 遵守格式要求
            - 避免有害回答
            - 稳定输出人类偏好的回答

            因此现代大模型通常会经历一个后训练链路：

            1. SFT
            2. Reward Modeling
            3. RLHF / PPO 或者 DPO / ORPO / KTO 等偏好优化
            """
        ),
        *professional_cells(
            model_title="对齐训练与偏好优化",
            definition="""
            对齐训练是指在预训练语言模型基础上，通过监督微调、偏好建模和策略优化等后训练步骤，使模型行为更接近人类期望的过程。其核心问题不是“让模型继续拟合文本分布”，而是“让模型生成更有用、更安全、更符合偏好的响应”。
            """,
            io_spec="""
            对齐训练的数据形式与预训练不同。预训练主要是无标注连续文本；对齐训练则引入：

            - 指令-回答对
            - 优选 / 劣选回答对
            - 奖励信号
            - 安全或风格标签

            因此，对齐训练首先是数据范式改变，其次才是优化目标改变。
            """,
            architecture="""
            对齐训练的“架构”更准确地说是训练管线架构：

            1. SFT：建立指令跟随能力
            2. Reward Modeling：学习偏好排序函数
            3. RLHF 或直接偏好优化：把模型推向期望行为区域

            这里的重点不在于更换主干网络，而在于逐阶段改变目标函数与监督信号形式。
            """,
            objective="""
            SFT 使用似然最大化；Reward Model 使用排序损失；RLHF 在奖励与 KL 约束之间折中；DPO / ORPO / KTO 则尝试以更直接、更稳定的方式优化偏好。

            因此，对齐训练的本质是“从 token-level likelihood 过渡到 behavior-level preference”。
            """,
            complexity="""
            RLHF 的优势在于目标灵活，但链路长、实现复杂、调参成本高。DPO 等离线方法实现更简洁，数据利用方式更稳定，但能表达的在线探索能力较弱。

            在现代工业实践中，是否采用 RLHF 还是 DPO，并非理论优劣之争，而是资源、数据、工具链和目标函数形式共同决定的工程选择。
            """,
            comparison="""
            与预训练相比，对齐训练关注模型行为而非知识覆盖。
            与单纯 SFT 相比，偏好优化能进一步塑造回答风格和偏序关系。
            与 RLHF 相比，DPO 等方法更易实现；与 DPO 相比，RLHF 在复杂奖励场景下更灵活。
            """,
        ),
        *teaching_cells(
            intuition="""
            预训练模型学到的是“像互联网那样继续写”，
            但用户真正想要的是“按要求、按偏好、按安全边界来回答”。

            对齐训练可以理解成把一个会写文本的模型，逐步变成一个更像助手的模型。

            这通常不是一步完成的，而是要经历：

            - 先学会模仿好回答
            - 再学会区分好回答和差回答
            - 再进一步朝着更符合人类偏好的方向优化
            """,
            architecture="""
            对齐训练不是改网络结构，而是改训练目标和数据形式。

            一个典型链路包含：

            - SFT：用示范数据告诉模型“理想回答长什么样”
            - Reward Model：学习人类偏好的排序信号
            - Preference Optimization：让模型更偏向 preferred answers

            所以这节的“架构”更像训练管线架构，而不是神经网络拓扑。
            """,
            training="""
            对齐训练的难点在于：人类偏好不是单一数值标签，而是复杂、主观、上下文相关的。

            因此工业界才会出现多条路线：

            - RLHF：更灵活，但更复杂
            - DPO：更直接、更稳定
            - ORPO / KTO：尝试进一步简化或改进偏好建模

            学这节时，重要的不是死记公式，而是理解：
            “为什么需要从 next-token prediction 走到 preference optimization？”
            """,
            usage="""
            如果你未来想做：

            - 指令微调
            - 对话助手
            - 偏好优化
            - 安全 / 有用性 / 风格控制

            那这节内容就是必须掌握的基础。
            """,
            pitfalls="""
            常见误区：

            1. **误以为 SFT 就等于对齐完成**
               SFT 只是第一步，很多偏好和安全问题还没有真正解决。

            2. **误以为 RLHF 一定优于 DPO**
               是否更优很依赖数据、算力和工程条件。

            3. **误以为偏好优化只是在刷 benchmark**
               它本质上是在改变模型行为方式。
            """,
        ),
        md(
            r"""
            ## 1. SFT：监督微调

            SFT 的目标很直接：用高质量示范数据让模型学会“该怎么回答”。

            目标函数通常就是标准交叉熵：

            $$
            \mathcal{L}_{\mathrm{SFT}} = - \sum_t \log \pi_\theta(y_t \mid x, y_{<t})
            $$

            SFT 的作用：

            - 让模型从 base model 进入“assistant 模式”
            - 学会指令跟随、格式跟随、角色跟随
            - 作为后续偏好优化的起点
            """
        ),
        md(
            r"""
            ## 2. Reward Model 与 Bradley-Terry

            在 RLHF 中，常见做法是先训练一个奖励模型 $r_\phi(x, y)$。

            如果给定同一 prompt 下的优胜回答 $y_w$ 和劣质回答 $y_l$，则常用 Bradley-Terry 形式：

            $$
            P(y_w \succ y_l \mid x) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))
            $$

            对应损失：

            $$
            \mathcal{L}_{\mathrm{RM}} = -\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))
            $$
            """
        ),
        md(
            r"""
            ## 3. RLHF / PPO 的核心思想

            RLHF 典型目标：

            $$
            \max_\pi \; \mathbb{E}_{y \sim \pi(\cdot \mid x)}[r(x,y)] - \beta \, \mathrm{KL}(\pi \| \pi_{\mathrm{ref}})
            $$

            含义：

            - 让策略更偏向高奖励回答
            - 但不要离参考模型偏得太远

            PPO 常通过 clipped objective 保持更新稳定：

            $$
            \mathcal{L}_{\mathrm{PPO}} =
            \mathbb{E}\left[
            \min\left(
            \rho_t A_t,
            \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t
            \right)
            \right]
            $$

            其中 $\rho_t$ 是新旧策略比值，$A_t$ 是 advantage。
            """
        ),
        md(
            r"""
            ## 4. DPO / ORPO / KTO

            ### DPO

            DPO 直接绕过显式 reward model + RL 过程，常见目标写作：

            $$
            \mathcal{L}_{\mathrm{DPO}} =
            - \log \sigma \left(
            \beta \left[
            \log \pi_\theta(y_w \mid x) - \log \pi_\theta(y_l \mid x)
            - \log \pi_{\mathrm{ref}}(y_w \mid x) + \log \pi_{\mathrm{ref}}(y_l \mid x)
            \right]
            \right)
            $$

            ### ORPO

            ORPO 的核心思想是把偏好优化直接并入 SFT 阶段，用 odds ratio 惩罚 rejected answer。

            ### KTO

            KTO 从 prospect theory 视角出发，把“收益”和“损失”不对称地建模，更强调人类偏好的行为学特性。
            """
        ),
        code(common_style_code(include_torch=True)),
        code(
            """
            # 用 margin 可视化几种常见偏好优化目标。
            delta = np.linspace(-5, 5, 400)
            beta = 1.0

            dpo_loss = -np.log(1 / (1 + np.exp(-beta * delta)))
            hinge_loss = np.maximum(0, 1 - delta)
            logistic_pref_loss = np.log(1 + np.exp(-delta))

            plt.figure(figsize=(11, 6))
            plt.plot(delta, dpo_loss, label="DPO / logistic preference", linewidth=3)
            plt.plot(delta, hinge_loss, label="Margin / hinge-style", linewidth=3)
            plt.plot(delta, logistic_pref_loss, label="BT logistic", linewidth=3, linestyle="--")
            plt.title("偏好 margin 与损失函数的关系")
            plt.xlabel("preferred minus rejected margin")
            plt.ylabel("loss")
            plt.legend()
            plt.show()
            """
        ),
        code(
            """
            # 构造一个玩具 preference 数据集，比较 SFT-only 和 preference-style 目标的效果。
            torch.manual_seed(42)
            np.random.seed(42)

            prompts = ["math", "coding", "writing", "safety"] * 60
            responses = ["strong", "weak"]
            prompt_to_id = {name: i for i, name in enumerate(sorted(set(prompts)))}
            response_to_id = {"strong": 0, "weak": 1}

            X = torch.tensor([prompt_to_id[p] for p in prompts], dtype=torch.long)
            y_pref = torch.tensor([0] * len(prompts), dtype=torch.long)  # 0 表示 strong 更优


            class TinyPreferenceModel(nn.Module):
                def __init__(self, num_prompts, hidden_dim=8):
                    super().__init__()
                    self.embedding = nn.Embedding(num_prompts, hidden_dim)
                    self.scorer = nn.Linear(hidden_dim, 2)

                def forward(self, prompt_ids):
                    h = self.embedding(prompt_ids)
                    return self.scorer(h)


            def train_sft_style(epochs=80):
                model = TinyPreferenceModel(len(prompt_to_id))
                optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
                criterion = nn.CrossEntropyLoss()
                history = []
                for _ in range(epochs):
                    optimizer.zero_grad()
                    logits = model(X)
                    loss = criterion(logits, y_pref)
                    loss.backward()
                    optimizer.step()
                    history.append(loss.item())
                return model, history


            def train_dpo_style(beta=1.0, epochs=80):
                model = TinyPreferenceModel(len(prompt_to_id))
                ref_model = TinyPreferenceModel(len(prompt_to_id))
                ref_model.load_state_dict(model.state_dict())
                optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
                history = []

                for _ in range(epochs):
                    optimizer.zero_grad()
                    logits = model(X)
                    with torch.no_grad():
                        ref_logits = ref_model(X)

                    logp = torch.log_softmax(logits, dim=-1)
                    ref_logp = torch.log_softmax(ref_logits, dim=-1)

                    chosen = logp[:, 0]
                    rejected = logp[:, 1]
                    ref_chosen = ref_logp[:, 0]
                    ref_rejected = ref_logp[:, 1]

                    margin = beta * ((chosen - rejected) - (ref_chosen - ref_rejected))
                    loss = -torch.log(torch.sigmoid(margin)).mean()
                    loss.backward()
                    optimizer.step()
                    history.append(loss.item())
                return model, history
            """
        ),
        code(
            """
            sft_model, sft_history = train_sft_style()
            dpo_model, dpo_history = train_dpo_style()

            plt.figure(figsize=(11, 6))
            plt.plot(sft_history, label="SFT-style loss")
            plt.plot(dpo_history, label="DPO-style loss")
            plt.title("SFT 与 DPO 风格目标的训练曲线")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend()
            plt.show()
            """
        ),
        code(
            """
            with torch.no_grad():
                sft_scores = torch.softmax(sft_model(X[:4]), dim=-1)[:, 0].numpy()
                dpo_scores = torch.softmax(dpo_model(X[:4]), dim=-1)[:, 0].numpy()

            compare_df = pd.DataFrame(
                {
                    "prompt": list(prompt_to_id.keys()),
                    "SFT 对 preferred 的概率": sft_scores,
                    "DPO 对 preferred 的概率": dpo_scores,
                }
            )
            compare_df
            """
        ),
        code(
            """
            compare_long = compare_df.melt(id_vars="prompt", var_name="方法", value_name="概率")
            plt.figure(figsize=(10, 6))
            sns.barplot(data=compare_long, x="prompt", y="概率", hue="方法")
            plt.title("不同训练目标对 preferred response 的偏好程度")
            plt.ylim(0, 1.0)
            plt.show()
            """
        ),
        md(
            r"""
            ## 5. 什么时候用什么方法

            - `SFT`：任何对齐链路的基础步骤，几乎总是先做
            - `RLHF / PPO`：当你需要显式 reward、在线探索、复杂交互目标时更灵活
            - `DPO`：当前最常见的离线偏好优化强基线，简单稳定
            - `ORPO`：希望减少 reference model 依赖，训练流程更简洁时可考虑
            - `KTO`：当偏好信号形式更接近“好 / 不好”二元标签时有吸引力

            实践里并不存在“一个方法永远最好”。
            更准确的理解是：

            - 资源少、数据是离线 pairwise preference：优先 DPO / ORPO
            - 有稳定 reward pipeline 和在线采样能力：RLHF / PPO 更灵活
            """
        ),
        md(
            r"""
            ## 6. 主要论文与官方资料

            - InstructGPT / RLHF（NeurIPS 2022）:
              https://papers.nips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html
            - DPO（2023-05-29）:
              https://arxiv.org/abs/2305.18290
            - ORPO（2024-03-12）:
              https://arxiv.org/abs/2403.07691
            - KTO（2024-02-02）:
              https://arxiv.org/abs/2402.01306
            - SimPO（2024-05-23）:
              https://arxiv.org/abs/2405.14734
            """
        ),
    ]
    write_notebook("14_对齐训练与偏好优化体系.ipynb", cells)


def inference_system_notebook():
    cells = [
        md(
            r"""
            # 15. 推理系统与部署优化体系

            训练一个模型只是开始。真实落地时更常见的瓶颈是：

            - 显存不够
            - 延迟太高
            - 吞吐太低
            - 长上下文成本太高

            因此，理解推理系统优化并不是“部署细节”，而是现代大模型能力的一部分。
            """
        ),
        *professional_cells(
            model_title="推理系统与部署优化",
            definition="""
            推理系统优化是指围绕已训练好的模型，在显存、吞吐、时延和并发约束下，对执行路径、缓存布局、数值表示和调度策略进行系统级重构的过程。它研究的不是“模型会不会答”，而是“模型能否在现实资源边界内高效地答”。
            """,
            io_spec="""
            输入是请求流、上下文序列和模型权重；系统状态包括 KV Cache、批处理队列、量化权重、调度器状态与缓存分页结构；输出是生成 token 流和相应的资源消耗特征。
            """,
            architecture="""
            现代大模型推理栈通常包含以下关键子系统：

            - 权重加载与量化执行层
            - 注意力 kernel 与 KV Cache 管理层
            - 请求调度与连续批处理层
            - 解码优化层，如 speculative decoding

            因此推理系统本质上是“模型计算图 + 缓存管理 + 调度器”的复合架构。
            """,
            objective="""
            推理优化的目标不是单一指标最大化，而是在多个系统指标间做折中：

            - 降低首 token 时延
            - 提高 tokens/s 吞吐
            - 提高显存利用率
            - 控制精度损失

            这也是为什么量化、PagedAttention、Continuous Batching 和 Speculative Decoding 会被同时讨论，因为它们分别对应不同瓶颈。
            """,
            complexity="""
            对于长上下文大模型，注意力与 KV Cache 成本通常是主瓶颈。随着上下文增长，显存占用会迅速上升；随着并发上升，调度效率成为吞吐上限的关键。

            因此，系统优化常常带来比“单纯换更大 GPU”更高的边际收益。
            """,
            comparison="""
            与训练优化相比，推理优化更关注在线服务约束；与单模型架构改造相比，推理优化更强调系统资源利用率。
            在实际生产中，模型架构、量化策略与调度系统往往必须协同设计，而不能彼此割裂。
            """,
        ),
        *teaching_cells(
            intuition="""
            很多人以为“大模型能力”只由参数和数据决定。
            但在真实产品里，用户感受到的往往是：

            - 回得快不快
            - 长上下文撑不撑得住
            - 成本能不能接受
            - 并发高时会不会崩

            这些问题大多属于推理系统，而不是模型公式本身。
            """,
            architecture="""
            推理系统可以理解成围绕模型主干搭起来的一整套基础设施：

            - KV Cache 管理
            - 请求调度
            - 批处理与连续批处理
            - 量化执行
            - 注意力 kernel 优化
            - speculative decoding

            所以同一个模型，在不同推理系统里，成本和速度可能差很多。
            """,
            training="""
            这一节虽然不训练大模型，但它和训练并不脱节。

            因为很多架构设计之所以出现，就是为了更好的推理：

            - GQA / MQA 减少 KV cache
            - MoE 控制激活计算量
            - 更轻量的量化格式降低显存

            也就是说，现代模型架构越来越不是“只为训练而设计”，而是训练和推理一起考虑。
            """,
            usage="""
            如果你未来想做线上部署，这节几乎是必修：

            - 显存紧张时优先考虑量化
            - 长上下文时优先考虑 KV cache 设计
            - 高并发服务时优先考虑 continuous batching
            - 低时延场景时优先考虑 speculative decoding
            """,
            pitfalls="""
            常见误区：

            1. **误以为换更大 GPU 就能解决所有问题**
               系统调度和 cache 管理常常同样关键。

            2. **误以为量化只会带来精度损失**
               在合适方法下，精度损失可能很有限，但收益非常大。

            3. **误以为推理优化只是工程问题**
               它会反过来影响模型架构选择和产品策略。
            """,
        ),
        md(
            r"""
            ## 1. KV Cache 显存公式

            对于 Decoder-only 模型，KV Cache 大致随下面因素线性增长：

            $$
            \mathrm{Memory} \propto 2 \times L \times T \times H_{kv} \times d_{head} \times \mathrm{bytes}
            $$

            其中：

            - $L$：层数
            - $T$：上下文长度
            - $H_{kv}$：KV heads 数
            - $d_{head}$：每个 head 的维度
            - `2`：因为要存 K 和 V

            这也是为什么：

            - GQA / MQA 很重要
            - 长上下文成本很高
            - KV Cache 管理决定了推理系统上限
            """
        ),
        md(
            r"""
            ## 2. 量化

            一个常见的均匀量化形式是：

            $$
            q = \mathrm{round}\left(\frac{x}{s}\right) + z
            $$

            反量化：

            $$
            \hat{x} = s(q-z)
            $$

            量化的目标是：

            - 降低显存
            - 提高吞吐
            - 尽量少损失精度

            常见路线包括：

            - INT8
            - INT4
            - GPTQ
            - AWQ
            - FP8
            """
        ),
        md(
            r"""
            ## 3. PagedAttention、Continuous Batching、Speculative Decoding

            ### PagedAttention

            核心思想：像操作系统分页那样管理 KV Cache，减少碎片，提高显存利用率。

            ### Continuous Batching

            不必等整批请求全部完成再加入新请求，而是持续把新请求插入 scheduler，提高 GPU 利用率。

            ### Speculative Decoding

            用一个更小更快的 draft model 先提议若干 token，再由大模型并行验证，从而降低单 token 延迟。
            """
        ),
        code(common_style_code(include_torch=False)),
        code(
            """
            # 计算不同 KV head 设计下的 cache 显存。
            context_lengths = np.array([4_096, 8_192, 16_384, 32_768, 65_536, 131_072])
            num_layers = 32
            head_dim = 128
            bytes_per_value = 2  # fp16


            def kv_cache_gb(num_kv_heads):
                memory_bytes = 2 * num_layers * context_lengths * num_kv_heads * head_dim * bytes_per_value
                return memory_bytes / (1024 ** 3)


            kv_df = pd.DataFrame(
                {
                    "上下文长度": context_lengths,
                    "MQA (1 KV head)": kv_cache_gb(1),
                    "GQA (8 KV heads)": kv_cache_gb(8),
                    "MHA (32 KV heads)": kv_cache_gb(32),
                }
            )
            kv_df
            """
        ),
        code(
            """
            kv_long = kv_df.melt(id_vars="上下文长度", var_name="设计", value_name="KV Cache 显存(GB)")
            plt.figure(figsize=(11, 6))
            sns.lineplot(data=kv_long, x="上下文长度", y="KV Cache 显存(GB)", hue="设计", marker="o")
            plt.xscale("log", base=2)
            plt.title("不同 KV 设计下，KV Cache 显存随上下文长度增长")
            plt.show()
            """
        ),
        code(
            """
            # 量化误差实验：比较不同 bit-width 的重建误差。
            np.random.seed(42)
            weights = np.random.randn(20000).astype(np.float32)


            def uniform_quantize(x, bits):
                qmin = -(2 ** (bits - 1))
                qmax = 2 ** (bits - 1) - 1
                scale = np.max(np.abs(x)) / qmax
                q = np.clip(np.round(x / scale), qmin, qmax)
                x_hat = q * scale
                return x_hat


            quant_records = []
            for bits in [16, 8, 6, 4]:
                x_hat = uniform_quantize(weights, bits=bits)
                mse = np.mean((weights - x_hat) ** 2)
                quant_records.append({"bits": bits, "重建 MSE": mse})

            quant_df = pd.DataFrame(quant_records)
            quant_df
            """
        ),
        code(
            """
            plt.figure(figsize=(9, 5))
            sns.barplot(data=quant_df, x="bits", y="重建 MSE", palette="magma")
            plt.title("不同量化位宽的重建误差")
            plt.show()
            """
        ),
        code(
            """
            # Toy speculative decoding：不同 acceptance rate 下的理论 speedup。
            acceptance_rates = np.linspace(0.1, 0.95, 18)
            draft_lengths = [2, 4, 8]

            spec_records = []
            for k in draft_lengths:
                for acc in acceptance_rates:
                    # 一个简单近似：每轮平均接受 1 + (k-1)*acc 个 token
                    speedup = 1 + (k - 1) * acc
                    spec_records.append({"draft 长度": k, "接受率": acc, "理论加速比": speedup})

            spec_df = pd.DataFrame(spec_records)
            """
        ),
        code(
            """
            plt.figure(figsize=(11, 6))
            sns.lineplot(data=spec_df, x="接受率", y="理论加速比", hue="draft 长度", marker="o")
            plt.title("Speculative Decoding 的理论加速趋势")
            plt.xlabel("draft token acceptance rate")
            plt.ylabel("approx speedup")
            plt.show()
            """
        ),
        code(
            """
            # Continuous batching 的简化模拟：比较静态批处理与持续批处理的平均完成时间。
            arrivals = np.array([0, 1, 2, 4, 5, 7, 9, 10, 12, 14])
            service = np.array([6, 5, 7, 3, 4, 6, 5, 4, 3, 2])

            static_finish = []
            batch_size = 4
            current_time = 0
            for i in range(0, len(arrivals), batch_size):
                batch_arrival = arrivals[i : i + batch_size]
                batch_service = service[i : i + batch_size]
                start = max(current_time, batch_arrival.max())
                finish = start + batch_service.max()
                static_finish.extend([finish] * len(batch_arrival))
                current_time = finish

            continuous_finish = []
            current_time = 0
            for a, s in zip(arrivals, service):
                start = max(current_time, a)
                finish = start + s
                continuous_finish.append(finish)
                current_time = finish

            schedule_df = pd.DataFrame(
                {
                    "请求": np.arange(len(arrivals)),
                    "到达时间": arrivals,
                    "静态批处理完成时间": static_finish,
                    "持续调度完成时间": continuous_finish,
                }
            )

            schedule_df["静态响应时延"] = schedule_df["静态批处理完成时间"] - schedule_df["到达时间"]
            schedule_df["持续响应时延"] = schedule_df["持续调度完成时间"] - schedule_df["到达时间"]
            schedule_df
            """
        ),
        code(
            """
            latency_long = schedule_df.melt(
                id_vars="请求",
                value_vars=["静态响应时延", "持续响应时延"],
                var_name="策略",
                value_name="响应时延",
            )
            plt.figure(figsize=(11, 6))
            sns.barplot(data=latency_long, x="请求", y="响应时延", hue="策略")
            plt.title("静态批处理 vs 持续调度 的响应时延对比（玩具示意）")
            plt.show()
            """
        ),
        md(
            r"""
            ## 4. 实战建议

            - 长上下文首先检查 `KV Cache` 是否撑得住，而不是先纠结算力峰值
            - 多请求服务场景里，`Continuous Batching` 往往比单点 kernel 优化更显著
            - 想低成本部署，先考虑 `INT4 / INT8 + KV 优化 + Prefix Cache`
            - 如果目标是极低时延，优先考虑 `Speculative Decoding + 小 draft model`
            - 如果是高吞吐服务，`PagedAttention + 调度器 + 多 LoRA 复用` 会更关键
            """
        ),
        md(
            r"""
            ## 5. 主要资料

            - vLLM 官方文档:
              https://docs.vllm.ai/en/v0.10.0/
            - vLLM 官网:
              https://vllm.ai/
            - FlashAttention（NeurIPS 2022）:
              https://arxiv.org/abs/2205.14135
            - Fast Inference from Transformers via Speculative Decoding（2022-11-30）:
              https://arxiv.org/abs/2211.17192
            - Accelerating Large Language Model Decoding with Speculative Sampling（2023-02-02）:
              https://arxiv.org/abs/2302.01318
            """
        ),
    ]
    write_notebook("15_推理系统与部署优化体系.ipynb", cells)


def rl_mdp_notebook():
    cells = [
        md(
            r"""
            # 16. 强化学习基础与 MDP 体系
            强化学习研究的是**序列决策问题**：智能体并不是一次性输出答案，而是在环境中持续观察、行动、接收反馈，并通过多步交互最大化长期累计回报。与监督学习直接拟合标注不同，强化学习的难点在于：

            - 监督信号往往是**延迟到达**的，当前动作的质量要靠未来回报来判断。
            - 数据分布由策略本身决定，策略一变，采样到的状态分布也会改变。
            - 智能体必须同时处理**探索**与**利用**之间的张力。
            """
        ),
        *professional_cells(
            model_title="强化学习与马尔可夫决策过程",
            definition=r"""
            强化学习通常形式化为马尔可夫决策过程（Markov Decision Process, MDP）：
            $$
            \mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)
            $$
            其中 $\mathcal{S}$ 为状态空间，$\mathcal{A}$ 为动作空间，$P(s' \mid s, a)$ 为状态转移概率，$R(s,a)$ 或 $R(s,a,s')$ 为奖励函数，$\gamma \in [0,1)$ 为折扣因子。策略 $\pi(a\mid s)$ 定义智能体在状态 $s$ 下采取动作 $a$ 的概率分布。强化学习的核心目标是求解最优策略
            $$
            \pi^\star = \arg\max_\pi \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{T} \gamma^t r_t\right].
            $$
            """,
            io_spec=r"""
            强化学习的输入不再是独立同分布样本，而是交互序列
            $$
            \tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots).
            $$
            输出可以是：

            - 直接给出动作分布的策略 $\pi(a\mid s)$；
            - 估计状态价值 $V^\pi(s)$；
            - 估计动作价值 $Q^\pi(s,a)$；
            - 同时学习策略与价值函数。

            对于有限状态空间，策略和值函数可以用表格表示；对高维状态空间，则需要用神经网络进行函数逼近。
            """,
            architecture=r"""
            强化学习的结构由“策略 - 环境 - 反馈”闭环构成。一个标准决策周期包含四步：

            1. 智能体接收当前状态 $s_t$；
            2. 根据策略 $\pi(a\mid s_t)$ 选择动作 $a_t$；
            3. 环境依据 $P(s_{t+1}\mid s_t, a_t)$ 发生转移并返回奖励 $r_t$；
            4. 智能体用该反馈更新策略或价值估计。

            这种闭环意味着强化学习中的数据生成过程与模型参数紧密耦合。策略改变之后，状态访问分布、奖励统计和训练难度都会随之改变。
            """,
            objective=r"""
            强化学习的统一优化目标是最大化期望累计回报。围绕这一目标形成了三条基本路线：

            - 动态规划（DP）：假设环境模型已知，直接做 Bellman 备份；
            - 蒙特卡洛（MC）：用完整轨迹回报估计价值，无偏但方差大；
            - 时序差分（TD）：用一步 bootstrap 目标更新价值，有偏但方差更低。

            价值函数定义为
            $$
            V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty}\gamma^t r_t \mid s_0=s\right], \qquad
            Q^\pi(s,a) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty}\gamma^t r_t \mid s_0=s, a_0=a\right].
            $$
            它们满足 Bellman 方程，是后续 DQN、Actor-Critic、PPO、SAC 等方法的共同基础。
            """,
            complexity=r"""
            强化学习的计算难点不只是单次前向计算，而是样本复杂度、探索成本和目标非平稳性。即使模型很小，只要奖励稀疏、状态空间大或信用分配链条长，训练就会显著变难。

            从统计角度看，强化学习的方差来源包括：随机环境转移、随机策略、有限采样和 bootstrap 误差。从工程角度看，训练稳定性依赖于奖励尺度、折扣因子、探索策略、目标网络、经验回放、归一化和优势估计等细节。
            """,
            comparison=r"""
            与监督学习相比，强化学习不依赖固定标签，而是从交互中构造训练信号。与 bandit 问题相比，MDP 额外引入了状态转移和长期信用分配。与搜索算法相比，强化学习不是在单次推理时规划一棵树，而是在参数空间中学习可重复执行的策略。

            这一差异决定了强化学习既可以解释控制任务，也可以解释现代大语言模型后训练中的在线策略优化。
            """,
        ),
        architecture_diagram_cell("mdp"),
        architecture_diagram_cell("bellman"),
        md(
            r"""
            ## 核心公式与概念

            1. 折扣回报
               $$
               G_t = \sum_{k=0}^{\infty}\gamma^k r_{t+k}
               $$

            2. Bellman 期望方程
               $$
               V^\pi(s) = \sum_a \pi(a\mid s)\sum_{s',r} P(s',r\mid s,a)\left[r + \gamma V^\pi(s')\right]
               $$

            3. Bellman 最优方程
               $$
               V^\star(s) = \max_a \sum_{s',r} P(s',r\mid s,a)\left[r + \gamma V^\star(s')\right]
               $$
               $$
               Q^\star(s,a) = \sum_{s',r} P(s',r\mid s,a)\left[r + \gamma \max_{a'} Q^\star(s',a')\right]
               $$

            4. 优势函数
               $$
               A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)
               $$
               它描述“在给定状态下，该动作相对平均水平到底好多少”，是现代策略梯度算法中最常见的学习信号。
            """
        ),
        md(
            r"""
            ## 方法谱系

            从知识结构上，可以把强化学习基础分成三层：

            - **问题设置层**：MDP、回报、策略、价值函数、Bellman 方程；
            - **估计层**：DP、MC、TD 分别对应模型已知、完整采样和一步 bootstrap；
            - **控制层**：在学会评估后，通过贪心改进、策略梯度或策略迭代获得更优策略。

            DQN 是把 TD 控制推广到深度函数逼近；Actor-Critic 是把价值估计与策略优化拼接成统一框架；PPO、SAC、GRPO 等方法都是在更复杂场景下继续改进“如何构造稳定的更新信号”。
            """
        ),
        code(common_style_code()),
        code(
            """
            states = ["S0", "S1", "S2", "Terminal"]
            gamma = 0.9
            policy = {
                0: {"stay": 0.25, "right": 0.75},
                1: {"stay": 0.15, "right": 0.85},
                2: {"stay": 0.10, "right": 0.90},
            }
            transitions = {
                0: {"stay": [(1.0, 0, 0.0)], "right": [(1.0, 1, 0.0)]},
                1: {"stay": [(1.0, 1, 0.0)], "right": [(1.0, 2, 0.0)]},
                2: {"stay": [(1.0, 2, 0.2)], "right": [(1.0, 3, 1.0)]},
            }

            def policy_evaluation(policy, transitions, gamma=0.9, iterations=20):
                V = np.zeros(4)
                history = [V.copy()]
                for _ in range(iterations):
                    new_V = V.copy()
                    for s in [0, 1, 2]:
                        total = 0.0
                        for action, action_prob in policy[s].items():
                            for prob, next_state, reward in transitions[s][action]:
                                total += action_prob * prob * (reward + gamma * V[next_state])
                        new_V[s] = total
                    V = new_V
                    history.append(V.copy())
                return np.array(history)

            value_history = policy_evaluation(policy, transitions, gamma=gamma, iterations=25)
            value_df = pd.DataFrame(value_history[:, :3], columns=["V(S0)", "V(S1)", "V(S2)"])
            value_df.head()
            """
        ),
        code(
            """
            value_long = value_df.reset_index().melt(id_vars="index", var_name="state", value_name="value")
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=value_long, x="index", y="value", hue="state", marker="o")
            plt.title("Iterative Policy Evaluation Convergence")
            plt.xlabel("iteration")
            plt.ylabel("state value")
            plt.show()
            """
        ),
        code(
            """
            true_values = np.arange(1, 6) / 6.0

            def generate_random_walk_episode():
                state = 2
                episode = []
                while True:
                    action = np.random.choice([-1, 1])
                    next_state = state + action
                    if next_state == -1:
                        reward = 0.0
                        episode.append((state, reward, None))
                        return episode
                    if next_state == 5:
                        reward = 1.0
                        episode.append((state, reward, None))
                        return episode
                    reward = 0.0
                    episode.append((state, reward, next_state))
                    state = next_state

            def mc_prediction(alpha=0.05, episodes=200):
                values = np.full(5, 0.5)
                errors = []
                for _ in range(episodes):
                    episode = generate_random_walk_episode()
                    G = 0.0
                    visited = set()
                    for state, reward, _ in reversed(episode):
                        G = reward + G
                        if state not in visited:
                            values[state] += alpha * (G - values[state])
                            visited.add(state)
                    errors.append(np.sqrt(np.mean((values - true_values) ** 2)))
                return np.array(errors)

            def td_prediction(alpha=0.1, episodes=200):
                values = np.full(5, 0.5)
                errors = []
                for _ in range(episodes):
                    episode = generate_random_walk_episode()
                    for state, reward, next_state in episode:
                        target = reward if next_state is None else reward + values[next_state]
                        values[state] += alpha * (target - values[state])
                    errors.append(np.sqrt(np.mean((values - true_values) ** 2)))
                return np.array(errors)

            mc_errors = mc_prediction()
            td_errors = td_prediction()
            error_df = pd.DataFrame(
                {
                    "episode": np.arange(1, len(mc_errors) + 1),
                    "Monte Carlo": mc_errors,
                    "TD(0)": td_errors,
                }
            )
            error_df.head()
            """
        ),
        code(
            """
            error_long = error_df.melt(id_vars="episode", var_name="method", value_name="rmse")
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=error_long, x="episode", y="rmse", hue="method")
            plt.title("Monte Carlo vs TD(0) on a Random Walk")
            plt.ylabel("RMSE against true value")
            plt.show()
            """
        ),
        md(
            r"""
            ## 建模与工程建议

            - 如果环境模型已知，先从 DP 出发，因为它最能帮助理解 Bellman 结构。
            - 如果采样代价昂贵，要优先考虑 TD 和 bootstrap，因为完整蒙特卡洛轨迹的方差通常过大。
            - 强化学习实验需要显式报告随机种子、平均回报曲线与方差，不应只给出单次最优结果。
            - 在进入深度强化学习之前，必须先把价值函数、优势函数、on-policy / off-policy、exploration / exploitation 这些概念彻底打通，否则后续算法只会变成公式堆叠。
            """
        ),
        md(
            r"""
            ## 主要资料

            - Sutton, Barto. *Reinforcement Learning: An Introduction*（第二版）
            - David Silver 的 RL 课程讲义
            - Bellman 方程是整个强化学习理论的共同语言，后续所有 notebook 都会回到这里
            """
        ),
    ]
    write_notebook("16_强化学习基础与MDP体系.ipynb", cells)


def value_dqn_notebook():
    cells = [
        md(
            r"""
            # 17. 值函数方法与 DQN 体系
            值函数方法的核心思想是：与其直接学习“在每个状态下该怎么做”，不如先学习“每个状态或状态-动作对究竟值多少钱”。一旦价值估计足够准确，策略就可以通过贪心改进自动出现。

            这条路线从表格型的 SARSA、Q-learning 出发，最终发展到深度 Q 网络（DQN）及其一系列稳定化变体，是离散动作空间强化学习最经典的方法谱系。
            """
        ),
        *professional_cells(
            model_title="值函数控制与 DQN",
            definition=r"""
            值函数方法试图估计最优动作价值函数
            $$
            Q^\star(s,a) = \mathbb{E}\left[r_t + \gamma \max_{a'} Q^\star(s_{t+1}, a') \mid s_t=s, a_t=a\right].
            $$
            一旦学到 $Q^\star$，就可以通过
            $$
            \pi^\star(s)=\arg\max_a Q^\star(s,a)
            $$
            导出最优策略。DQN 的创新在于用神经网络 $Q_\theta(s,a)$ 取代表格，从而把 TD 控制推广到高维感知空间。
            """,
            io_spec=r"""
            输入可以是离散状态、连续特征、图像帧或其他高维表征；输出是每个离散动作的价值估计。对离散动作空间，网络通常把同一状态下所有动作的 $Q$ 值一起输出：
            $$
            Q_\theta(s) = [Q_\theta(s,a_1), \dots, Q_\theta(s,a_{|\mathcal{A}|})].
            $$
            这使得动作选择只需做一次前向传播和一次 $\arg\max$。
            """,
            architecture=r"""
            DQN 的工程结构由四个关键部件组成：

            1. 在线网络 $Q_\theta$：负责当前估计和动作选择；
            2. 目标网络 $Q_{\bar{\theta}}$：生成较稳定的 bootstrap 目标；
            3. 经验回放（Replay Buffer）：打破样本时间相关性；
            4. ε-greedy 探索：在贪心利用之外保留随机探索。

            Double DQN 进一步分离“选动作”和“评估动作”，缓解最大化带来的过估计；Dueling DQN 把状态价值和优势分开建模；PER 则优先抽取 TD 误差大的样本。
            """,
            objective=r"""
            DQN 通过最小化 TD 残差训练网络：
            $$
            \mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s')\sim \mathcal{D}}
            \left[\left(Q_\theta(s,a) - y\right)^2\right],
            $$
            其中
            $$
            y = r + \gamma \max_{a'} Q_{\bar{\theta}}(s', a').
            $$
            Double DQN 把目标改写为
            $$
            y_{\text{double}} = r + \gamma Q_{\bar{\theta}}\left(s', \arg\max_{a'} Q_\theta(s', a')\right),
            $$
            以降低正偏差。
            """,
            complexity=r"""
            表格型值函数方法的更新开销很低，但只能处理小规模状态空间。DQN 把表示能力扩大到神经网络级别，却引入了目标漂移、样本相关性、过估计和训练不稳定等问题，因此必须依赖回放缓冲区、目标网络、梯度裁剪和奖励尺度控制。

            从样本效率角度看，值函数方法通常优于纯 on-policy 策略梯度；但一旦动作空间连续化，基于 $\arg\max$ 的控制就不再方便，需要转向 Actor-Critic。
            """,
            comparison=r"""
            SARSA 是 on-policy 控制，它学习“当前行为策略真正会得到的价值”；Q-learning 是 off-policy 控制，它直接逼近最优贪心策略。DQN 继承了 Q-learning 的 off-policy 属性，并用深度网络扩展表示能力。Double / Dueling / PER 都是在 DQN 主框架上对偏差、方差和表示分解做工程修正。
            """,
        ),
        architecture_diagram_cell("dqn"),
        architecture_diagram_cell("bellman"),
        md(
            r"""
            ## 表格方法到深度方法的过渡

            - **SARSA** 更新目标是当前策略实际选出的下一个动作：
              $$
              Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\left[r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t,a_t)\right]
              $$
              因此更新更保守，也更能反映当前探索策略的真实风险。

            - **Q-learning** 更新目标使用下一状态的最大动作价值：
              $$
              Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\left[r_t + \gamma \max_a Q(s_{t+1}, a) - Q(s_t,a_t)\right]
              $$
              因而更偏向直接逼近最优策略。

            - **DQN** 用神经网络近似 $Q$ 函数，把表格更新推广到高维观测空间，但必须用目标网络和回放缓冲区才能避免训练发散。
            """
        ),
        code(common_style_code(include_torch=True)),
        code(
            """
            def run_cliff_control(method="q_learning", episodes=250, alpha=0.5, gamma=0.95, epsilon=0.1):
                rows, cols = 4, 6
                start = (3, 0)
                goal = (3, 5)
                cliff = {(3, c) for c in range(1, 5)}
                actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                q = np.zeros((rows, cols, len(actions)))
                returns = []

                def step(state, action_idx):
                    dr, dc = actions[action_idx]
                    nr = int(np.clip(state[0] + dr, 0, rows - 1))
                    nc = int(np.clip(state[1] + dc, 0, cols - 1))
                    next_state = (nr, nc)
                    reward = -1.0
                    done = False
                    if next_state in cliff:
                        reward = -100.0
                        next_state = start
                    elif next_state == goal:
                        done = True
                    return next_state, reward, done

                for _ in range(episodes):
                    state = start
                    action = np.random.randint(len(actions)) if np.random.rand() < epsilon else np.argmax(q[state])
                    episode_return = 0.0
                    for _ in range(200):
                        next_state, reward, done = step(state, action)
                        episode_return += reward
                        if method == "sarsa":
                            next_action = np.random.randint(len(actions)) if np.random.rand() < epsilon else np.argmax(q[next_state])
                            target = reward if done else reward + gamma * q[next_state][next_action]
                            q[state][action] += alpha * (target - q[state][action])
                            state, action = next_state, next_action
                        else:
                            target = reward if done else reward + gamma * np.max(q[next_state])
                            q[state][action] += alpha * (target - q[state][action])
                            state = next_state
                            action = np.random.randint(len(actions)) if np.random.rand() < epsilon else np.argmax(q[state])
                        if done:
                            break
                    returns.append(episode_return)
                return q, np.array(returns)

            q_sarsa, sarsa_returns = run_cliff_control(method="sarsa")
            q_qlearning, qlearning_returns = run_cliff_control(method="q_learning")
            cliff_df = pd.DataFrame(
                {
                    "episode": np.arange(1, len(sarsa_returns) + 1),
                    "SARSA": sarsa_returns,
                    "Q-learning": qlearning_returns,
                }
            )
            cliff_df.head()
            """
        ),
        code(
            """
            cliff_long = cliff_df.melt(id_vars="episode", var_name="method", value_name="return")
            cliff_long["smoothed_return"] = cliff_long.groupby("method")["return"].transform(
                lambda x: pd.Series(x).rolling(15, min_periods=1).mean()
            )
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=cliff_long, x="episode", y="smoothed_return", hue="method")
            plt.title("SARSA vs Q-learning on Cliff Walking")
            plt.ylabel("moving average return")
            plt.show()
            """
        ),
        code(
            """
            from collections import deque

            class LineWorldEnv:
                def __init__(self, n_states=7, max_steps=14):
                    self.n_states = n_states
                    self.max_steps = max_steps
                    self.reset()

                def _obs(self):
                    obs = np.zeros(self.n_states, dtype=np.float32)
                    obs[self.pos] = 1.0
                    return obs

                def reset(self):
                    self.pos = np.random.randint(0, self.n_states - 2)
                    self.steps = 0
                    return self._obs()

                def step(self, action):
                    self.steps += 1
                    self.pos = int(np.clip(self.pos + (1 if action == 1 else -1), 0, self.n_states - 1))
                    reward = 1.0 if self.pos == self.n_states - 1 else -0.03
                    done = self.pos == self.n_states - 1 or self.steps >= self.max_steps
                    return self._obs(), reward, done

            class QNet(nn.Module):
                def __init__(self, input_dim, output_dim):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(input_dim, 32),
                        nn.ReLU(),
                        nn.Linear(32, 32),
                        nn.ReLU(),
                        nn.Linear(32, output_dim),
                    )

                def forward(self, x):
                    return self.net(x)

            env = LineWorldEnv()
            q_net = QNet(env.n_states, 2)
            target_net = QNet(env.n_states, 2)
            target_net.load_state_dict(q_net.state_dict())
            optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
            replay = deque(maxlen=3000)
            gamma = 0.95
            epsilon = 1.0
            batch_size = 64
            dqn_returns = []

            for episode in range(220):
                state = env.reset()
                done = False
                episode_return = 0.0
                while not done:
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    if np.random.rand() < epsilon:
                        action = np.random.randint(2)
                    else:
                        with torch.no_grad():
                            action = int(q_net(state_tensor).argmax(dim=1).item())

                    next_state, reward, done = env.step(action)
                    replay.append((state, action, reward, next_state, float(done)))
                    state = next_state
                    episode_return += reward

                    if len(replay) >= batch_size:
                        batch = random.sample(replay, batch_size)
                        states_b = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
                        actions_b = torch.tensor([b[1] for b in batch], dtype=torch.long).unsqueeze(1)
                        rewards_b = torch.tensor([b[2] for b in batch], dtype=torch.float32)
                        next_states_b = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32)
                        dones_b = torch.tensor([b[4] for b in batch], dtype=torch.float32)

                        q_values = q_net(states_b).gather(1, actions_b).squeeze(1)
                        with torch.no_grad():
                            next_q = target_net(next_states_b).max(dim=1).values
                            targets = rewards_b + gamma * next_q * (1.0 - dones_b)

                        loss = nn.functional.mse_loss(q_values, targets)
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
                        optimizer.step()

                epsilon = max(0.05, epsilon * 0.985)
                if (episode + 1) % 15 == 0:
                    target_net.load_state_dict(q_net.state_dict())
                dqn_returns.append(episode_return)

            dqn_df = pd.DataFrame(
                {
                    "episode": np.arange(1, len(dqn_returns) + 1),
                    "return": dqn_returns,
                    "moving_avg": pd.Series(dqn_returns).rolling(15, min_periods=1).mean(),
                }
            )
            dqn_df.tail()
            """
        ),
        code(
            """
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=dqn_df, x="episode", y="moving_avg")
            plt.title("DQN Learning Curve on a Tiny LineWorld")
            plt.ylabel("moving average return")
            plt.show()

            with torch.no_grad():
                state_grid = torch.eye(env.n_states)
                q_grid = q_net(state_grid).numpy()
            q_value_df = pd.DataFrame(q_grid, columns=["go_left", "go_right"])
            q_value_df["state"] = np.arange(env.n_states)
            q_value_df
            """
        ),
        code(
            """
            q_long = q_value_df.melt(id_vars="state", var_name="action", value_name="q_value")
            plt.figure(figsize=(10, 5))
            sns.barplot(data=q_long, x="state", y="q_value", hue="action")
            plt.title("Q-values Learned by DQN")
            plt.show()
            """
        ),
        code(
            """
            true_q = np.array([1.0, 0.8, 0.6])
            single_estimates = []
            double_estimates = []
            for _ in range(10000):
                q1 = true_q + np.random.normal(0, 0.5, size=3)
                q2 = true_q + np.random.normal(0, 0.5, size=3)
                single_estimates.append(np.max(q1))
                double_estimates.append(q2[np.argmax(q1)])

            bias_df = pd.DataFrame(
                {
                    "estimator": ["single max", "double estimate", "true max"],
                    "value": [np.mean(single_estimates), np.mean(double_estimates), np.max(true_q)],
                }
            )

            td_errors = np.linspace(0.05, 2.0, 12)
            priorities = (td_errors + 1e-3) ** 0.6
            priorities = priorities / priorities.sum()
            per_df = pd.DataFrame({"td_error": td_errors, "sampling_prob": priorities})
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
            sns.barplot(data=bias_df, x="estimator", y="value", palette="viridis", ax=axes[0])
            axes[0].set_title("Why Double DQN Reduces Overestimation")
            sns.barplot(data=per_df, x="td_error", y="sampling_prob", color="#F58518", ax=axes[1])
            axes[1].set_title("Prioritized Replay Sampling Probability")
            axes[1].tick_params(axis="x", rotation=45)
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            r"""
            ## 使用建议与失效模式

            - 如果状态空间较小，优先先用表格型 SARSA / Q-learning 把更新逻辑理解清楚，再进入 DQN。
            - DQN 适合**离散动作空间**。一旦动作连续化，$\max_a Q(s,a)$ 的求解会变得困难，应转向 DDPG、TD3、SAC 等 Actor-Critic 方法。
            - 训练发散通常不是网络“太弱”，而是 TD 目标漂移过快、奖励尺度失衡、探索不足或回放缓冲区分布质量太差。
            - Double DQN、Dueling DQN、PER 不应被理解成彼此竞争的独立算法，而应理解为对 DQN 主体框架不同脆弱点的修补。
            """
        ),
        md(
            r"""
            ## 主要资料

            - DQN（Nature 2015）: https://www.nature.com/articles/nature14236
            - Prioritized Experience Replay（2015）: https://arxiv.org/abs/1511.05952
            - Dueling Network Architectures（2016）: https://arxiv.org/abs/1511.06581
            """
        ),
    ]
    write_notebook("17_值函数方法与DQN体系.ipynb", cells)


def policy_gradient_notebook():
    cells = [
        md(
            r"""
            # 18. 策略梯度与 Actor-Critic 体系
            当动作空间较大、策略需要显式随机性、或值函数方法难以稳定做 $\arg\max$ 时，直接在策略空间中优化就成为更自然的选择。策略梯度方法不再先学“动作值表”，而是直接沿着“让高回报动作概率变大”的方向更新参数。

            这条路线从 REINFORCE 出发，引入 baseline 降方差，进一步发展为 Actor-Critic、A2C/A3C、TRPO 与 PPO，构成现代策略优化的主干。
            """
        ),
        *professional_cells(
            model_title="策略梯度与 Actor-Critic",
            definition=r"""
            策略梯度方法直接参数化策略 $\pi_\theta(a\mid s)$，目标是最大化
            $$
            J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T}\gamma^t r_t\right].
            $$
            策略梯度定理给出
            $$
            \nabla_\theta J(\theta) =
            \mathbb{E}_{\pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t\mid s_t) Q^{\pi_\theta}(s_t,a_t)\right].
            $$
            它说明策略更新并不需要对环境转移概率求导，只需要对策略的对数概率求导并乘上回报或优势信号。
            """,
            io_spec=r"""
            输入是状态 $s$，输出是动作分布参数：

            - 离散动作场景中，输出通常是 logits，再通过 softmax 得到 $\pi_\theta(a\mid s)$；
            - 连续动作场景中，输出通常是高斯分布的均值与方差。

            如果再加入价值函数估计器，就得到 Actor-Critic 结构：Actor 输出策略，Critic 输出 $V(s)$ 或 $Q(s,a)$，用于降低梯度估计方差并提供更稳定的信用分配信号。
            """,
            architecture=r"""
            REINFORCE 的结构最简单：采样一整条轨迹，计算每一步的回报 $G_t$，然后让提高回报的动作概率上升。它的优点是无偏，缺点是方差大、学习慢。

            Actor-Critic 的关键思想是把“谁负责决策”和“谁负责评价”拆开：

            - Actor：输出动作分布并执行动作；
            - Critic：估计状态价值或动作价值；
            - Advantage / TD error：把 Critic 的估计转成 Actor 的学习信号。

            TRPO 和 PPO 则进一步在策略更新时显式限制新旧策略之间的偏移，避免一步更新过大导致性能崩塌。
            """,
            objective=r"""
            常见目标函数包括：

            - REINFORCE：
              $$
              \mathcal{L}_{\text{pg}} = - \mathbb{E}\left[\log \pi_\theta(a_t\mid s_t) G_t\right]
              $$
            - 加 baseline 后：
              $$
              \mathcal{L}_{\text{pg}} = - \mathbb{E}\left[\log \pi_\theta(a_t\mid s_t)\left(G_t - b(s_t)\right)\right]
              $$
            - Actor-Critic 常用 TD 优势：
              $$
              \hat{A}_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
              $$
            - PPO 的 clipped surrogate：
              $$
              \mathcal{L}_{\text{PPO}} =
              \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t, \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]
              $$
              其中 $r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}$。
            """,
            complexity=r"""
            策略梯度方法的主要代价来自高方差与 on-policy 采样需求。相比离策略值函数方法，它们通常样本效率更低，但在处理随机策略、连续动作和受约束更新时更加自然。

            Actor-Critic 的稳定性高度依赖优势估计、价值函数偏差、熵正则、学习率和 mini-batch 重复使用策略。PPO 之所以广泛使用，并不是因为它在理论上最强，而是因为它在“稳定性 - 实现复杂度 - 性能”之间取得了较好的折中。
            """,
            comparison=r"""
            REINFORCE 是最基础的无偏策略梯度；Actor-Critic 通过 Critic 降低方差；A2C/A3C 主要是并行化与同步/异步训练框架；TRPO 用近似信赖域保证更新保守；PPO 用 clipping 近似 TRPO 的约束思想，以更简单的实现获得较好的稳定性，因此成为工业与研究中最常见的策略优化基线。
            """,
        ),
        architecture_diagram_cell("actor_critic"),
        architecture_diagram_cell("ppo"),
        md(
            r"""
            ## 优势函数、GAE 与稳定更新

            在现代策略优化中，优势函数是连接 Critic 与 Actor 的关键桥梁。它回答的问题不是“这个动作绝对值多少钱”，而是“这个动作比当前状态的平均动作好多少”。如果只用回报 $G_t$ 训练策略，方差很大；如果用价值函数差分构造优势，方差会显著降低。

            广义优势估计（GAE）进一步把多步 TD 误差按 $\lambda$ 衰减累加：
            $$
            \hat{A}^{\text{GAE}(\gamma,\lambda)}_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l},
            \qquad
            \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t).
            $$
            其中 $\lambda$ 控制偏差与方差之间的折中：$\lambda$ 越大，越接近 Monte Carlo；$\lambda$ 越小，越接近一步 TD。
            """
        ),
        code(common_style_code(include_torch=True)),
        code(
            """
            class CorridorEnv:
                def __init__(self, n_states=7, max_steps=15):
                    self.n_states = n_states
                    self.max_steps = max_steps
                    self.reset()

                def _obs(self):
                    obs = np.zeros(self.n_states, dtype=np.float32)
                    obs[self.pos] = 1.0
                    return obs

                def reset(self):
                    self.pos = 0
                    self.steps = 0
                    return self._obs()

                def step(self, action):
                    self.steps += 1
                    self.pos = int(np.clip(self.pos + (1 if action == 1 else -1), 0, self.n_states - 1))
                    reward = 1.0 if self.pos == self.n_states - 1 else -0.02
                    done = self.pos == self.n_states - 1 or self.steps >= self.max_steps
                    return self._obs(), reward, done

            class PolicyNet(nn.Module):
                def __init__(self, input_dim, action_dim):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(input_dim, 32),
                        nn.Tanh(),
                        nn.Linear(32, action_dim),
                    )

                def forward(self, x):
                    return self.net(x)

            class ValueNet(nn.Module):
                def __init__(self, input_dim):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(input_dim, 32),
                        nn.Tanh(),
                        nn.Linear(32, 1),
                    )

                def forward(self, x):
                    return self.net(x).squeeze(-1)

            def train_reinforce(episodes=260, gamma=0.98):
                env = CorridorEnv()
                policy = PolicyNet(env.n_states, 2)
                optimizer = torch.optim.Adam(policy.parameters(), lr=2e-3)
                returns = []

                for _ in range(episodes):
                    state = env.reset()
                    log_probs, rewards = [], []
                    done = False
                    while not done:
                        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                        logits = policy(state_tensor)
                        dist = torch.distributions.Categorical(logits=logits)
                        action = dist.sample()
                        next_state, reward, done = env.step(int(action.item()))
                        log_probs.append(dist.log_prob(action))
                        rewards.append(reward)
                        state = next_state

                    discounted_returns = []
                    G = 0.0
                    for reward in reversed(rewards):
                        G = reward + gamma * G
                        discounted_returns.insert(0, G)
                    returns_tensor = torch.tensor(discounted_returns, dtype=torch.float32)
                    returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-6)

                    loss = 0.0
                    for log_prob, ret in zip(log_probs, returns_tensor):
                        loss = loss - log_prob * ret

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    returns.append(sum(rewards))
                return np.array(returns)

            def train_actor_critic(episodes=260, gamma=0.98):
                env = CorridorEnv()
                policy = PolicyNet(env.n_states, 2)
                value = ValueNet(env.n_states)
                policy_opt = torch.optim.Adam(policy.parameters(), lr=2e-3)
                value_opt = torch.optim.Adam(value.parameters(), lr=2e-3)
                returns = []

                for _ in range(episodes):
                    state = env.reset()
                    episode_return = 0.0
                    done = False
                    while not done:
                        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                        logits = policy(state_tensor)
                        dist = torch.distributions.Categorical(logits=logits)
                        action = dist.sample()
                        next_state, reward, done = env.step(int(action.item()))
                        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

                        value_s = value(state_tensor)
                        with torch.no_grad():
                            value_next = value(next_state_tensor) if not done else torch.tensor([0.0])
                            td_target = torch.tensor([reward], dtype=torch.float32) + gamma * value_next
                            advantage = td_target - value_s.detach()

                        actor_loss = -(dist.log_prob(action) * advantage).mean()
                        critic_loss = nn.functional.mse_loss(value_s, td_target)

                        policy_opt.zero_grad()
                        actor_loss.backward()
                        policy_opt.step()

                        value_opt.zero_grad()
                        critic_loss.backward()
                        value_opt.step()

                        state = next_state
                        episode_return += reward
                    returns.append(episode_return)
                return np.array(returns)

            reinforce_returns = train_reinforce()
            actor_critic_returns = train_actor_critic()
            pg_df = pd.DataFrame(
                {
                    "episode": np.arange(1, len(reinforce_returns) + 1),
                    "REINFORCE": reinforce_returns,
                    "Actor-Critic": actor_critic_returns,
                }
            )
            pg_df.head()
            """
        ),
        code(
            """
            pg_long = pg_df.melt(id_vars="episode", var_name="method", value_name="return")
            pg_long["moving_avg"] = pg_long.groupby("method")["return"].transform(
                lambda x: pd.Series(x).rolling(15, min_periods=1).mean()
            )
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=pg_long, x="episode", y="moving_avg", hue="method")
            plt.title("REINFORCE vs Actor-Critic")
            plt.ylabel("moving average return")
            plt.show()
            """
        ),
        code(
            """
            td_deltas = np.array([0.9, 0.4, -0.1, 0.7, -0.2])

            def gae_from_deltas(deltas, gamma=0.99, lam=0.95):
                advantages = np.zeros_like(deltas, dtype=float)
                running = 0.0
                for t in reversed(range(len(deltas))):
                    running = deltas[t] + gamma * lam * running
                    advantages[t] = running
                return advantages

            gae_records = []
            for lam in [0.0, 0.5, 0.95, 1.0]:
                advantages = gae_from_deltas(td_deltas, gamma=0.99, lam=lam)
                for step, advantage in enumerate(advantages):
                    gae_records.append({"step": step, "lambda": lam, "advantage": advantage})
            gae_df = pd.DataFrame(gae_records)
            """
        ),
        code(
            """
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=gae_df, x="step", y="advantage", hue="lambda", marker="o")
            plt.title("Effect of lambda in Generalized Advantage Estimation")
            plt.show()

            ratios = np.linspace(0.0, 2.0, 200)
            clip_eps = 0.2
            ppo_df = pd.DataFrame({"ratio": ratios})
            for advantage in [1.0, -1.0]:
                unclipped = ratios * advantage
                clipped = np.clip(ratios, 1 - clip_eps, 1 + clip_eps) * advantage
                ppo_df[f"adv_{advantage:+.0f}"] = np.minimum(unclipped, clipped)
            ppo_df.head()
            """
        ),
        code(
            """
            plt.figure(figsize=(10, 5))
            plt.plot(ppo_df["ratio"], ppo_df["adv_+1"], label="advantage > 0")
            plt.plot(ppo_df["ratio"], ppo_df["adv_-1"], label="advantage < 0")
            plt.axvline(1 - clip_eps, color="gray", linestyle="--")
            plt.axvline(1 + clip_eps, color="gray", linestyle="--")
            plt.title("Clipped PPO Surrogate as a Function of Policy Ratio")
            plt.xlabel("probability ratio r")
            plt.ylabel("surrogate objective term")
            plt.legend()
            plt.show()
            """
        ),
        md(
            r"""
            ## 方法比较

            - **REINFORCE**：概念最纯粹，但方差最高，通常只作为入门与理论基线。
            - **Actor-Critic**：通过 Critic 提供低方差学习信号，是现代 RL 的基础骨架。
            - **TRPO**：强调近似信赖域约束，理论上更严格，但实现复杂。
            - **PPO**：用 clipping 或 KL penalty 在近似保守更新的同时保持实现简单，因此成为最主流的在线策略优化基线之一。
            - **A2C / A3C**：重点在并行采样和训练框架，而不是改变目标函数本身。
            """
        ),
        md(
            r"""
            ## 主要资料

            - A3C（2016）: https://arxiv.org/abs/1602.01783
            - TRPO（2015）: https://arxiv.org/abs/1502.05477
            - PPO（2017）: https://arxiv.org/abs/1707.06347
            """
        ),
    ]
    write_notebook("18_策略梯度与Actor_Critic体系.ipynb", cells)


def continuous_control_notebook():
    cells = [
        md(
            r"""
            # 19. 连续控制与现代深度强化学习体系
            在连续动作空间中，值函数方法面临一个直接困难：动作不再是有限集合，$\max_a Q(s,a)$ 不能像离散动作那样直接枚举。于是现代连续控制算法大多转向 Actor-Critic 框架，让 Actor 直接输出连续动作，让 Critic 负责评价。

            DDPG、TD3 与 SAC 是这条路线最核心的三种代表。它们既共享离策略、经验回放与目标网络的工程基座，又在过估计控制、探索机制与熵正则化上形成了清晰分工。
            """
        ),
        *professional_cells(
            model_title="连续控制中的离策略 Actor-Critic",
            definition=r"""
            连续控制问题通常仍定义在 MDP 上，但动作空间变为连续域 $a \in \mathbb{R}^d$。此时确定性策略梯度（Deterministic Policy Gradient, DPG）给出
            $$
            \nabla_\theta J(\theta) =
            \mathbb{E}_{s\sim\mathcal{D}}\left[
                \nabla_a Q_\phi(s,a)\vert_{a=\mu_\theta(s)} \nabla_\theta \mu_\theta(s)
            \right].
            $$
            DDPG 用神经网络实现这一思想；TD3 通过双 Critic 与目标平滑修正 DDPG 的偏差问题；SAC 则把最大熵目标引入 Actor-Critic，使策略在优化回报的同时保持充分随机性。
            """,
            io_spec=r"""
            输入是连续状态特征，Actor 输出连续动作或动作分布参数，Critic 输出标量价值估计：

            - DDPG / TD3：Actor 常输出确定性动作 $\mu_\theta(s)$；
            - SAC：Actor 输出随机策略 $\pi_\theta(a\mid s)$，通常为高斯分布再经 squashing。

            在工程实现中，Replay Buffer、目标网络、Polyak averaging 与动作范围裁剪是必不可少的基础组件。
            """,
            architecture=r"""
            DDPG 的信息流为：用 Actor 产生动作，用 Critic 评估动作，再沿 Critic 对动作的梯度方向更新 Actor。TD3 进一步增加两个关键修正：

            1. **Clipped Double Q**：两个 Critic 取最小值，减轻过估计；
            2. **Target Policy Smoothing**：给目标动作加小噪声，降低 Critic 对尖锐误差的过拟合；
            3. **Delayed Policy Update**：Critic 多更新几步后再更新 Actor。

            SAC 则把目标改成最大熵 RL：
            $$
            J(\pi)=\sum_t \mathbb{E}[r(s_t,a_t)+\alpha \mathcal{H}(\pi(\cdot\mid s_t))].
            $$
            因而策略不仅追求高回报，也追求合适的随机性和覆盖度。
            """,
            objective=r"""
            三类方法的目标可以概括为：

            - **DDPG / TD3 Critic**：
              $$
              \mathcal{L}_Q = \mathbb{E}\left[\left(Q_\phi(s,a) - y\right)^2\right]
              $$
              $$
              y = r + \gamma Q_{\bar{\phi}}(s', \mu_{\bar{\theta}}(s'))
              $$

            - **DDPG / TD3 Actor**：
              $$
              \mathcal{L}_{\text{actor}} = -\mathbb{E}_{s\sim \mathcal{D}}[Q_\phi(s,\mu_\theta(s))]
              $$

            - **SAC Actor**：
              $$
              \mathcal{L}_{\pi} = \mathbb{E}_{s\sim\mathcal{D}, a\sim\pi_\theta}
              \left[\alpha \log \pi_\theta(a\mid s) - \min_j Q_{\phi_j}(s,a)\right]
              $$

            熵系数 $\alpha$ 控制探索与利用的权衡，是 SAC 的关键超参数。
            """,
            complexity=r"""
            连续控制算法的核心难点不是网络规模，而是 Critic 误差被 Actor 放大后形成的反馈环。只要 Critic 有系统性偏差，Actor 就会主动把策略推向被误估的动作区域。这正是 TD3 和 SAC 相比 DDPG 更稳定的根本原因。

            从样本效率看，这类算法通常优于 on-policy 的 PPO；但对奖励尺度、目标网络更新率、探索噪声和归一化策略非常敏感。
            """,
            comparison=r"""
            DDPG 适合作为连续控制的最小离策略基线；TD3 适合在确定性策略框架下追求更高稳定性；SAC 在实践中通常最稳健，尤其适合奖励噪声大、探索困难或多峰动作分布的场景，但其实现和调参复杂度也更高。
            """,
        ),
        architecture_diagram_cell("actor_critic"),
        md(
            r"""
            ## 何时使用 DDPG、TD3、SAC

            - 若任务结构简单、需要快速构建最小可运行基线，可先从 DDPG 出发。
            - 若 DDPG 出现明显过估计、策略抖动或训练发散，TD3 往往是第一选择。
            - 若需要更稳健的探索、更强的多样性保持或更平滑的训练过程，SAC 通常是更可靠的主力方案。

            在机器人控制、连续参数调节和高维动作优化场景中，SAC 已成为最常见的通用基线之一。
            """
        ),
        code(common_style_code(include_torch=True)),
        code(
            """
            from collections import deque
            import copy

            def optimal_action_np(states):
                return np.tanh(np.sin(np.pi * states))

            def reward_fn_np(states, actions):
                return 1.0 - (actions - optimal_action_np(states)) ** 2

            def optimal_action_torch(states):
                return torch.tanh(torch.sin(torch.tensor(np.pi, dtype=states.dtype) * states))

            def reward_fn_torch(states, actions):
                return 1.0 - (actions - optimal_action_torch(states)) ** 2

            class DeterministicActor(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(1, 32),
                        nn.ReLU(),
                        nn.Linear(32, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1),
                        nn.Tanh(),
                    )

                def forward(self, x):
                    return self.net(x)

            class GaussianActor(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.backbone = nn.Sequential(
                        nn.Linear(1, 32),
                        nn.ReLU(),
                        nn.Linear(32, 32),
                        nn.ReLU(),
                    )
                    self.mean = nn.Linear(32, 1)
                    self.log_std = nn.Linear(32, 1)

                def forward(self, x):
                    h = self.backbone(x)
                    mean = torch.tanh(self.mean(h))
                    log_std = torch.clamp(self.log_std(h), -2.5, -0.2)
                    return mean, log_std

                def sample(self, x):
                    mean, log_std = self.forward(x)
                    std = log_std.exp()
                    normal = torch.distributions.Normal(mean, std)
                    z = normal.rsample()
                    action = torch.tanh(z)
                    log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
                    return action, log_prob.sum(dim=1), mean

            class Critic(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(2, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1),
                    )

                def forward(self, states, actions):
                    x = torch.cat([states, actions], dim=1)
                    return self.net(x).squeeze(-1)

            def train_ddpg_style(steps=900, batch_size=128):
                actor = DeterministicActor()
                critic = Critic()
                actor_targ = copy.deepcopy(actor)
                critic_targ = copy.deepcopy(critic)
                actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
                critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
                replay = deque(maxlen=5000)
                tau = 0.02
                reward_curve = []

                for step in range(steps):
                    state = np.random.uniform(-1.0, 1.0, size=(1, 1)).astype(np.float32)
                    with torch.no_grad():
                        action = actor(torch.tensor(state)).numpy()
                    action = np.clip(action + np.random.normal(0, 0.15, size=action.shape), -1.0, 1.0)
                    reward = reward_fn_np(state, action).astype(np.float32)
                    next_state = np.random.uniform(-1.0, 1.0, size=(1, 1)).astype(np.float32)
                    replay.append((state[0], action[0], reward[0], next_state[0], 1.0))

                    if len(replay) >= batch_size:
                        batch = random.sample(replay, batch_size)
                        states_b = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
                        actions_b = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32)
                        rewards_b = torch.tensor(np.array([b[2] for b in batch]).reshape(-1), dtype=torch.float32)

                        q_values = critic(states_b, actions_b)
                        critic_loss = nn.functional.mse_loss(q_values, rewards_b)
                        critic_opt.zero_grad()
                        critic_loss.backward()
                        critic_opt.step()

                        actor_actions = actor(states_b)
                        actor_loss = -critic(states_b, actor_actions).mean()
                        actor_opt.zero_grad()
                        actor_loss.backward()
                        actor_opt.step()

                        for online, target in zip(actor.parameters(), actor_targ.parameters()):
                            target.data.mul_(1 - tau).add_(tau * online.data)
                        for online, target in zip(critic.parameters(), critic_targ.parameters()):
                            target.data.mul_(1 - tau).add_(tau * online.data)

                    if (step + 1) % 20 == 0:
                        eval_states = torch.linspace(-1, 1, 200).unsqueeze(1)
                        with torch.no_grad():
                            eval_actions = actor(eval_states)
                            eval_reward = reward_fn_torch(eval_states, eval_actions).mean().item()
                        reward_curve.append({"step": step + 1, "algorithm": "DDPG-style", "avg_reward": eval_reward})
                return actor, pd.DataFrame(reward_curve)

            def train_sac_style(steps=900, batch_size=128, alpha=0.15):
                actor = GaussianActor()
                q1 = Critic()
                q2 = Critic()
                actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
                q1_opt = torch.optim.Adam(q1.parameters(), lr=1e-3)
                q2_opt = torch.optim.Adam(q2.parameters(), lr=1e-3)
                replay = deque(maxlen=5000)
                reward_curve = []

                for step in range(steps):
                    state = np.random.uniform(-1.0, 1.0, size=(1, 1)).astype(np.float32)
                    with torch.no_grad():
                        action, _, _ = actor.sample(torch.tensor(state))
                    action_np = action.numpy()
                    reward = reward_fn_np(state, action_np).astype(np.float32)
                    next_state = np.random.uniform(-1.0, 1.0, size=(1, 1)).astype(np.float32)
                    replay.append((state[0], action_np[0], reward[0], next_state[0], 1.0))

                    if len(replay) >= batch_size:
                        batch = random.sample(replay, batch_size)
                        states_b = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
                        actions_b = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32)
                        rewards_b = torch.tensor(np.array([b[2] for b in batch]).reshape(-1), dtype=torch.float32)

                        q1_loss = nn.functional.mse_loss(q1(states_b, actions_b), rewards_b)
                        q2_loss = nn.functional.mse_loss(q2(states_b, actions_b), rewards_b)
                        q1_opt.zero_grad()
                        q1_loss.backward()
                        q1_opt.step()
                        q2_opt.zero_grad()
                        q2_loss.backward()
                        q2_opt.step()

                        sampled_actions, log_prob, _ = actor.sample(states_b)
                        q_min = torch.minimum(q1(states_b, sampled_actions), q2(states_b, sampled_actions))
                        actor_loss = (alpha * log_prob - q_min).mean()
                        actor_opt.zero_grad()
                        actor_loss.backward()
                        actor_opt.step()

                    if (step + 1) % 20 == 0:
                        eval_states = torch.linspace(-1, 1, 200).unsqueeze(1)
                        with torch.no_grad():
                            eval_actions, _, means = actor.sample(eval_states)
                            eval_reward = reward_fn_torch(eval_states, eval_actions).mean().item()
                        reward_curve.append({"step": step + 1, "algorithm": "SAC-style", "avg_reward": eval_reward})
                return actor, pd.DataFrame(reward_curve)

            ddpg_actor, ddpg_curve = train_ddpg_style()
            sac_actor, sac_curve = train_sac_style()
            control_df = pd.concat([ddpg_curve, sac_curve], ignore_index=True)
            control_df.head()
            """
        ),
        code(
            """
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=control_df, x="step", y="avg_reward", hue="algorithm", marker="o")
            plt.title("Continuous Control on a Synthetic Contextual Bandit")
            plt.ylabel("average reward on evaluation grid")
            plt.show()
            """
        ),
        code(
            """
            state_grid = torch.linspace(-1, 1, 200).unsqueeze(1)
            with torch.no_grad():
                ddpg_actions = ddpg_actor(state_grid).numpy().reshape(-1)
                _, _, sac_means = sac_actor.sample(state_grid)
                sac_actions = sac_means.numpy().reshape(-1)

            curve_df = pd.DataFrame(
                {
                    "state": state_grid.numpy().reshape(-1),
                    "optimal": optimal_action_np(state_grid.numpy().reshape(-1)),
                    "DDPG-style": ddpg_actions,
                    "SAC-style mean": sac_actions,
                }
            )
            curve_long = curve_df.melt(id_vars="state", var_name="policy", value_name="action")
            curve_df.head()
            """
        ),
        code(
            """
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=curve_long, x="state", y="action", hue="policy")
            plt.title("Learned Continuous Policies vs Optimal Action Curve")
            plt.ylabel("action")
            plt.show()

            true_q = np.array([1.2, 1.1, 0.95])
            single_q, clipped_double_q = [], []
            for _ in range(10000):
                q_a = true_q + np.random.normal(0, 0.35, size=3)
                q_b = true_q + np.random.normal(0, 0.35, size=3)
                single_q.append(np.max(q_a))
                clipped_double_q.append(np.max(np.minimum(q_a, q_b)))
            bias_compare_df = pd.DataFrame(
                {
                    "estimator": ["single critic max", "clipped double max", "true max"],
                    "value": [np.mean(single_q), np.mean(clipped_double_q), np.max(true_q)],
                }
            )
            """
        ),
        code(
            """
            plt.figure(figsize=(8, 4.5))
            sns.barplot(data=bias_compare_df, x="estimator", y="value", palette="crest")
            plt.title("Why TD3 Uses Clipped Double Critics")
            plt.show()
            """
        ),
        md(
            r"""
            ## 场景匹配建议

            - **DDPG**：适合做确定性策略与连续控制的入门基线，但对超参数和 Critic 偏差敏感。
            - **TD3**：当 DDPG 出现明显过估计、Q 值爆炸、策略被单点误差带偏时，TD3 往往是更稳的替代。
            - **SAC**：当探索困难、动作分布多峰、奖励噪声较大或需要更稳定泛化时，优先考虑 SAC。

            在工业控制或机器人任务中，SAC 和 TD3 常常是默认首选；PPO 则更多用于 on-policy、仿真稳定或并行采样容易的场景。
            """
        ),
        md(
            r"""
            ## 主要资料

            - DDPG（2015）: https://arxiv.org/abs/1509.02971
            - TD3（2018）: https://arxiv.org/abs/1802.09477
            - SAC（2018）: https://arxiv.org/abs/1801.01290
            """
        ),
    ]
    write_notebook("19_连续控制与现代深度强化学习体系.ipynb", cells)


def llm_rl_po_notebook():
    cells = [
        md(
            r"""
            # 20. LLM 后训练中的 RL 与 PO 体系
            大语言模型后训练的核心问题不是“继续做一次预训练”，而是把模型从“会补全文本”推进到“更符合人类偏好、任务目标和系统约束”。这一步通常被称为 post-training，包括监督微调（SFT）、奖励建模、在线强化学习和离线偏好优化（Preference Optimization, PO）。

            截至 **2026-03-17**，公开文献与主流开源实践已经形成两条相互交织的主线：

            - **RL 主线**：PPO、RLOO、GRPO、GSPO、SAPO、GiGPO 等，强调从在线采样中构造策略改进；
            - **PO 主线**：DPO、ORPO、KTO、SimPO、chiPO 等，强调直接从偏好数据构造可优化目标，不显式训练奖励模型或 critic。

            本 notebook 的目标是把这些方法放进统一视角：它们都在回答同一个问题，即“如何稳定地把偏好信号转化为策略更新”。
            """
        ),
        *professional_cells(
            model_title="LLM 后训练中的策略优化问题",
            definition=r"""
            对大语言模型而言，策略就是条件分布 $\pi_\theta(y\mid x)$，其中 $x$ 是提示词，$y$ 是完整响应。后训练的目标不再只是最大化下一个 token 的似然，而是最大化某种**序列级偏好目标**。如果有奖励函数 $r(x,y)$，可以写成
            $$
            \max_\pi \ \mathbb{E}_{x\sim\mathcal{D}, y\sim\pi(\cdot\mid x)}[r(x,y)] - \beta \, D(\pi \,\|\, \pi_{\mathrm{ref}})
            $$
            其中 $\pi_{\mathrm{ref}}$ 通常是 SFT 模型，$D$ 可以是 KL 或其他散度，$\beta$ 控制“向奖励靠拢”和“不要偏离参考模型太远”之间的平衡。
            """,
            io_spec=r"""
            输入是提示 $x$、候选响应 $y$、偏好标签或奖励分数；输出仍然是语言模型参数 $\theta$。不同方法的差别在于：

            - 学习信号是**成对偏好**、**分组相对分数**还是**在线采样回报**；
            - 更新单位是**token 级**、**序列级**还是**组级**；
            - 是否显式训练 reward model / critic；
            - 是否需要 old policy / reference policy / trust region。
            """,
            architecture=r"""
            后训练链路通常分为四段：

            1. 预训练模型提供通用语言能力；
            2. SFT 把输出分布拉到任务相关区域；
            3. 偏好数据、规则反馈或可验证奖励提供“好坏排序”；
            4. RL 或 PO 把这些排序/奖励转换为参数更新。

            PPO、GRPO、GSPO、SAPO、GiGPO 更像“在线策略优化器”；DPO、ORPO、KTO、SimPO、chiPO 更像“离线偏好目标构造器”；GDPO 则试图把偏好优化与分布级多样性建模连接起来。
            """,
            objective=r"""
            本体系的关键目标函数包括：

            - **PPO / RLHF**：
              $$
              \mathcal{L}_{\text{PPO}} = \mathbb{E}\left[\min\left(r_t \hat{A}_t, \operatorname{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right] - \beta \,\mathrm{KL}(\pi_\theta \,\|\, \pi_{\mathrm{ref}})
              $$

            - **DPO**：
              $$
              \mathcal{L}_{\text{DPO}} =
              -\log \sigma\left(
              \beta \left[
              \log \frac{\pi_\theta(y_w\mid x)}{\pi_\theta(y_l\mid x)}
              -
              \log \frac{\pi_{\mathrm{ref}}(y_w\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
              \right]\right)
              $$

            - **GRPO**：对同一提示采样一组响应，利用组内相对奖励构造归一化优势
              $$
              A_i = \frac{r_i - \mathrm{mean}(r)}{\mathrm{std}(r)+\varepsilon}
              $$
              再做 PPO 风格更新，但不显式训练 value model。

            - **GSPO / SAPO / GiGPO**：继续改造组式策略优化中的 ratio 控制与 credit assignment。
            """,
            complexity=r"""
            LLM 后训练的难点不只是优化器本身，而是信号噪声、长度偏差、奖励可被投机利用、KL 漂移、样本成本和工程吞吐共同作用的结果。一个看似更“先进”的目标函数，如果不能控制长度偏置、难以并行采样或对 batch 组成极端敏感，实际效果可能反而更差。

            因此评价现代 PO / RL 方法时，必须同时看：稳定性、样本效率、是否依赖 critic、是否需要在线采样、是否保留多样性、是否容易 reward hacking。
            """,
            comparison=r"""
            PPO 是 RLHF 的经典基线，但需要 value model、old policy 和较重的在线训练链路。DPO 家族绕过 critic，适合离线偏好数据；GRPO 通过组内相对比较去掉 value model，在大模型在线训练中更高效；GSPO 和 SAPO 聚焦 ratio 控制的稳定性；GiGPO 进一步把信用分配扩展到 group-in-group 结构；GDPO 则强调分布级多样性与多模态高质量输出，而不是单峰模式塌缩。
            """,
        ),
        architecture_diagram_cell("rlhf_pipeline"),
        architecture_diagram_cell("group_po"),
        md(
            r"""
            ## 统一视角：从奖励正则化到偏好优化

            许多方法表面形式不同，但可以放进同一个奖励正则化框架：
            $$
            \pi^\star(y\mid x) \propto \pi_{\mathrm{ref}}(y\mid x)\exp\left(\frac{1}{\beta}r(x,y)\right).
            $$
            这表示最优策略会在参考模型分布上对高奖励响应做指数重加权。差别在于：

            - PPO 用在线采样 + advantage 近似这个过程；
            - DPO 把偏好对 $(y_w, y_l)$ 直接转成 logit 差的分类目标；
            - ORPO 把偏好项叠加到有监督目标上；
            - KTO 用“值得保留 / 值得压低”的 Kahneman-Tversky 风格效用构造目标；
            - SimPO 去掉参考模型，直接用当前策略的平均 log-prob margin 做比较；
            - chiPO 则用更稳健的散度形式缓和 KL 型目标的极端梯度。
            """
        ),
        md(
            r"""
            ## 重点算法详解

            ### 1. PPO 与 RLOO

            PPO 是 RLHF 的标准起点。它依赖 reward model 与 value model，通过 old policy ratio 限制策略偏移。RLOO（REINFORCE Leave-One-Out）则在不训练 critic 的前提下，用同组样本的 leave-one-out baseline 估计优势，降低训练链路复杂度，常被视为从 PPO 走向 group-based RL 的过渡方法。

            ### 2. GRPO

            GRPO（Group Relative Policy Optimization，2024-02-06）对同一提示采样一组候选响应，通过组内相对奖励构造标准化优势，从而移除 value model。它特别适合“同提示下相对质量可比较”的任务，例如数学推理、代码生成、可验证答案任务。GRPO 的关键优点是工程简化，但它对组内奖励方差、长度偏差和 token 级 ratio 的累积噪声较敏感。

            ### 3. GDPO

            GDPO（GFlowNet-based Distributional Preference Optimization，EMNLP 2024）不把目标理解成“找唯一最优输出”，而是试图学习**与奖励成比例的高质量分布**。这使它在多峰偏好任务中更强调多样性保持，特别适合存在多种等价高质量答案的场景。可以把它理解成“把偏好优化从点估计推进到分布估计”。

            ### 4. GSPO

            GSPO（Generalized / Group Sequence Policy Optimization，2025-07-24 的公开论文）把更新重心从 token 级 ratio 推向**序列级 ratio**。这背后的判断是：LLM 偏好通常是序列级质量，不应被单个 token 的极端比例扰动主导。序列级 ratio 更接近真正的响应级偏好信号，因此在长序列场景下更稳定。公开资料显示，GSPO 已成为 2025 年后 group-based LLM RL 中的重要方向之一。

            ### 5. SAPO

            SAPO（Softmax / Smoothed Approximate Policy Optimization，2025-11-25 arXiv）针对 PPO / GRPO 中硬 clipping 带来的不可导拐点与梯度截断问题，引入更平滑的 ratio 抑制机制。它的核心思想不是彻底放弃 trust region，而是把“超过阈值就硬截断”改成“偏离越大，惩罚越平滑地增强”。这在高方差序列奖励下更容易获得连续稳定的梯度。

            ### 6. GiGPO

            GiGPO（Group-in-Group Policy Optimization，2025-05-15）把 group-based RL 从“每个 prompt 一个组”进一步推进到“组中再分组”的层级信用分配。对于多轮对话、agent trajectory、带工具调用的复杂流程任务，它允许同时建模全局轨迹质量与局部关键步骤质量，从而缓解“整条轨迹一个总分”过于粗糙的问题。
            """
        ),
        md(
            r"""
            ## 适用场景对比

            | 方法 | 数据类型 | 是否在线采样 | 是否需要 critic / reward model | 更新粒度 | 典型优势 | 主要风险 |
            | --- | --- | --- | --- | --- | --- | --- |
            | PPO | 在线轨迹 + reward | 是 | 通常需要 reward model 与 value model | token / trajectory | 通用 RLHF 基线成熟 | 工程链路重，训练成本高 |
            | RLOO | 在线组采样 | 是 | 不需要 critic | trajectory / group | 简化 PPO | 方差仍可能较大 |
            | DPO | 离线偏好对 | 否 | 不需要 reward model | pairwise sequence | 实现简单、训练稳定 | 受参考模型与数据偏差影响 |
            | ORPO | 离线偏好对 + SFT | 否 | 不需要 reward model | pairwise sequence | 一步联合训练 | 目标耦合较强 |
            | KTO | 离线正负示例 | 否 | 不需要 reward model | sequence | 不要求严格成对数据 | 效用刻画依赖设计 |
            | SimPO | 离线偏好对 | 否 | 不需要 reference model | sequence | 更轻量 | 对 margin 与长度敏感 |
            | GRPO | 在线组采样 | 是 | 不需要 value model | group relative | 工程更轻、适合可验证任务 | 组方差与 token ratio 噪声 |
            | GDPO | 离线 / 近离线偏好分布 | 可选 | 不强调传统 critic | distribution | 保持多样性、避免单峰塌缩 | 实现与解释门槛较高 |
            | GSPO | 在线组采样 | 是 | 不需要 value model | sequence-level ratio | 长序列更稳定 | 仍需高质量组内奖励 |
            | SAPO | 在线组采样 | 是 | 不需要 value model | smooth ratio control | 梯度更平滑 | 新方法，工程经验仍在积累 |
            | GiGPO | 在线层级轨迹 | 是 | 不需要传统 critic | group-in-group | 更细粒度信用分配 | 管线更复杂、标注更难 |
            """
        ),
        code(common_style_code()),
        code(
            """
            def softmax(x, axis=-1):
                x = x - np.max(x, axis=axis, keepdims=True)
                exp_x = np.exp(x)
                return exp_x / exp_x.sum(axis=axis, keepdims=True)

            prompts = [f"prompt_{i}" for i in range(1, 6)]
            responses = [f"resp_{j}" for j in range(1, 6)]

            reward_matrix = np.array(
                [
                    [0.95, 0.82, 0.40, 0.35, 0.20],
                    [0.60, 0.91, 0.88, 0.30, 0.10],
                    [0.15, 0.25, 0.92, 0.89, 0.55],
                    [0.78, 0.76, 0.20, 0.18, 0.12],
                    [0.25, 0.42, 0.73, 0.85, 0.81],
                ]
            )
            ref_logits = np.array(
                [
                    [1.4, 1.0, 0.2, -0.2, -0.5],
                    [0.6, 1.1, 0.9, -0.1, -0.4],
                    [-0.3, 0.1, 1.0, 0.9, 0.5],
                    [1.0, 0.9, 0.0, -0.2, -0.4],
                    [-0.1, 0.2, 0.7, 1.1, 0.8],
                ]
            )
            policy_logits = ref_logits + np.array(
                [
                    [0.10, -0.05, 0.20, -0.10, -0.15],
                    [-0.10, 0.05, 0.08, -0.05, -0.08],
                    [-0.05, -0.03, 0.12, 0.05, 0.02],
                    [0.05, 0.04, -0.10, -0.08, -0.10],
                    [-0.08, -0.02, 0.04, 0.10, 0.06],
                ]
            )

            pi_ref = softmax(ref_logits, axis=1)
            pi_cur = softmax(policy_logits, axis=1)

            reward_df = pd.DataFrame(reward_matrix, index=prompts, columns=responses)
            reward_df
            """
        ),
        code(
            """
            plt.figure(figsize=(9, 5))
            sns.heatmap(reward_df, annot=True, cmap="YlGnBu", fmt=".2f")
            plt.title("Toy Prompt-Response Reward Matrix")
            plt.show()
            """
        ),
        code(
            """
            beta = 3.0
            winner_idx = reward_matrix.argmax(axis=1)
            loser_idx = reward_matrix.argmin(axis=1)

            dpo_records = []
            for i, prompt in enumerate(prompts):
                y_w = winner_idx[i]
                y_l = loser_idx[i]
                delta_model = np.log(pi_cur[i, y_w] / pi_cur[i, y_l])
                delta_ref = np.log(pi_ref[i, y_w] / pi_ref[i, y_l])
                margin = beta * (delta_model - delta_ref)
                loss = -np.log(1 / (1 + np.exp(-margin)))
                dpo_records.append(
                    {
                        "prompt": prompt,
                        "winner": responses[y_w],
                        "loser": responses[y_l],
                        "dpo_margin": margin,
                        "dpo_loss": loss,
                    }
                )
            dpo_df = pd.DataFrame(dpo_records)
            dpo_df
            """
        ),
        code(
            """
            group_advantages = []
            gdpo_targets = []
            for i, prompt in enumerate(prompts):
                rewards = reward_matrix[i]
                advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
                for response, reward, adv in zip(responses, rewards, advantages):
                    group_advantages.append(
                        {"prompt": prompt, "response": response, "reward": reward, "grpo_advantage": adv}
                    )

                reward_regularized = pi_ref[i] * np.exp(beta * rewards)
                reward_regularized = reward_regularized / reward_regularized.sum()
                for response, prob in zip(responses, reward_regularized):
                    gdpo_targets.append({"prompt": prompt, "response": response, "gdpo_target_prob": prob})

            grpo_df = pd.DataFrame(group_advantages)
            gdpo_df = pd.DataFrame(gdpo_targets)
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
            sns.barplot(
                data=grpo_df[grpo_df["prompt"] == "prompt_1"],
                x="response",
                y="grpo_advantage",
                palette="coolwarm",
                ax=axes[0],
            )
            axes[0].set_title("GRPO Normalized Group Advantages")

            sns.barplot(
                data=gdpo_df[gdpo_df["prompt"] == "prompt_2"],
                x="response",
                y="gdpo_target_prob",
                color="#54A24B",
                ax=axes[1],
            )
            axes[1].set_title("GDPO-style Reward-Proportional Target Distribution")
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            ratios = np.linspace(0.4, 1.8, 240)
            eps = 0.2
            grpo_update = np.minimum(ratios * 1.0, np.clip(ratios, 1 - eps, 1 + eps) * 1.0)
            gspo_update = np.minimum(ratios * 1.0, np.clip(ratios, 1 - eps, 1 + eps) * 1.0)
            sapo_gate = 1 / np.cosh((ratios - 1.0) / 0.18) ** 2
            sapo_update = ratios * sapo_gate

            ratio_df = pd.DataFrame(
                {
                    "ratio": ratios,
                    "GRPO / PPO clipped": grpo_update,
                    "GSPO sequence-level clipped": gspo_update,
                    "SAPO smooth gate": sapo_update,
                }
            )
            ratio_long = ratio_df.melt(id_vars="ratio", var_name="method", value_name="update_strength")
            """
        ),
        code(
            """
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=ratio_long, x="ratio", y="update_strength", hue="method")
            plt.axvline(1 - eps, color="gray", linestyle="--")
            plt.axvline(1 + eps, color="gray", linestyle="--")
            plt.title("Trust-Region Control: Hard Clip vs Smooth Gate")
            plt.ylabel("effective update strength")
            plt.show()
            """
        ),
        code(
            """
            token_ratios = np.array([1.02, 1.01, 0.99, 1.03, 1.65, 0.98, 1.00, 1.01])
            seq_ratio = token_ratios.prod() ** (1 / len(token_ratios))
            ratio_compare_df = pd.DataFrame(
                {
                    "unit": [f"token_{i}" for i in range(1, len(token_ratios) + 1)] + ["sequence_geomean"],
                    "ratio": list(token_ratios) + [seq_ratio],
                }
            )
            ratio_compare_df
            """
        ),
        code(
            """
            plt.figure(figsize=(10, 4.5))
            sns.barplot(data=ratio_compare_df, x="unit", y="ratio", palette="mako")
            plt.axhline(1.0, color="gray", linestyle="--")
            plt.title("Why GSPO Prefers Sequence-Level Ratio Control")
            plt.xticks(rotation=45)
            plt.show()
            """
        ),
        code(
            """
            trajectories = pd.DataFrame(
                {
                    "trajectory": ["traj_A"] * 4 + ["traj_B"] * 4 + ["traj_C"] * 4,
                    "stage": ["plan", "tool", "verify", "final"] * 3,
                    "local_score": [0.9, 0.4, 0.8, 0.85, 0.5, 0.95, 0.45, 0.60, 0.8, 0.82, 0.78, 0.88],
                    "global_score": [0.84] * 4 + [0.63] * 4 + [0.82] * 4,
                }
            )
            trajectories["global_adv"] = trajectories["global_score"] - trajectories["global_score"].mean()
            trajectories["local_adv"] = trajectories.groupby("stage")["local_score"].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-6)
            )
            trajectories["gigpo_credit"] = 0.6 * trajectories["global_adv"] + 0.4 * trajectories["local_adv"]
            trajectories
            """
        ),
        code(
            """
            gigpo_pivot = trajectories.pivot(index="trajectory", columns="stage", values="gigpo_credit")
            plt.figure(figsize=(8, 4.8))
            sns.heatmap(gigpo_pivot, annot=True, cmap="RdBu_r", center=0.0, fmt=".2f")
            plt.title("GiGPO-style Hierarchical Credit Assignment")
            plt.show()
            """
        ),
        md(
            r"""
            ## 实践判断标准

            在真实 LLM 后训练中，不应只问“某算法在榜单上是否更高”，而应同时问：

            - 它依赖的是离线偏好数据，还是昂贵的在线采样？
            - 它是否需要 reward model / critic，工程链路是否能承担？
            - 它的更新单位是 token、sequence 还是 group，是否与任务奖励粒度一致？
            - 它是否显著放大长度偏差、参考模型偏差或奖励作弊风险？
            - 它是适合单峰最优答案任务，还是更强调多样性与多模态高质量分布？

            这也是为什么当前主流实践往往不是“只用一个算法”，而是根据任务类型在 SFT、DPO 类方法和在线 group-based RL 方法之间做组合。
            """
        ),
        md(
            r"""
            ## 主要资料

            - InstructGPT / RLHF（NeurIPS 2022）:
              https://papers.nips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html
            - PPO（2017）: https://arxiv.org/abs/1707.06347
            - DPO（2023-05-29）: https://arxiv.org/abs/2305.18290
            - KTO（2024-02-02）: https://arxiv.org/abs/2402.01306
            - GRPO（2024-02-06）: https://arxiv.org/abs/2402.03300
            - ORPO（2024-03-12）: https://arxiv.org/abs/2403.07691
            - SimPO（2024-05-23）: https://arxiv.org/abs/2405.14734
            - chiPO（2024-07-18）: https://arxiv.org/abs/2407.13399
            - GDPO（EMNLP 2024）: https://aclanthology.org/2024.emnlp-main.951/
            - GiGPO（2025-05-15）: https://arxiv.org/abs/2505.10978
            - GSPO（2025-07-24）: https://arxiv.org/abs/2507.18071
            - SAPO（2025-11-25）: https://arxiv.org/abs/2511.20347
            """
        ),
    ]
    write_notebook("20_LLM后训练中的RL与PO体系.ipynb", cells)


def main():
    intro_notebook()
    linear_models_notebook()
    decision_tree_notebook()
    ensemble_notebook()
    clustering_dimensionality_notebook()
    mlp_notebook()
    cnn_notebook()
    rnn_notebook()
    transformer_notebook()
    optimizer_loss_notebook()
    training_stability_notebook()
    llm_moe_notebook()
    peft_decoding_notebook()
    model_families_notebook()
    alignment_notebook()
    inference_system_notebook()
    rl_mdp_notebook()
    value_dqn_notebook()
    policy_gradient_notebook()
    continuous_control_notebook()
    llm_rl_po_notebook()
    print(f"已生成 {len(list(NOTEBOOK_DIR.glob('*.ipynb')))} 个 notebook 文件。")


if __name__ == "__main__":
    main()
