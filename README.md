# 机器学习、深度学习、强化学习与大语言模型 Notebook 课程

本项目提供一套按知识体系拆分的 Jupyter Notebook 学习材料，覆盖：

- 经典机器学习模型
- 深度学习基础网络与训练机制
- 强化学习基础、值函数方法、策略梯度与连续控制
- 大语言模型架构、MoE、参数高效微调、推理系统与后训练优化

## Notebook 列表

1. `notebooks/00_课程导航与学习方法.ipynb`
2. `notebooks/01_线性模型体系.ipynb`
3. `notebooks/02_决策树体系.ipynb`
4. `notebooks/03_集成学习体系.ipynb`
5. `notebooks/04_聚类与降维体系.ipynb`
6. `notebooks/05_PyTorch基础与MLP体系.ipynb`
7. `notebooks/06_CNN体系.ipynb`
8. `notebooks/07_RNN_LSTM_GRU体系.ipynb`
9. `notebooks/08_Transformer体系.ipynb`
10. `notebooks/09_优化器与损失函数体系.ipynb`
11. `notebooks/10_训练稳定性与正则化技巧.ipynb`
12. `notebooks/11_大语言模型架构与MoE体系.ipynb`
13. `notebooks/12_参数高效微调与推理解码体系.ipynb`
14. `notebooks/13_现代开源模型家族与新架构体系.ipynb`
15. `notebooks/14_对齐训练与偏好优化体系.ipynb`
16. `notebooks/15_推理系统与部署优化体系.ipynb`
17. `notebooks/16_强化学习基础与MDP体系.ipynb`
18. `notebooks/17_值函数方法与DQN体系.ipynb`
19. `notebooks/18_策略梯度与Actor_Critic体系.ipynb`
20. `notebooks/19_连续控制与现代深度强化学习体系.ipynb`
21. `notebooks/20_LLM后训练中的RL与PO体系.ipynb`

## 课程特点

- 机器学习部分统一使用 `sklearn`
- 深度学习与深度强化学习部分统一使用 `PyTorch`
- 每个 notebook 都尽量包含：
  - 模型形式化定义
  - 输入输出与参数化方式
  - 架构与信息流说明
  - 目标函数、训练机制与复杂度分析
  - 代码实验与可视化
  - 相邻方法之间的比较与适用场景说明

## 目录说明

- `notebooks/`: 课程 notebook 文件
- `scripts/generate_notebooks.py`: 重新生成全部 notebook
- `scripts/validate_notebooks.py`: 顺序执行 notebook 代码单元并检查可运行性

## 重新生成

```powershell
python scripts\generate_notebooks.py
```

## 验证运行

```powershell
$env:TMP='C:\Users\41424\.codex\memories\tmpml'
$env:TEMP='C:\Users\41424\.codex\memories\tmpml'
$env:ML_DL_TMP='C:\Users\41424\.codex\memories\tmpml'
$env:MPLBACKEND='Agg'
$env:LOKY_MAX_CPU_COUNT='4'
python scripts\validate_notebooks.py
```

## 环境兼容说明

- notebook 会自动把临时目录设置到 ASCII 路径，避免 `scipy / sklearn` 在中文工作目录下创建临时文件时出错
- 聚类相关 notebook 包含针对当前 Windows 环境中 `KMeans` 线程探测问题的兼容处理
- 当前版本已经在本工作区完成生成与全量验证

## 强化学习与 PO 新增内容

新增强化学习主线与 LLM 后训练 RL/PO 主线，重点覆盖：

- MDP、Bellman 方程、DP / MC / TD
- SARSA、Q-learning、DQN、Double DQN、Dueling DQN、PER
- REINFORCE、Actor-Critic、A2C/A3C、TRPO、PPO
- DDPG、TD3、SAC 与连续控制中的熵正则化
- PPO、RLOO、GRPO、GDPO、GSPO、SAPO、GiGPO 与 DPO 家族的统一比较
