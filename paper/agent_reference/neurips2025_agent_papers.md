# NeurIPS 2025 Agent 论文精选

> 选择原则：优先选 **NeurIPS 2025 Spotlight**、主题与 LLM agent / multi-agent / web agent / computer-use agent 高相关、且对后续写作和方法设计有参考价值的论文。

---

## 1. G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems

- **会议/级别**：NeurIPS 2025 Spotlight
- **主题**：multi-agent memory / self-evolving MAS
- **链接**：
  - OpenReview: https://openreview.net/forum?id=mmIAp3cVS0
  - arXiv: https://arxiv.org/abs/2506.07398

### 一句话总结
这篇论文把 multi-agent system 的“记忆”问题做成了一个三层图结构，让 agent team 能跨任务复用高层经验，也能回溯细粒度协作轨迹。

### 核心问题
现有 MAS 的 memory 机制过于粗糙，通常只保存最终答案、摘要或简单历史，无法表达真实的 inter-agent collaboration process，也不支持跨试次、跨任务地持续演化。

### 方法要点
- 提出三层 memory hierarchy：
  - `Insight Graph`：存高层、可泛化的经验与教训
  - `Query Graph`：存任务查询及其关联关系
  - `Interaction Graph`：存细粒度 agent 间通信和执行轨迹
- 检索时做双向 traversal：
  - 向上取 generalizable insights
  - 向下取最相关的 interaction subgraph
- 执行后再把新的协作轨迹写回三层图，实现持续进化

### 结果与价值
- 在 5 个 benchmark、3 个 LLM backbone、3 个 MAS framework 上评估
- embodied action 和 knowledge QA 上分别最高提升 **20.89%**、**10.12%**
- 优势不只是性能，还强调 **token cost 可控** 和 **plug-and-play**

### 对你有用的点
- 很适合借鉴其“**问题诊断 -> 架构化解决方案 -> 检索/更新闭环**”的写法
- 如果你自己的工作也涉及经验积累、历史复用、协作轨迹建模，这篇是很直接的参考

---

## 2. Distilling LLM Agent into Small Models with Retrieval and Code Tools

- **会议/级别**：NeurIPS 2025 Spotlight
- **主题**：agent distillation / tool-using small agents
- **链接**：
  - OpenReview: https://openreview.net/forum?id=VkicTqszOn
  - arXiv: https://arxiv.org/abs/2505.17612

### 一句话总结
这篇论文不再只蒸馏 chain-of-thought，而是直接蒸馏完整的 agent trajectory，让小模型学会“思考 + 调工具 + 观察 + 修正”。

### 核心问题
传统 CoT distillation 对 factual knowledge 和精确 computation 很脆弱，小模型容易 hallucinate，也很难在测试时泛化到新事实、新计算需求。

### 方法要点
- 提出 **Agent Distillation**
- 蒸馏对象从静态推理文本升级为 **reason-act-observe trajectory**
- 两个关键设计：
  - `first-thought prefix`：提高 teacher 轨迹质量
  - `self-consistent action generation`：提升 student 在测试时的鲁棒性
- agent 可调用 retrieval 和 code tools，而不是把所有知识和算力硬塞进参数

### 结果与价值
- 在 8 个 factual / mathematical reasoning 任务上验证
- 0.5B、1.5B、3B 小模型能达到接近更大一档 CoT-distilled 模型的表现
- 论文的真正贡献是证明：**可以蒸馏“agent behavior”，而不只是蒸馏“文本推理痕迹”**

### 对你有用的点
- 如果你后面写 PSAgent，也想强调行为层面而不是 token 层面的能力转移，这篇的 framing 很值得学
- 它对“为什么不是普通 CoT，而是 agent trajectory”解释得非常清楚

---

## 3. Web-Shepherd: Advancing PRMs for Reinforcing Web Agents

- **会议/级别**：NeurIPS 2025 Spotlight
- **主题**：web agent / process reward model / verifier
- **链接**：
  - OpenReview: https://openreview.net/forum?id=G2kMroO9UV
  - arXiv: https://arxiv.org/abs/2505.15277

### 一句话总结
这篇论文为 web agent 提出了专门的 process reward model，并把数据、训练集、meta-eval benchmark 一起补齐了。

### 核心问题
web navigation 是长程 sequential decision making，单靠 prompted MLLM evaluator 太贵、太慢、也不稳定；而 outcome reward 在 web 场景里又不够细，无法有效指导中间步骤。

### 方法要点
- 提出专门面向 web trajectory 的 **PRM: Web-Shepherd**
- 用结构化 checklist 把高层用户目标分解为可判定的 step-level subgoals
- 同时构建两套资源：
  - `WebPRM Collection`：40K step-level preference pairs
  - `WebRewardBench`：用于评估 PRM 的 meta-benchmark

### 结果与价值
- 在 WebRewardBench 上，比 GPT-4o evaluator 高约 **30 个点**
- 在 WebArena-lite 上，用 GPT-4o-mini policy + Web-Shepherd verifier，可获得 **+10.9** 的性能提升
- 同时成本约低 **10x**

### 对你有用的点
- 很适合学习其“**系统问题必须同时给出 model + data + benchmark**”的完整叙事
- 这篇对 efficiency、deployment cost、real-world usability 的强调很强，agent 论文里非常加分

---

## 4. Language Models can Self-Improve at State-Value Estimation for Better Search

- **会议/级别**：NeurIPS 2025 Spotlight
- **主题**：search agent / self-improvement / value model
- **链接**：
  - OpenReview: https://openreview.net/forum?id=W2874Arl4g
  - arXiv: https://arxiv.org/abs/2503.02878

### 一句话总结
这篇论文证明了 value model 可以不依赖人工奖励，自举式地学会更好的 lookahead，从而更高效地指导 search agent。

### 核心问题
在 web tasks 等交互式环境里，获取 ground-truth reward 或 demonstration 成本很高，而 tree search 的效果又强依赖 value model 的质量。

### 方法要点
- 提出 **Self-Taught Lookahead (STL)**
- 核心思想是把 “一步 lookahead + 状态转移 + rationale” 变成可学习文本目标
- 用 environment state transitions 自举改进 value model，而不是依赖人工奖励
- 学到的 value model 在推理时替代昂贵的大模型 evaluator

### 结果与价值
- 在 web agent 和 math reasoning 上都验证有效
- 相比之前的 LLM tree search，性能提升约 **20%**
- 成本降低约 **37x**
- 8B open-weight value model 可接近使用 GPT-4o 作为 value model 的效果

### 对你有用的点
- 这篇非常适合学习如何把“一个 RL / search 直觉”翻译成自然语言可训练对象
- 它的叙事方式也很强：从 cost bottleneck 切入，再把方法解释成 Bellman-style lookahead 的语言模型版本

---

## 5. OpenCUA: Open Foundations for Computer-Use Agents

- **会议/级别**：NeurIPS 2025 Spotlight
- **主题**：computer-use agent / data engine / foundation model
- **链接**：
  - OpenReview: https://openreview.net/forum?id=6iRZvJiC9Q
  - arXiv: https://arxiv.org/abs/2508.09123
  - Project: https://opencua.xlang.ai/

### 一句话总结
这篇论文做的是 open CUA foundation：从数据采集工具、跨平台大规模数据集、训练流水线到模型评测，一整套打通。

### 核心问题
computer-use agent 很热，但最强系统长期闭源，学术界缺少开放、可复现、可扩展的数据和基础模型体系。

### 方法要点
- OpenCUA 不是单一模型，而是一整套开放框架：
  - `AgentNetTool`：跨 Windows / macOS / Ubuntu 的演示采集工具
  - `AgentNet`：覆盖 200+ apps/sites 的大规模 computer-use dataset
  - 数据处理与训练流水线：把 demonstration 转成带 reflective long CoT 的 state-action pairs
  - `AgentNetBench`：离线评测基准
- 模型侧强调：
  - reflective CoT
  - multi-image histories
  - mixed-domain training

### 结果与价值
- OpenCUA-72B 在 OSWorld-Verified 上平均成功率 **45.0%**
- 论文更重要的价值在于：它建立了 open computer-use agent research 的基础设施

### 对你有用的点
- 如果你自己的工作也涉及环境、交互数据、复杂 agent pipeline，这篇很值得学其“**平台化叙事**”
- 它不是单点 trick，而是完整 research stack：tool -> dataset -> training pipeline -> benchmark -> model

---

## 6. 这 5 篇为什么值得优先看

- **都属于正式 NeurIPS 2025 Spotlight**，质量门槛相对更高
- 覆盖了 agent 研究里最重要的几条主线：
  - `memory`
  - `trajectory/behavior distillation`
  - `reward modeling / verifier`
  - `search / self-improvement`
  - `computer-use foundation`
- 都不是只做 prompt trick，而是有比较完整的方法论或基础设施贡献
- 对后续写论文尤其有帮助，因为它们普遍写得像“系统 + 方法 + 实验协议”一体化工作

---

## 7. 如果你只想先精读 2 篇

- **偏方法设计**：`G-Memory` + `Distilling LLM Agent into Small Models with Retrieval and Code Tools`
- **偏系统与评测**：`Web-Shepherd` + `OpenCUA`
- **偏 agentic search / RL 视角**：`Language Models can Self-Improve at State-Value Estimation for Better Search`

