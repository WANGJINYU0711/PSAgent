# Agent 论文写作偏好总结 (NeurIPS 2025)

> 基于以下 5 篇论文总结的写作风格与结构偏好：
> 1. G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems
> 2. Distilling LLM Agent into Small Models with Retrieval and Code Tools
> 3. Web-Shepherd: Advancing PRMs for Reinforcing Web Agents
> 4. Language Models can Self-Improve at State-Value Estimation for Better Search
> 5. OpenCUA: Open Foundations for Computer-Use Agents
>
> 参考风格文件：`/home/ubuntu/data/PSAgent/paper/preference.md`

---

## 1. 总体观察

这组论文整体上更偏 **agent systems / empirical ML / infrastructure + method** 风格，而不是纯理论或纯算法推导型。和你现有 `preference.md` 里的 graph learning 论文相比，它们有几个明显差异：

- **更强调系统闭环**：不是只讲一个 module，而是方法、数据、benchmark、cost、部署价值一起讲
- **更强调 practicality**：速度、成本、token usage、environment interaction 开销经常和主指标并列
- **更强调 benchmark 生态**：很多论文会同时贡献 framework / dataset / evaluator / benchmark
- **Related Work 往往留在正文**，而不是放附录
- **第一页就放大图或 teaser figure** 很常见，用来快速建立 problem-method-result 的全局印象

---

## 2. 论文整体结构偏好

### 2.1 常见正文结构

- **主体结构**：
  - Abstract
  - Introduction
  - Related Work
  - Preliminaries / Background / Problem Setup
  - Method
  - Experiments
  - Conclusion / Discussion

### 2.2 与你已有偏好文件的一个关键不同

- 这批 agent 论文大多 **不把 Related Work 放附录**
- 原因很直接：agent 赛道更新快、相邻工作多，作者通常需要在正文中尽早做 positioning
- 如果你的 PSAgent 工作与现有 agent literature 强相关，正文保留一节简短 Related Work 会比完全放附录更稳

### 2.3 首页布局偏好

- **第一页通常同时出现 Abstract + Figure 1**
- Figure 1 不只是 framework 图，经常兼具以下至少一个功能：
  - 任务示例
  - 方法总览
  - 性能/成本 teaser
  - 对比现有方法的直观证据
- 常见形式：
  - 左边问题现象，右边方法框架
  - 上方流程图，下方性能点图/柱状图

---

## 3. 标题与命名偏好

### 3.1 标题风格

- 高质量 agent 论文非常偏爱 **“名字 + 冒号 + 功能描述”**
- 典型模式：
  - `G-Memory: ...`
  - `Web-Shepherd: ...`
  - `OpenCUA: ...`

### 3.2 方法命名特点

- 名字通常短、可记忆、带隐喻或系统感
- 名字本身会暗示论文核心卖点：
  - `Memory`
  - `Shepherd`
  - `Open`
  - `Self-Taught`
- 如果你的方法也想走 NeurIPS agent 风格，建议方法名做到：
  - 2 个音节到 1 个短词缩写
  - 能自然承载系统意象
  - 能一眼看出关键能力

---

## 4. Abstract 写作偏好

### 4.1 标准逻辑线

1. **场景与重要性**：先说明 agent 所在环境或任务为什么重要
2. **现有瓶颈**：指出 cost / reliability / memory / data / reward / openness 等关键缺口
3. **提出方法/框架**：给出方法名，并用一句话定义它是什么
4. **展开组件**：常用 `(1) (2) (3)` 列出核心组成部分
5. **实验结果**：给出 1-3 个最有冲击力的数字
6. **开放资源**：代码、数据、模型、benchmark 是否开源

### 4.2 语言特征

- 一般是 **单段式 Abstract**
- 高频转折词：
  - `However`
  - `To address this`
  - `To bridge this gap`
  - `Despite`
- 高频结果句式：
  - `Extensive experiments ... demonstrate ...`
  - `Our results show that ...`
  - `achieves state-of-the-art ...`
- 高频组件列举方式：
  - `Our framework consists of: (1) ..., (2) ..., (3) ...`

### 4.3 可直接借鉴

- agent 论文的 abstract 特别强调 **“问题是真实存在的工程瓶颈”**
- 建议你自己的 abstract 少讲泛泛愿景，多讲：
  - 现有 agent 为什么不行
  - 你的系统到底补了哪一块缺口
  - 性能/成本/泛化/鲁棒性哪一项变好了

---

## 5. Introduction 写作偏好

### 5.1 常见 4 段式结构

1. **第一段：领域背景 + agent 价值**
   - 从 LLM agents / web agents / computer-use agents / MAS 的宏观背景切入
   - 快速列举应用场景
   - 说明该方向的重要性和现实价值

2. **第二段：现有方法的硬伤**
   - 不是泛泛说“仍有挑战”
   - 而是指出很具体的 failure mode：
     - memory 太粗糙
     - evaluator 太贵
     - reward 太稀疏
     - 小模型会 hallucinate
     - 闭源系统不可复现
   - 往往配 **具体数字** 或 **具体例子**

3. **第三段：本文方法与 key idea**
   - 直接给出方法名
   - 用一句话解释“它是什么”
   - 再用 2-4 句拆解成关键模块或关键机制

4. **第四段：贡献总结**
   - 很多论文会显式写 `Our contributions are summarized as follows`
   - 一般 3 点最常见：
     - 问题诊断 / 新视角
     - 新方法 / 新框架 / 新数据
     - 大规模实验结果

### 5.2 高频写法

- **研究问题显式化**
  - `a core research question emerges: ...`
  - `a natural question arises: ...`
- **方法导入句**
  - `In this work, we propose ...`
  - `To address these challenges, we present ...`
  - `In response to the above question, we introduce ...`
- **总结句**
  - `The key contribution of this paper is ...`
  - `To summarize, our work makes the following key contributions:`

### 5.3 风格特点

- 比较少空泛叙事，更多是“**现实问题 -> 方法设计**”
- 很喜欢用 **具体失败案例** 来证明痛点不是抽象的
- 很喜欢在 introduction 就提前讲：
  - 成本有多高
  - token 有多大
  - benchmark 有多难
  - 为什么 existing evaluator / memory / reward 不适用

---

## 6. Related Work 写作偏好

### 6.1 组织方式

- 通常放正文 `Section 2`
- 采用 **子方向拆分**，而不是一大段流水账
- 典型小节名：
  - `Reasoning distillation of language models`
  - `Language agents and agentic reasoning`
  - `Inference-time scaling for web agents`
  - `Rewards for web navigation`

### 6.2 写法特点

- 每个子方向先讲主流做法
- 再指出缺口
- 最后一句专门写“**Unlike prior work, we ...**”

### 6.3 对你的启发

- 如果 PSAgent 牵涉到多篇 agent baseline，建议正文留一节短 Related Work
- 不需要很长，但一定要形成清晰分类，否则 reviewer 很容易觉得定位不清

---

## 7. Preliminaries / Problem Setup 写作偏好

### 7.1 风格

- 通常比较短，服务于后文 method
- 重点不是堆定义，而是把任务 formalize 到能支撑方法和实验

### 7.2 常见形式

- `POMDP` 或 trajectory formalization
- state / action / observation / reward 定义
- multi-agent graph formalization
- distillation / search / rollout 的训练目标定义

### 7.3 写法特征

- 先给直觉，再给公式
- 公式数量通常 **够用即可**，不会像纯理论论文那样密
- 数学符号紧贴实际 agent 流程，不追求抽象美感，而追求可落地

---

## 8. Methodology 写作偏好

### 8.1 开篇结构

- 方法节开头通常先引用 Figure 1
- 先讲整体流程，再讲子模块
- 典型顺序：
  - overall pipeline
  - data construction / retrieval / update / reward / training
  - inference-time usage

### 8.2 子模块组织偏好

- 模块名通常非常清晰，直接对应系统组件
- 例子：
  - `Insight Graph / Query Graph / Interaction Graph`
  - `first-thought prefix`
  - `self-consistent action generation`
  - `WebPRM Collection`
  - `WebRewardBench`

### 8.3 论述逻辑

- **先说为什么这个模块必要**
- **再说怎么做**
- **最后说它如何和整体目标闭环**

### 8.4 这类 agent 论文很常见的额外模块

- dataset / annotation pipeline
- verifier / reward model
- search-time integration
- memory update mechanism
- scaling setup

### 8.5 一个非常明显的共性

- 方法不只写 training，还会明确写 **inference-time 怎么用**
- 因为 agent 工作里，test-time orchestration 往往就是性能来源的一部分

---

## 9. Experiments 写作偏好

### 9.1 核心实验问题

高质量 agent 论文通常不只问“性能是否更高”，而是同时回答：

- 是否更强
- 是否更便宜
- 是否更稳定
- 是否更能泛化
- 是否能 plug into existing frameworks

### 9.2 常见实验结构

1. 主结果
2. 泛化分析
3. 消融实验
4. 成本/效率分析
5. 额外案例分析或可视化

### 9.3 高频指标

- success rate
- accuracy
- average score
- token usage
- inference cost
- wall-clock time
- environment usage / states expanded

### 9.4 典型写法

- 结果段落会同时给 **绝对数值** 和 **相对提升**
- 很喜欢用这类表述：
  - `improves by X`
  - `outperforms ... while reducing cost by Y`
  - `without modifying the original framework`
  - `plug-and-play`

### 9.5 对 agent 论文特别重要的一点

- **效率不是附属指标，而是主结果的一部分**
- 所以如果你的 PSAgent 有通信、记忆、规划或 partial sharing 的优势，建议实验里显式加：
  - token budget
  - latency
  - tool calls / interaction turns
  - compute or API cost

---

## 10. 图表偏好

### 10.1 Figure 风格

- Figure 1 必须承担“快速说服 reviewer”的作用
- 常见类型：
  - 框架总览图
  - 任务示意 + 方法流程
  - 性能/成本 teaser

### 10.2 表格风格

- 主表通常突出：
  - 多 benchmark
  - 多模型规模
  - 多 setting
- 很多 agent 论文会把 **性能和成本放在同一表或同一图里**

### 10.3 最值得借鉴的点

- 不要只画“方法框图”
- 最好让前两张图分别回答：
  - 你的方法怎么工作
  - 它为什么值得关注

---

## 11. 语言风格总结

### 11.1 总体语气

- 直接
- 工程导向
- 少修辞，多证据
- 强调 bottleneck、scalability、cost-efficiency、real-world deployment

### 11.2 高频词

- `long-horizon`
- `cost-effective`
- `self-improvement`
- `plug-and-play`
- `generalization`
- `trajectory`
- `state transition`
- `step-level`
- `open-source`
- `foundation`

### 11.3 高频句式

- `To address this challenge, we propose ...`
- `The key idea is to ...`
- `Our framework consists of ...`
- `Extensive experiments show that ...`
- `We release ... to support future research`

---

## 12. 对 PSAgent 写作的具体建议

如果你后面想让论文更接近这批 NeurIPS 2025 agent paper 的风格，我建议：

- **正文保留 Related Work**
  - agent 方向 reviewer 很看重 positioning

- **第一页放一个强 Figure 1**
  - 同时展示 PSAgent 的流程、通信/共享机制、以及效果或成本收益

- **Introduction 里必须把真实瓶颈写具体**
  - 例如通信冗余、信息共享失真、局部最优、长程协作失败、token/cost 爆炸

- **贡献点不要写虚**
  - 最好写成：
    - 新问题刻画
    - 新机制/新框架
    - 更强且更省的实验结果

- **实验里把效率指标升格**
  - token、cost、轮数、延迟，至少选 2 个做主指标

- **强调系统闭环**
  - reviewer 更喜欢“机制 + 环境 + 协议 + benchmark + ablation”成套叙事

---

## 13. 推荐的 PSAgent 结构草案

- Abstract
- Introduction
- Related Work
- Problem Setup / Task Formalization
- PSAgent Framework
- Communication / Memory / Sharing Mechanism
- Training or Inference Algorithm
- Experiments
- Conclusion

如果篇幅紧张，`Problem Setup` 和 `Framework` 可以合并；如果实验动机很强，也可以像你原始偏好文件那样插入一个简短 `Empirical Motivation` 小节。

