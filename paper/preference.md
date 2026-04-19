# 写作偏好总结 (Writing Preference Guide)

> 基于以下两篇论文总结的写作习惯与偏好：
> 1. TMTE: Effective Multimodal Graph Learning with Task-aware Modality and Topology Co-evolution (NeurIPS'26)
> 2. AdaFGC: Adaptive Federated Graph Clustering via Global Community-aware Contrastive Learning (NeurIPS'26)

---

## 1. 论文整体结构偏好

### 1.1 章节组织
- **主体结构**：Abstract → Introduction → Preliminaries/Problem Formulation → (Empirical Study, 可选) → Methodology → Experiments → Conclusion
- **附录结构**：Table of Content → Related Works → More Experiments → Theoretical Proofs → Pseudocode → More Experimental Setups → Environment → Dataset Details → Baseline Details → Limitations & Broader Impact
- **Related Works 放在附录**（非正文），正文空间留给更核心的 Empirical Study 或方法细节
- **Empirical Study 独立成节**：如有必要（如 TMTE），在 Preliminaries 之后、Methodology 之前加入经验性分析章节，用于为方法设计提供动机支撑

### 1.2 页面布局偏好
- 善用 `wrapfigure` 和 `wraptable`：将小型表格/图片嵌入正文段落旁，节省空间
- 全宽度 figure 采用 `width=0.998\textwidth`
- 嵌入式图/表采用 `[RT]` 右对齐定位
- 每篇论文有一个全幅 framework overview 图

---

## 2. Abstract 写作偏好

### 2.1 逻辑线
1. **背景定位**（1-2 句）：定义研究对象/数据结构，说明其重要性
2. **问题引出**（1-2 句）：指出现有方法的核心问题/limitation，用 "However" 转折
3. **方法提出**（2-3 句）：提出方法名称（缩写+全称+下划线标注），概述核心思路
4. **技术细节**（1-2 句）：简要描述方法的关键技术组件，使用 "Concretely" 引出
5. **实验结论**（1 句）：强调实验规模和 state-of-the-art 结果
6. **代码链接**（1 句）：提供匿名仓库链接

### 2.2 语言特征
- Abstract 为一整段，不分段落
- 方法名称用 `\underline{\textbf{}}` 标注首字母缩写来源
- 使用 "Extensive experiments on X datasets ... demonstrate/show that ..." 结尾句式
- 典型转折词："However", "To address this challenge"
- 典型引出词："Concretely", "Specifically"

---

## 3. Introduction 写作偏好

### 3.1 逻辑线（4段式结构）
1. **第一段：大背景 + 研究意义**
   - 从宏观领域出发，定义核心概念，列举应用场景（附引用）
   - 引出研究方向的关注度（"has attracted growing attention in recent years"）
   - 使用 "On the one hand... on the other hand..." 展示研究方向的多面性

2. **第二段：现有问题/Limitation**
   - 使用 "Despite their notable advances" 或类似转折开头
   - **问题编号化呈现**：用 **(1) \textbf{问题名称}**, **(2) \textbf{问题名称}** 格式列出 2-3 个核心问题
   - 每个问题附 1-2 句解释，包含具体例子和引用
   - 对现有方法的不足做精准定位："Consequently, ... inevitably ... which compromises ..."
   - Limitation 用 **L1**, **L2** 等粗体标签命名（如 AdaFGC）

3. **第三段：方法动机 + 方法概述**
   - 以斜体**研究问题**开头（"*How can we ...?*"），明确本文要解决的核心问题
   - 或以 "Building upon these insights, we propose..." 引出方法
   - 提出方法名称（全称+缩写），概述核心思路
   - 阐述 **key insight**（"Our key insight lies in..."），强调方法的直觉来源
   - 使用编号 **(1)**, **(2)** 分点阐述 insight 的多个维度
   - 以 "Concretely, ..." 引出技术细节的简要描述
   - 描述方法的闭环/迭代特性

4. **第四段：Contributions**
   - 以 `\textbf{Our Contributions:}` 开头
   - 固定**三点贡献**格式：
     - **(1) \textbf{In-depth Investigation / Valuable Insights}**：强调对问题的深刻分析
     - **(2) \textbf{Novel Method}**：概述方法的核心创新
     - **(3) \textbf{State-of-the-art Performance}**：强调实验的全面性和优越性
   - 贡献描述简洁，每条 1-2 句

### 3.2 语言与排版特征
- 研究问题用斜体呈现（如 "*How can we adaptively learn ...*"）
- 广泛使用粗体+斜体组合标注关键概念
- 使用 `$\rightarrow$` 表达因果关系（如 "Modality $\rightarrow$ Topology"）
- 引用密集但精准，每个论点附 1-2 个代表性引用

---

## 4. Preliminaries / Problem Formulation 写作偏好

### 4.1 结构
- 使用 **Problem Formulation** 作为标题（而非 "Preliminaries" 在 TMTE 中）
- 或使用 **Preliminaries** 包含 Problem Formulation + 相关概念介绍（如 AdaFGC）
- 数学符号定义紧凑清晰，在引入数据结构时统一定义所有符号
- 下游任务以编号列表呈现，分为类别（如 Graph-centric Tasks / Modality-centric Tasks）

### 4.2 符号体系偏好
- 图：$\mathcal{G} = (\mathcal{V}, \mathcal{E}, ...)$
- 节点集：$\mathcal{V}$，边集：$\mathcal{E}$，节点数：$N = |\mathcal{V}|$
- 邻接矩阵：$\mathbf{A}$，度矩阵：$\mathbf{D}$
- 归一化邻接矩阵：$\tilde{\mathbf{A}} = \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$
- 拉普拉斯矩阵：$\mathbf{L} = \mathbf{I} - \tilde{\mathbf{A}}$
- 特征矩阵：大写粗体 $\mathbf{X}$，特征向量：小写粗体 $\mathbf{x}$
- 嵌入/隐藏表示：$\mathbf{H}$, $\mathbf{Z}$
- 模态索引集：$\mathcal{M}$
- 可学习参数：$\mathbf{W}$, $\mathbf{\Phi}$, $\mathbf{w}$
- 损失函数：$\mathcal{L}$ 加下标（如 $\mathcal{L}_{\mathrm{anc}}$, $\mathcal{L}_{\mathrm{node}}$）
- 温度参数：$\tau$
- 平衡系数：$\lambda$, $\eta$, $\alpha$

### 4.3 写作习惯
- 在 Preliminaries 中，如果需要介绍相关工作的背景知识，使用粗体分段标题（如 "\textbf{Attributed Graph Clustering.}", "\textbf{Federated Graph Learning.}"）
- 每个概念介绍段落以该概念的定义或最新进展开头，引用 3-5 篇代表性工作
- 最后一段通常回到本文要解决的问题，承接 Introduction 中的 Limitation

---

## 5. Empirical Study / Motivation 写作偏好

### 5.1 结构（仅 TMTE 有独立 Empirical Study）
- 开头明确提出**研究问题**（Research Questions），使用 **Q1**, **Q2**, **Q3** 编号
- 描述实验设置：**Datasets, Methods, and Tasks**（粗体标题）
- 描述对比变量：**Topology Variants**（粗体标题），编号 (1), (2), (3) 描述每种变体
- 分析结论以**粗体陈述句**作为小标题，后附 "(Answer for Qx)" 标注
  - 如 "\textbf{The original topology is not optimal (Answer for Q1).}"
- 每个结论段落紧密关联实验图表，通过 "As shown in Fig.X" 引用
- 嵌入式图（wrapfigure）节省空间

### 5.2 分析风格
- 先给出观察结论（粗体），再解释原因
- 使用对比分析："In contrast", "More importantly", "Notably"
- 结论上升到方法设计层面，为后续 Methodology 做铺垫

---

## 6. Methodology 写作偏好

### 6.1 开篇结构
- 第一段为**概述段**：说明本节将介绍什么，给出 framework overview 图的引用
- 概述段列出子模块及其对应的小节编号（"We then elaborate on the key components: ... in Sec.X, ... in Sec.Y, and ... in Sec.Z"）
- 全幅 framework 图放在 Method 节中（`\includegraphics[width=0.998\textwidth]`）

### 6.2 子模块组织
- 每个子模块为一个 `\subsection`，有明确的标题
- 子模块标题命名模式："[动作/过程] from/via [来源/手段]"
  - 如 "Topology Evolution from Original Modality Feature Space"
  - 如 "Modality Evolution from Evolved Topology"
- 子模块内部使用**粗体段落标题**（`\textbf{}`）区分不同技术组件
  - 如 "\textbf{Multimodal and Multi-perspective Similarity Metric Learning.}"
  - 如 "\textbf{Node-anchor Affinity Matrix for Scalability.}"
  - 如 "\textbf{Learning Smooth Fused Representations.}"
  - 如 "\textbf{Global Community-aware Clustering.}"
- 每个公式有编号（`\label{eq: xxx}`），且在正文中明确引用
- 公式后紧跟符号解释（"where $\odot$ denotes..., $\tau$ is..."）

### 6.3 公式呈现偏好
- 重要公式单独成行（`equation` 环境），辅助公式可内联
- 复杂推导放附录（Theoretical Proofs），正文只展示关键结果和直觉
- 公式编号连续，通过 `\eqref{}` 引用
- 公式推导的完整证明使用 `\begin{proof}...\end{proof}` 环境
- 使用定理环境（Theorem, Lemma）呈现理论结果

### 6.4 最终目标函数
- 最终优化目标放在 Method 最后一个 subsection
- 总损失函数简洁明了：$\mathcal{L} = \mathcal{L}_1 + \eta \mathcal{L}_2$
- 描述组件之间的闭环/协同关系

### 6.5 方法描述的逻辑线
- 先动机/问题 → 再技术方案 → 再数学形式化 → 再实现细节/效率讨论
- 使用 "As discussed in Sec.X" 频繁回溯前文建立联系
- 使用 "Motivated by..." 或 "Inspired by..." 引出技术方案的理论基础
- 对每个模块的直觉（intuition）做解释："For intuition, ..."

---

## 7. Experiments 写作偏好

### 7.1 开篇结构
- 第一段声明实验设置概述 + 详细设置推到附录
- 明确列出**研究问题**（Research Questions），使用 **Q1**-**Q4/Q6** 编号
- 典型问题模式：
  - Q1: 与现有方法对比（Main Results）
  - Q2: 可解释性/可视化分析（Interpretability Investigation）
  - Q3: 消融实验（Ablation Study）
  - Q4: 超参数敏感性（Hyperparameter Sensitivity）
  - Q5: 效率分析（Efficiency Analysis）
  - Q6: 鲁棒性分析（Robustness Analysis）

### 7.2 Experimental Setup
- 使用粗体段落标题分块：**Datasets.**, **Baselines.**, **Downstream Tasks.** / **Clustering Metrics.**
- Datasets 描述：列出数据集名称、领域，总数，简洁概述
- Baselines 分类描述：(1) 类别一; (2) 类别二
- 详细设置推附录（"Due to space limitations, ... are provided in Appendix X"）

### 7.3 Main Results 写作模式
- 以 "To answer **Q1**" 开头
- 分数据集类别或任务类别组织结论
- **段落粗体小标题**区分不同任务/对比维度（如 "\textbf{Graph Tasks.}", "\textbf{Modality Tasks.}"）
- 定量结论格式："+X.XX% in Metric" 或 "surpasses ... by +X.XX%"
- 先数据说结论 → 再解释原因 → 上升到方法层面
- 强调 "consistently achieves the best/state-of-the-art performance"
- 表格中使用颜色标注前三名（darkred=第一，royalblue=第二，orange=第三）

### 7.4 Ablation Study 写作模式
- 以 "To address **Q2/Q3**" 开头
- 由于模块间有复杂交互，不能简单移除，因此定义**变体**（variants）
- 变体以斜体命名："*One-shot Topology Evolution (One-shot TE)*"
- 分析每个变体的性能下降，指出哪个模块最关键
- 关键分析模式："Notably, *[variant]* generally causes the largest degradation, highlighting the critical role of [module]"

### 7.5 Robustness Analysis
- 研究拓扑噪声/特征噪声等干扰场景
- 使用逐步增加扰动比例的图表展示鲁棒性
- 强调方法在噪声场景下的稳定性优势

### 7.6 Efficiency Analysis
- 比较参数量、每轮训练/推理时间
- 使用 wraptable 或 wrapfigure 展示效率结果
- 强调方法在效果和效率之间的 trade-off

### 7.7 表格排版偏好
- 使用 `\toprule`, `\midrule`, `\bottomrule` 和 `\thickhline` 分隔
- 表头灰色背景 (`\rowcolor{gray!80}`) + 白色字体
- 奇偶行交替灰色背景 (`\rowcolor{gray!10}`)
- 结果高亮：`\textcolor{darkred}{\mathbf{}}` (最佳), `\textcolor{royalblue}{\mathbf{}}` (次佳), `\textcolor{orange}{\mathbf{}}` (第三)
- 结果附标准差：`$\mathbf{XX.XX_{\pm X.XX}}$`

---

## 8. Conclusion 写作偏好

### 8.1 结构（1段式）
- Conclusion 为简洁的**单段总结**，5-7 句话
- 逻辑线：
  1. "In this paper, we..." 引出问题回顾
  2. "... identify / revisit ..." 概述发现的问题
  3. "Motivated by... / we propose..." 概述方法
  4. "... iteratively / jointly / adaptively ..." 描述方法核心机制
  5. "Experiments show/demonstrate that ..." 概述实验结论
  6. 可选：展望句（"We believe this work provides a valuable foundation for ..."）

### 8.2 语言特征
- 不引入新信息，是全文的浓缩
- 使用更高层次的概括性语言
- 强调方法的核心创新点和实验验证的广泛性

---

## 9. Appendix 写作偏好

### 9.1 组织结构
- 有独立的 **Table of Content** 页
- 分节清晰，每节有 `\label{appendix: xxx}` 供正文引用
- Related Works 放在附录最前面
- More Experiments 紧随其后
- Theoretical Proofs 用 Theorem + Proof 环境
- Pseudocode 用 Algorithm2e 环境
- Dataset Details 包含统计表格 + 文字描述
- Baseline Details 每个 baseline 一段，用粗体方法名开头
- Limitations and Broader Impact 放在最后

### 9.2 Pseudocode 偏好
- 使用 `algorithm2e` 包，带行号、规则线、竖线
- 有 Input 和 Output 声明
- 使用 `\tcc{}` 添加注释分隔不同阶段
- 对应正文公式编号的引用

---

## 10. 通用写作风格偏好

### 10.1 转折与连接词使用
- 转折："However", "Despite", "Nevertheless", "In contrast"
- 推进："Furthermore", "Moreover", "In addition", "Notably"
- 具体化："Specifically", "Concretely", "In particular"
- 因果："Consequently", "Therefore", "Thus", "As a result"
- 总结："Overall", "In summary", "These results indicate/demonstrate/highlight that"

### 10.2 强调手法
- 关键概念使用 `\textbf{}` 或 `\textit{}` 或两者结合
- 方法名称使用 `\textbf{}` 加下划线标注缩写来源
- Limitation/问题使用粗体编号标签（**L1**, **L2**）
- 研究问题使用粗体编号（**Q1**, **Q2**）

### 10.3 引用风格
- 每个技术论点附 1-3 个引用
- 应用场景列举时用 `~\cite{}` 紧跟
- 相关领域的综述以 "Comprehensive reviews can be found in recent surveys~\cite{}" 引用
- 方法描述中引用时使用方法名+引用："FedNCN~\cite{fedncn}"

### 10.4 句式偏好
- **避免使用破折号（em-dash `---`）**：不使用破折号做插入语或补充说明，改用从句（e.g., "that/which/where ..."）、冒号、或调整句式结构来表达
- 被动语态和主动语态混用，偏好主动语态描述方法贡献
- 频繁使用 "we" 作为主语："we propose", "we observe", "we further", "we conduct"
- 结论性语句常用 "consistently", "demonstrates/demonstrates stable superiority", "highlighting the importance/effectiveness of"
- 定量比较："improves over / surpasses ... by +X.XX% in Metric"
- 实验规模强调："Extensive experiments on X datasets demonstrate..."

### 10.5 命名偏好
- 方法名为大写缩写，如 TMTE, AdaFGC
- 方法命名结构：[核心特点] + [领域/任务] + [技术手段]
- 模块命名清晰，如 "Topology Evolution", "Modality Evolution", "Global Community-aware Contrastive Learning"

---

## 11. 图表设计偏好

### 11.1 Framework Overview 图
- 全幅宽度，放置于 Methodology 章节中
- 包含所有核心模块的流程示意
- Caption 以 "Overview of the proposed [method] framework, which [简述功能]." 格式

### 11.2 实验结果图
- 柱状图/折线图用于对比分析
- 热力图用于超参数分析
- t-SNE 可视化用于 embedding 分析
- Caption 以粗体前缀标注图类型："\textbf{Experimental results of our ...}"

### 11.3 表格
- Caption 以粗体前缀描述："**\textbf{Per-epoch efficiency}** on [dataset]"
- 分组使用 `\cline` 或 `\multirow`
- 表头灰底白字，数据行灰白交替

---

## 12. 论证逻辑偏好

### 12.1 Problem → Motivation → Solution 三段式
- 每个技术组件的引入都遵循：先说问题 → 再给动机 → 再给方案
- 不直接展示技术，总是先解释 "为什么需要这个"

### 12.2 Bidirectional/Closed-loop 论证
- 偏好强调方法中的**双向耦合/闭环/协同优化**关系
- 如 TMTE："Modality ↔ Topology Co-evolution"
- 如 AdaFGC："Client-side clustering ↔ Server-side anchor evolution"

### 12.3 Research Questions 驱动
- Empirical Study 和 Experiments 均以 Research Questions 组织
- 每个分析结论/实验结果段落以 "(Answer for Qx)" 标注
- 先提问 → 再展示实验 → 再回答问题

### 12.4 层次化设计
- 方法设计通常包含多层次/多粒度的组件
  - TMTE: multi-perspective → multi-modality → co-evolution
  - AdaFGC: community-level → node-level → topology-level
- 实验分析也层次化：主实验 → 消融 → 超参数 → 效率 → 鲁棒性
