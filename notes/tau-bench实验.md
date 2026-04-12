## 总目标

两周内要拿到的，不是“完美论文”，而是这 4 样东西：

1. **一个能稳定跑的 derived benchmark / fixed-tree environment**

2. **至少 4 个方法的可复现实验结果**

3. **至少 3 组关键图表**

4. **一版能写进论文的实验结论**

如果两周想“所有 family 全做、真实 LLM agent 全接、理论和实验都极致完整”，大概率会炸。
所以你这 14 天必须按 **先闭环、再扩张** 的方式推进。

***

# 两周整体节奏

## Day 4：baseline 接口统一，只做 Group A 的 plumbing

把当前的 fixed-tree env 接成一个**统一 runner**，让 Group A 的 baseline 都能在同一个 env 上跑。

1. 定义统一 policy 接口

   * `select_path(instance, env)`

   * `update(result)`

   * `reset()`

2. 接三个最简单的 baseline

   * full-share

   * full-unshare

   * naive mixed

3. 写统一 runner

   * 输入：method / dataset / seed / episodes

   * 输出：episode logs / summary

4. 统一日志格式
   至少固定：

   * method

   * seed

   * instance\_id

   * selected\_path

   * leaf\_type

   * final\_action

   * oracle\_action

   * total\_cost

   * success



* 不接真实 DB

* 不接真实 evaluator

* 不接真实 LLM

* 不接 Group B / C



* `run_baseline.py`

* `baselines/full_share.py`

* `baselines/full_unshare.py`

* `baselines/naive_mixed.py`

***

## Day 5：把 $$\epsilon$$-EXP3 和 direct multi-stage EXP3 接进 Group A



把 Group A 需要的“原论文 / 直接多层 bandit” baseline 接进来。



1. 接 (\epsilon)-EXP3

   * 先做可运行版

   * 不求第一天就完全理论精修

2. 接 direct multi-stage EXP3

   * 每层各自用 EXP3

   * 不利用 risky/safe 结构

3. 接 random path baseline

4. 跑一轮 smoke test

   * 5～10 条样本

   * 每个 baseline 1 seed



* 所有 baseline 都能跑完

* 日志字段一致

* oracle best path 可以作为 reference 放进 summary



* `baselines/epsilon_exp3.py`

* `baselines/multistage_exp3.py`

* `baselines/random_path.py`

* 第一版 Group A smoke test 结果

***

## Day 6：先做 Group A 的小规模结果 + 修环境设计缺陷

### &#x20;

先把 Group A 跑出第一版结果，同时处理你刚发现的问题：**不要让 shared 路径天然永远最优**。

### &#x20;

1. 修 agent catalog
   重点把：

   * `g(a)`

   * agent 能力

   * agent cost
     解耦

2. 至少让以下现象能出现：

   * 有些 instance 上 best path 是 shared

   * 有些 instance 上 best path 是 unshared

   * mixed 结构不是摆设

3. 重新跑 Group A 小规模实验

   * 10～20 条样本

   * 2～3 seeds

   * 看平均 cost / success rate

因为如果不修这个，后面所有实验都会偏向 full-share。

### &#x20;

* 修正后的 `agent_catalog`

* 一版更合理的 env 行为

* Group A 小规模结果 v1

***

## Day 7：升级 evaluator，但先不接真实 DB

### &#x20;

把当前只看 `final_action` 对错的 surrogate evaluator 升级成**更像真实任务**的 evaluator。

### &#x20;

1. 设计 reservation-level evaluator
   至少支持：

   * false cancel penalty

   * false refuse penalty

   * subset mismatch penalty

   * cancelled ids / refused ids 对比

2. 对单 reservation 和多 reservation 分开处理

   * Tier 1：单 reservation

   * Tier 2：subset / multi-reservation

3. 保持 env 主体不变，只升级 cost 逻辑



因为你后面就算接真实 DB，也还是要有一个**适合 fixed-tree 的终局 cost**。
原 tau2 的 replay evaluator 不能直接拿来当你的 leaf cost。

### &#x20;

* `evaluator.py` 或增强版 `oracle_eval.py`

* 新 cost spec

* Group A 在新 evaluator 下的小跑结果

***

## Day 8：接真实 DB / 真实环境语义（第一阶段）

开始接真实 DB，但**只接 environment 语义，不接真实 LLM**。

1. 确认你要接入的真实信息源
   最少先用：

   * reservation snapshot

   * user snapshot

   * flight info

   * policy reference time

2. 让 Stage 2 / 3 不再完全吃现成 oracle
   改成：

   * 从真实或半真实 snapshot 中解析 candidate reservations

   * 从真实或半真实 snapshot 中计算 feature

3. 保留 Stage 4 / 5 先吃结构化逻辑
   先别全都真实化

这是“半真实 DB-backed derived env”，

不是把整个 tau2 orchestrator 搬回来。

* `hidden_env_state` / snapshot 接入

* DB-backed stage2 / stage3 v1

***

## Day 9：接真实 DB / evaluator 第二阶段，跑 Group A 主实验版

让 Group A 在“接近真实环境”的版本上正式跑第一批主实验。

1. 查 DB-backed env 是否稳定

2. 跑 Group A 正式版

   * Risky-PS

   * full-share

   * full-unshare

   * naive mixed

   * (\epsilon)-EXP3

   * direct multi-stage EXP3

   * random

   * oracle path

3. 先跑 2 seeds，少量 episodes

4. 画第一版主表



你应该已经有：

* **真实环境语义下的 Group A 初版结果**

这会成为你后面最重要的主干。



现在要完成的是一套 **3 个交互机制 × 8 个方法 × 1\~2 个数据集** 的实验主线：

### 三组机制

1. **算法直接选 agent**

2. **算法给分，agent 参考后再选**

3. **agent 自己选，不看算法**

### 每组都放的方法

* Risky-PS / 你的 half-share 方法

* all-share / full-share

* all-unshare

* multi-stage (\epsilon)-EXP3

* 直接多级 EXP3 / 结构无感知 per-node EXP3

* naive mixed

* random path

* oracle best fixed path（参考上界）

你已经强调了，**naive mixed 和 random path 绝对不能少**，我同意。

### 数据集

* **主数据集：1 个更大的 airline 子数据集**

* **可选鲁棒性数据集：第 2 个更大的 airline 子数据集**

* 当前退款这 10 条子数据集，只保留为 smoke test / debug，不进入最终主结果

## Day 1：锁定实验边界和数据集

目标：**把真实 LLM 接到 Stage 2 / 3 上**
&#x20;仍然用现在的小退款子集。

当天必须完成：

* &#x20;Stage 2 / 3 的 LLM-tool-DB 主链打通&#x20;

* &#x20;至少 3\~5 条小子集任务跑通&#x20;

* &#x20;保存完整 trace&#x20;

不要做：

* &#x20;正式数据集全量实验&#x20;

* &#x20;大规模 seed sweep&#x20;

***

## Day 2：做主数据集 adapter 和数据检查

目标：**切正式数据集的小样本 smoke**
&#x20;这一天才开始换正式数据集，但只抽 10\~20 条。

当天必须完成：

* &#x20;adapter 在正式数据集上可用&#x20;

* &#x20;Stage 2 / 3 tool path 在正式数据集上不崩&#x20;

* &#x20;至少跑通 10\~20 条&#x20;

不要做：

* &#x20;全矩阵实验&#x20;

* &#x20;大量 baseline 全开

***

## Day 3：接通 Stage 2 的真实 DB + LLM

目标：真正开始进入“耦合链”，不再扩展规则 planner。

必须完成：

* Stage 2 用真实 LLM 产出 tool call

* 通过原 bench 的真实工具：

  * 查用户

  * 查 reservation

* 输出：

  * candidate reservations

  * resolved reservations

  * resolution\_status

* 跑通主数据集上至少 5 条任务 smoke test

当天产出：

* Stage 2 trace

* 真实 tool call 样例

* 成功/失败 case 各若干

通过标准：

* Stage 2 不再靠规则 fake query 做主路

* 真实 LLM + DB 的 path 可以稳定运行几条任务

***

## Day 4：接通 Stage 3 的真实 DB + LLM

目标：让 Stage 3 从真实 reservation / flight 工具里提 feature。

必须完成：

* Stage 3 用真实 LLM 产出 tool call

* 调真实工具查：

  * reservation details

  * flight status

* 生成结构化 feature rows

* 和 Stage 4 成功对接

当天产出：

* Stage 2→3 全链路 trace

* feature rows 示例

* 10 条任务 smoke run

通过标准：

* Stage 3 不再吃现成 oracle feature

* Stage 4 可以消费真实 facts 产生 adjudication

***

## Day 5：把 3 组机制全部接进同一 runner

目标：从“能跑一条链”升级成“能跑完整实验矩阵”。

必须完成：

* 实现 3 组机制的统一接口：

  1. 算法直接选 agent

  2. 算法给分，agent 参考后再选

  3. agent 自己选，不看算法

* 把 8 个方法全部挂进统一 runner

* 确保每种方法都能在主数据集上完成 1 seed smoke test

当天产出：

* 统一实验 runner

* 机制切换开关

* 方法注册表

通过标准：

* 所有方法都能起跑

* 不是只跑 Risky-PS 一条

***

## Day 6：稳定评测和日志

目标：今天不扩实验规模，专注于“结果能不能信”。

必须完成：

* 主 evaluator 输出统一格式

* 辅助 bench-style 诊断输出统一格式

* episode / seed / method / dataset / regime 的日志路径定型

* 检查：

  * tool errors

  * DB state 污染

  * 同一实例跨 run 可复现性

* 先补 `oracle best fixed path`

当天产出：

* 稳定版日志 schema

* summary 表头

* oracle path 参考结果

通过标准：

* 你已经能自动汇总：

  * cost

  * success

  * DB diagnostics

  * shared/unshared 比例

* 不再需要手动翻日志看单条结果

***

## Day 7：主数据集小规模预跑

目标：先用 1 seed 跑完整主矩阵，找崩点。

必须完成：

* 在主数据集上跑：

  * 3 组机制 × 8 方法 × 1 seed

* 检查：

  * 哪些方法崩

  * 哪些 regime 不稳定

  * 哪些 task 类型特别异常

* 修 runner / prompt / state 隔离 / tool call 格式等工程 bug

当天产出：

* 第一版全矩阵结果

* 崩点清单

* bug 修复记录

通过标准：

* 主数据集全矩阵能完整跑完 1 seed

* 没有大面积 crash / 空结果

***

## Day 8：主数据集正式版

目标：跑出第一版可写主表的结果。

必须完成：

* 主数据集上跑：

  * 3 组机制 × 8 方法 × 2 seeds

* 汇总第一版主表

* 生成：

  * 总 cost 表

  * success / exact match 表

  * shared / unshared 比例表

  * 方法排序表

* 特别检查：

  * naive mixed

  * random path

  * oracle path
    是否都正常进入比较框架

当天产出：

* 第一版主表

* 第一版主图

* 主数据集完整 results.json/csv

通过标准：

* 已经有一版能放进报告/论文的主结果

* 方法间差异和三组机制差异能够被肉眼看出来

***

## Day 9：第二数据集 / 鲁棒性实验

目标：做“不是只在一个子集成立”的验证。

必须完成：

* 用第 2 个数据集跑**缩减版复现**

* 推荐只跑：

  * 3 组机制

  * 重点方法子集：

    * Risky-PS

    * full-share

    * full-unshare

    * naive mixed

    * random path

    * oracle path

* 若时间够，再补 EXP3 系列

当天产出：

* 第二数据集鲁棒性表

* 与主数据集结果的对比图

* 若第二数据集来不及全跑，至少跑核心方法

通过标准：

* 你的结论不再只建立在一个退款子集上

* 至少有一个“外部复现/鲁棒性”支撑

***

## Day 10：收尾、补图、失败案例分析

目标：今天不再大规模加新实验，只做打磨和收尾。

必须完成：

* 汇总最终主表和附表

* 画最终图：

  * 各机制对比

  * 各方法对比

  * cost / success / shared ratio

* 做失败案例分析：

  * resolution 错

  * feature 错

  * adjudication 错

  * execute/refuse 错

* 写出最终实验说明：

  * 数据集

  * 机制定义

  * baseline 说明

  * evaluator 说明

* 对最关键结果补 1 个 seed（如果前面某组差异很关键但还不稳）

当天产出：

* 最终表格

* 最终图

* failure cases

* 实验部分草稿

通过标准：

* 你已经可以开始写论文/报告中的实验部分

* 不是只有代码和 logs，而是已经有“可展示结果”

***

# 项目目录



已在 `/home/ubuntu/data/PSAgent` 下新建好这套实验目录，和现有的 `/home/ubuntu/data/PSAgent/tau2-bench` 并列，后续可以保持“原 benchmark 不乱改，derived benchmark 和算法单独开发”。

[ bench env](https://ocnm5tq3r01q.feishu.cn/wiki/PXoOwM9VmitUQYk6RtOcoImVn4g?from=from_copylink)

当前结构：

```plain&#x20;text
PSAgent/
├── data/
│   ├── raw/
│   ├── interim/
│   └── derived/
├── configs/
├── envs/
├── algos/
├── baselines/
├── scripts/
├── analysis/
├── outputs/
├── notes/
├── tests/
├── docs/
└── third_party/
```



每个目录建议这样用：



* `data/raw/`

放原始输入数据的镜像或只读拷贝，比如从 `tau2-bench` 抽出来的 `airline/tasks.json`、`db.json`、原始 task id 白名单。原则上不手改。



* `data/interim/`

放构建过程中的中间产物，比如 family filter 结果、normalized task dump、stage decomposition 草稿、人工校对表。这个目录很适合调 builder。



* `data/derived/`

放最终给算法训练/评测的 derived benchmark，例如 `train.jsonl`、`eval.jsonl`、`agent_catalog.json`、`schema_version.json`。



* `configs/`

放实验配置。后面可以分成数据构建配置、环境配置、算法超参、baseline 配置、实验 sweep 配置。



* `envs/`

放你的固定树环境代码。比如 `fixed_tree_env.py`、`stage_interfaces.py`、`leaf_protocol.py`、`cost_model.py`。这是从原 tau-bench 抽象出来的新环境层。



* `algos/`

放你的核心算法实现，比如 `Risky-PS`、`Partial-Share`、online learner、selector、update rule。这里尽量只放“算法本体”，不要混数据清洗逻辑。



* `baselines/`

放基线策略和对照组，例如 `all_share`、`all_unshare`、`random`、`greedy`、`oracle`。



* `scripts/`

放可执行脚本入口，比如：

`build_dataset.py`

`run_baseline.py`

`run_algo.py`

`sweep.py`

`evaluate.py`



* `analysis/`

放分析脚本和 notebook，做 regret 曲线、share/unshare 对比、family 分布、错误分析。



* `outputs/`

放实验结果，不放源码。可以再细分成 `runs/`、`figures/`、`tables/`、`logs/`。



* `notes/`

放研究笔记、任务归类、family 定义、实验决策记录。这个目录很适合你现在这种“边设计 benchmark 边推进算法”的工作流。



我额外补了几个目录：



* `tests/`

放你新环境、builder、算法协议的单元测试。这个很重要，尤其是你后面要验证 `z(l)=max_h g(a_h)`、safe/risky internal node、feedback routing。



* `docs/`

放更正式的设计文档，比如 derived benchmark spec、JSONL schema、MVP 路线图。



* `third_party/`

放未来可能引入但不想和主代码混在一起的外部参考实现、vendor wrappers、临时适配代码。



# 原bench task数据流

一条 task 的当前数据流：

* task source：runner.helpers.get\_tasks() 从 registry 找 task loader；airline loader 在 src/tau2/domains/airline/environment.py (line 35)，读 data/tau2/domains/airline/tasks.json。

* user simulator：默认是通用 src/tau2/user/user\_simulator.py (line 99)，把 task.user\_scenario 填进系统提示词。

* agent interaction：build\_agent() 把 environment 的 tools 和 policy 给 agent；orchestrator 在 turn-based 循环里让 agent/user 交替产出 message。

* tool execution：agent 发 ToolCall 后，Environment.get\_response() (line 437) 执行具体 toolkit 方法。

* environment / DB state：Environment.set\_state() 先重放初始数据和历史 tool call，再在 episode 中持续更新 DB。

* evaluation / reward：evaluate\_simulation() 根据 task 的 reward\_basis 组合 DB hash 对比、动作比对、通信检查、NL assertion。

* output logs / trajectories：保存在 SimulationRun.messages/ticks/reward\_info，批量结果写到 Results，文本任务默认单一 results.json，voice 走目录格式。

# PS Agent task 数据流

## 1. instance source

### 原 bench 对应

原 bench 是：

* `runner.helpers.get_tasks()` 从 registry 找 task loader

* airline loader 读 `data/tau2/domains/airline/tasks.json`

### 你这边对应

你这边不再直接读原始 `tasks.json` 作为 episode 输入，
而是读你自己的 **derived JSONL / derived task set**。

也就是：

* `fixed_tree_loader.get_instances()`

* 读取
  `data/derived/airline_cancellation_fixed_tree/train.jsonl`
  或
  `eval.jsonl`

### 它的职责

负责返回一个**结构化 instance**，包含：

* `instance_id`

* `family`

* `original_task_id`

* `user_context`

* `stage1~stage5 oracle`

* `hidden_env_state`

* `metadata`

### 一句话理解

原 bench 的 source 给你的是“任务文本”；

你的 source 给你的是“已结构化、可用于 fixed-tree 的 instance”。

***

## 2. fixed-tree loader

### 原 bench 对应

原 bench 没有这一步，因为它默认就是自由对话环境。

### 你这边对应

你需要一个显式的：

* `build_fixed_tree(instance)`

* 或 `FixedTreeEnvironment.from_instance(instance)`

### 它做什么

把一条 derived instance 变成当前 episode 的**固定树环境定义**：

* 当前一共几层

* 每层是什么 task slot

* 每层有哪些 candidate agents

* 哪些 stage input 可见

* 哪些 hidden oracle 只给 evaluator / env 看

### 输出

形成一个 episode-specific tree，例如：

* Stage 1 = user grounding

* Stage 2 = reservation resolution

* Stage 3 = feature extraction

* Stage 4 = adjudication

* Stage 5 = execute/refuse

并把 agent catalog 挂上去。

### 一句话理解

这一步相当于把“结构化样本”装配成“当前这一轮要玩的树”。

***

## 3. stage-wise candidate catalog / disclosure config

### 原 bench 对应

原 bench 的 `build_agent()` 会把工具和 policy 交给单个 agent。

### 你这边对应

你不是一个 agent，而是**每层多个候选 agent**。
所以你需要显式加载：

* stage-wise candidate agents

* 每个 agent 的 disclosure 属性 (g(a))

* 每个 agent 的 execution mode / cost / noise model

### 例如

Stage 1:

* `ground_regex_g0`

* `ground_llm_g0`

* `ground_crm_linker_g1`

Stage 2:

* `lookup_redacted_snapshot_g0`

* `lookup_live_db_g1`

* ...

### 它做什么

告诉系统：

* 本轮这个 stage 有哪些可选 action

* 这些 action 的 `g(a)` 是多少

* 后面 leaf 标签怎么从路径推出来

### 一句话理解

原 bench 是“给一个 agent 工具箱”；

你这里是“给每一层一个候选动作集合”。

***

## 4. policy selector

### 原 bench 对应

原 bench 里，agent 自己在对话中隐式决定下一步做什么。

### 你这边对应

这一层就是你的核心 learner：

* `RiskyPSPolicy`

* `EpsilonEXP3Policy`

* `AllSharePolicy`

* `AllUnsharePolicy`

* `NaiveMixedPolicy`

### 它做什么

在当前 round / episode 的 fixed tree 上：

* 从 root 开始

* 在 Stage 1 选一个 candidate

* 在 Stage 2 选一个 candidate

* ...

* 最终形成一条完整 path

例如：

```plain&#x20;text
[ground_llm_g0,
 lookup_live_db_g1,
 adjudicate_rule_engine_g0,
 executor_local_commit_g0]
```

### 输出

* `selected_path`

* 每层选择概率

* 整条路径概率 (\Pi\_t(\ell))

### 一句话理解

原 bench 的“agent interaction”是边对话边决定；

你的“policy selector”是在固定树上显式挑一条路。

***

## 5. stage execution on selected path

### 原 bench 对应

原 bench 的 agent 发 ToolCall，`Environment.get_response()` 执行工具。
DB 会持续更新。

### 你这边对应

你的环境不再执行“任意工具序列”，而是**按选中的 path 逐 stage 执行**。

也就是：

* Stage 1 agent 吃 `stage1.input`，输出一个 grounding result

* Stage 2 agent 吃 `stage2.input + stage1_output`，输出 resolved reservations

* Stage 3 agent 输出 policy features

* Stage 4 agent 输出 adjudication

* Stage 5 agent 输出 final action

### 这里的关键

这一步不一定是真实 LLM。

第一版完全可以是：

* rule-based executor

* scripted noisy agent

* oracle + controlled corruption

这也是你现在最应该先做的。

### 一句话理解

原 bench 的执行单位是“消息 + tool”；

你的执行单位是“stage input → stage output”。

***

## 6. leaf protocol derivation

### 原 bench 对应

原 bench 没有 shared / unshared protocol 这一步。

### 你这边对应

这是你实验最独特的一步。

在 path 执行完后，环境根据整条路径的 disclosure 属性决定这条 leaf 的协议：

\[

z(\ell)=\max\_h g(a\_h)

]

* 若 (z(\ell)=0)，leaf 是 shared

* 若 (z(\ell)=1)，leaf 是 unshared

同时内部节点 safe/risky 由后代 leaf 诱导。

### 它做什么

决定接下来：

* 用 shared update

* 还是用 unshared / IPS update

### 一句话理解

原 bench 评的是“任务有没有做对”；

你这里还要额外判断“这条正确/错误反馈该走哪种更新通道”。

***

## 7. terminal evaluation / cost construction

### 原 bench 对应

原 bench 的 `evaluate_simulation()` 会根据：

* DB hash

* action match

* communication

* NL assertions
  组合 reward。

### 你这边对应

你也要有一个 episode 末端 evaluator，但它更结构化。

我建议叫：

* `evaluate_leaf(path_outputs, oracle, cost_spec)`

### 它做什么

根据：

* Stage 5 最终动作

* reservation 是否 cancel 对了

* 是否误 cancel / 漏 cancel

* 是否最终与 oracle 一致

* 加上 stage agent costs

构造一个 terminal cost，例如：

\[

c\_t(\ell)=\sum\_h \text{agent\_cost}(a\_h) + \text{terminal\_loss}

]

其中：

* `terminal_loss = 0` 如果最终动作与 oracle 一致

* `terminal_loss = 1` 如果最终动作错误
  第一版先这样最稳。

你以后也可以细化成：

* false cancel penalty

* false refuse penalty

* subset mismatch penalty

* execution penalty

### 一句话理解

原 bench 的 reward 更贴近“整段 agent 模拟表现”；

你的 cost 是“这条 leaf path 在 fixed-tree 上的终局损失”。

***

## 8. learner update

### 原 bench 对应

原 bench 不做跨 episode 的在线学习更新；它更偏评测框架。

### 你这边对应

这一步是你的实验核心。

episode 得到 terminal cost 后：

* 若 leaf 是 shared：

  * 走 shared 更新

  * (\Delta) 沿祖先累计回传

* 若 leaf 是 unshared：

  * 走 bandit / IPS 更新

  * 用路径概率修正

  * 再以 unshared 方式影响祖先

这就是 Risky-PS 与 baseline 分开的地方。

### 一句话理解

原 bench 的 episode 跑完就结束；

你的 episode 跑完以后，policy 还要被更新，为下一轮服务。

***

## 9. episode logs / aggregate results

### 原 bench 对应

原 bench 会把：

* `SimulationRun.messages`

* `ticks`

* `reward_info`

* `Results`
  写出来。

### 你这边对应

你也应该有两层输出：

### episode-level log

每一轮记录：

* `instance_id`

* `selected_path`

* `path_prob`

* `leaf_type`

* `stage_outputs`

* `final_action`

* `oracle_action`

* `terminal_cost`

* `success`

* `update_type`

### aggregate results

按方法/seed 汇总：

* cumulative regret

* average cost

* success rate

* false cancel rate

* false refuse rate

* risky-node visit frequency

* unshared-leaf frequency

### 一句话理解

原 bench 保存“完整模拟轨迹”；

你保存“路径、终局 cost、更新结果、长期学习曲线”。

***

## 三、把它写成你论文/代码里的标准版

我建议你以后就这样写你的实验数据流：

```plain&#x20;text
derived instance source
-> fixed-tree loader
-> stage-wise candidate catalog + disclosure config
-> online policy selector
-> selected path execution
-> leaf protocol derivation
-> terminal evaluator / cost constructor
-> learner update
-> episode logs / aggregate results
```

***

## 四、再给你一个“具体到 cancellation family”的实例流

假设当前 instance 是 task `1`。

## Step 1: instance source

从 `train.jsonl` 读到：

* `instance_id = airline_cancel_task_1`

* family = cancellation\_refund

* stage oracle 都已经写好

## Step 2: fixed-tree loader

构造 5-stage tree：

* grounding

* resolution

* feature extraction

* adjudication

* execute/refuse

## Step 3: candidate catalog

挂上：

* Stage 1: `ground_regex_g0`, `ground_llm_g0`, `ground_crm_linker_g1`

* Stage 2: ...

* Stage 5: ...

## Step 4: policy selector

Risky-PS 选出一条路径：

* `ground_llm_g0`

* `lookup_live_db_g1`

* `adjudicate_rule_engine_g0`

* `executor_local_commit_g0`

## Step 5: stage execution

各 stage executor 顺次产出：

* user\_id / target route

* reservation `Q69X3R`

* hours since booking / insurance / ...

* deny\_outside\_24h

* final\_action = refuse\_all

## Step 6: leaf protocol derivation

因为 path 中含 `lookup_live_db_g1`，所以：

* (z(\ell)=1)

* 该 leaf 是 unshared

## Step 7: terminal evaluation

对比 oracle：

* 最终 refuse\_all 正确

* `terminal_loss = 0`

* 但有若干 agent\_cost

* 得到总 cost

## Step 8: learner update

因为是 unshared：

* 用 IPS / bandit 形式更新

* 不走 shared (\Delta) 回传

## Step 9: logs

写一条 episode log：

* selected path

* leaf type = unshared

* cost = ...

* success = 1

***

## 五、你这个数据流和原 bench 最大的差别

我帮你浓缩成 4 个核心差别：

## 1. 输入对象不同

* 原 bench：原始 task + user simulator prompt

* 你：derived structured instance

## 2. 执行单位不同

* 原 bench：message / tool-call loop

* 你：stage-by-stage path execution

## 3. reward 定义不同

* 原 bench：整段 simulation 的 evaluator score

* 你：leaf-level terminal cost

## 4. episode 后处理不同

* 原 bench：保存结果即可

* 你：还要做 online learner update

***

# 最终要哪些实验

1. **算法直接选 agent**

2. **算法给分，agent 参考后再选**

3. **agent 自己选，不看算法**

***

### 第一组：全算法选择

这是你的**主实验组**，也是最重要的一组。
这里建议至少放：

* **Risky-PS / 你的 half-share 方法**

* **all-share / full-share**

* **all-unshare**

* **原论文 multi-stage (\epsilon)-EXP3**

* **直接多级 EXP3 / 结构无感知的 per-node EXP3**

* **naive mixed**

* **random path**

* **oracle best fixed path**（参考上界，不算公平 baseline，但一定要放）

这里面最不能少的是 `naive mixed` 和 `random path`。

### 为什么

因为如果你只有：

* full-share

* full-unshare

* Risky-PS

* (\epsilon)-EXP3

别人会说：

> 你的提升是不是只是因为 mixed tree 本身容易，或者因为你手工把 unshared 放到了好位置？

所以你需要一个：

#### `naive mixed`

保留真实的 mixed disclosure 结构，但**不利用 risky/safe 结构**，只是普通做路径学习。
这样才能证明：

> 不是 mixed 本身有优势，而是你真的利用了 feedback structure。

#### `random path`

这是 sanity check。

如果 random 都不比某个 baseline 差，那 baseline 实现就有问题。

#### `oracle best fixed path`

你现在 env 已经能 brute-force 找最优 path 了，这个非常值。

它相当于：

* 理论上的静态最好路线

* regret 的参考对象

* debug 工具

这组实验要回答的是：

> 在“完全由算法直接选 agent”的前提下，谁最能利用 partial-share 的反馈结构？

这正是你的理论主线。你自己的 PS 推导、safe/risky 结构、all-share 与 all-unshare 两个端点，都属于这一组。

而 (\epsilon)-EXP3 则是原论文在 end-to-end bandit + education setting 下的主 baseline，也必须在这一组里。

***

### 第二组：算法给分，agent 参考后再选

这是你的**系统扩展组**，很有意思，但不是主理论组。

这里我建议对比：

* **Agent + Risky-PS score**

* **Agent + (\epsilon)-EXP3 score**

* **Agent + simple heuristic score**

* **Agent + top-k from Risky-PS**

* **Agent direct only**（不看分数，作为本组对照）

这里的关键不是再比一堆算法，而是比：

#### A. 给 agent 什么信号最有用

比如给：

* 历史平均 cost

* 近期成功率

* disclosure 风险标签

* shared/unshared 后果

* top-k 候选

#### B. 分数是“软参考”还是“硬筛选”

你可以拆成两个子版本：

* **soft guidance**：LLM/agent 看所有候选，只是参考分数

* **hard gating**：Risky-PS 先筛 top-k，agent 只在 top-k 里选

我更推荐你先做 `top-k + agent`，因为它最容易解释：

> 算法负责结构感知和裁剪搜索空间，agent 负责语义细判。

这组实验要回答的是：

> 你的算法除了能自己选，还能不能作为 router prior / routing signal，帮助 agent 做更好的选择？

这和你之前想法完全一致：
“上一层 agent/LLM 选择下一层 agent 时，可以参考算法给出的分数。”
这个方向是成立的，但更适合做**扩展实验**，不是主定理的承载组。

***

### 第三组：直接 agent 自己选

这是你的**无算法对照组**。

这里建议放：

* **direct agent routing**

* **direct agent routing + chain-of-thought / rationale**

* **direct agent routing + retrieval of agent descriptions**

* **rule-based heuristic router**（不是 LLM，但也不看你的算法）

* **random agent router**（可选）

这里的目的是回答：

> 如果完全不引入在线学习算法，只靠 agent/router 自己根据任务文本挑下一层 agent，会怎样？

这组很重要，因为它给你的整篇论文一个非常强的现实对照：

* 如果 agent 自己选已经很好，你要证明你的算法仍然有价值

* 如果 agent 自己选不稳定，你要证明 Risky-PS score 作为 routing prior 能明显改善它

***

## 我建议你最后的三组标题直接这么写

### Group A. Algorithmic Routing

全由算法直接选路径

### Group B. Score-Assisted Agent Routing

算法提供 routing score，agent 做最终决策

### Group C. Pure Agent Routing

不看算法，agent 直接自己选

这个分组方式非常整齐。

***

## 每组具体建议放哪些方法

### Group A：Algorithmic Routing

至少放这 8 个：

1. **Risky-PS**

2. **Full-Share**

3. **Full-Unshare**

4. **(\epsilon)-EXP3**

5. **Direct multi-stage EXP3**

6. **Naive Mixed**

7. **Random Path**

8. **Oracle Best Fixed Path**

#### 说明

* `Full-Share` 和 `Full-Unshare` 是你的两个结构端点。

* `\(\epsilon\)-EXP3` 是原论文主 baseline，强调 exploration-exploitation-education。

* `Direct multi-stage EXP3` 用来说明“直接把 EXP3 推到多层”不够。原论文也正是为了说明简单的一阶段 bandit 思路在多阶段下不够，才提出 (\epsilon)-EXP3。

* `Naive Mixed` 是最重要的结构无感知对照。

* `Random` 和 `Oracle` 分别是下界 sanity check 和上界参考。

***

### Group B：Score-Assisted Agent Routing

建议放这 5 个：

1. **Agent + Risky-PS score**

2. **Agent + (\epsilon)-EXP3 score**

3. **Agent + simple cost score**

4. **Agent + Risky-PS top-k**

5. **Direct Agent Routing**（本组 anchor）

#### 我最推荐的主版本

主做两个就够：

* `Direct Agent`

* `Agent + Risky-PS top-k`

因为这个最容易讲故事：

> 不让算法彻底取代 agent，而是让算法先做结构化裁剪，再由 agent 做语义判断。

***

### Group C：Pure Agent Routing

建议放这 3 到 4 个：

1. **Direct Agent Routing**

2. **Direct Agent + rationale**

3. **Heuristic Router**

4. **Random Router**（可选）

#### 为什么还要 `Heuristic Router`

因为不然别人会说：

> 你是在拿算法和一个很随意的 agent route 比。

加一个 heuristic router，可以说明：

* 不是所有非学习方法都差

* 但你的方法仍然更稳定/更好

***

## 你最终真正该比较的“核心对象”

如果你怕 baseline 太多，最核心的其实就这些：

### 主表最核心

* Risky-PS

* Full-Share

* Full-Unshare

* (\epsilon)-EXP3

* Naive Mixed

* Direct Agent

* Agent + Risky-PS top-k

* Oracle Best Fixed Path

这 8 个已经足够有论文味。

***

## 这三组分别在回答什么问题

### Group A 回答

**你的理论算法本身强不强？**

### Group B 回答

**你的算法能不能作为 agent 的 routing prior？**

### Group C 回答

**如果完全不用算法，只靠 agent 自己路由，会怎样？**

这样整篇实验的叙事就很完整，不会只是“又一个 bandit 算法对比”。

***

## 我建议你每组都统一看的指标

不然三组会不好对齐。

至少统一看：

* **average total cost**

* **success rate**

* **false cancel rate**

* **false refuse rate**

* **shared leaf ratio / unshared leaf ratio**

* **oracle gap**（和 oracle best fixed path 的差距）

如果是 Group A，还可以额外看：

* cumulative regret

* risky node visit frequency

# cancellation/refund eligibility 任务

用户提出取消+退款诉求→系统识别目标 reservation→抽取 policy 相关特征→做 eligibility adjudication→最终 refuse 或 cancel eligible reservations



5-stage schema：
intent grounding、reservation resolution、eligibility feature extraction、eligibility adjudication、execute/refuse

用户想做什么 → 是哪张单 → 相关事实是什么 → 政策怎么判 → 最后执行什么

cost：

顶层动作错：`+1.0`

&#x20;每个误取消：`+1.5`

&#x20;每个漏取消：`+1.0`

&#x20;每个误拒绝：`+1.0`

&#x20;每个漏拒绝：`+1.0`

&#x20;最后再加：`0.1 × path_agent_cost`



需要手动设置agent的强弱以及cost

现在暂时是每层，后面要再修改

**oracle\_g0**：正确、share-compatible、标准成本&#x20;

**weak\_g0**：share-compatible、更便宜、带轻度噪声&#x20;

**specialist\_g1**：restricted、成本更高，按名字应当更强，但当前还没实现出 specialist 行为&#x20;

**noisy\_g1**：restricted、最贵、带明显噪声



# 树结构设计

### task

```python
@dataclass
class TaskDescriptor:
    task_id: str
    attribute_weights: dict[int, float]
    stage_difficulty: dict[str, float]
```

### agent

```python
@dataclass
class AgentSpec:
    agent_id: str
    g: int
    base_cost: float

    competence_level: str      # high / low
    scope_level: str           # broad / narrow
    stability_level: str       # stable / unstable

    attribute_skill: dict[int, float]
```

### family

```python
@dataclass
class FamilySpec:
    family_name: str
    stages: list[str]
    stage_agents: dict[str, list[str]]
```

这样：

* tree family 复用

* dataset 可替换

* 同 agent 跨 task 表现不同

* 属性体系稳定

