# Distributed No-Regret Learning for Multi-Stage Systems with End-to-End Bandit Feedback

> Extracted from the PDF into Markdown for reference. Page boundaries are preserved.


## Page 1

```text
                                             Distributed No-Regret Learning for Multi-Stage Systems with
                                                            End-to-End Bandit Feedback
                                                                                                                           I-Hong Hou
                                                                                                                 ihou@tamu.edu
                                                                                                     Department of ECE, Texas A& M University
                                                                                                            College Station, Texas, USA

                                         ABSTRACT                                                                                    edge server, the edge server needs to decide which neural network
                                         This paper studies multi-stage systems with end-to-end bandit                               to employ for this job. The performance of the job depends on the
arXiv:2404.04509v2 [cs.LG] 16 Aug 2024


                                         feedback. In such systems, each job needs to go through multiple                            accuracy of the result and the end-to-end latency, which includes
                                         stages, each managed by a different agent, before generating an                             both communication and computation delays. As another example,
                                         outcome. Each agent can only control its own action and learn the                           consider packet deliveries in multi-hop networks consisting of
                                         final outcome of the job. It has neither knowledge nor control on                           multiple routers. Upon receiving a packet, a router needs to decide
                                         actions taken by agents in the next stage. The goal of this paper                           which router to forward the packet to. The performance of the
                                         is to develop distributed online learning algorithms that achieve                           packet depends on the end-to-end latency.
                                         sublinear regret in adversarial environments.                                                   This paper studies the problem of designing distributed online
                                             The setting of this paper significantly expands the traditional                         learning algorithms under which all agents jointly learn the optimal
                                         multi-armed bandit problem, which considers only one agent and                              decisions with minimum coordination, even when the outcomes of
                                         one stage. In addition to the exploration-exploitation dilemma in the                       decisions are determined by an adversary. Developing such algo-
                                         traditional multi-armed bandit problem, we show that the consid-                            rithms are challenging due to three major reasons. First, in most
                                         eration of multiple stages introduces a third component, education,                         computer and network systems, it is desirable to employ distributed
                                         where an agent needs to choose its actions to facilitate the learn-                         algorithms where each agent can only make decisions of its own
                                         ing of agents in the next stage. To solve this newly introduced                             action and has neither knowledge nor control on the actions taken
                                         exploration-exploitation-education trilemma, we propose a simple                            by agents in the next stages. Second, in many systems, an agent
                                         distributed online learning algorithm, 𝜖−EXP3. We theoretically                             can only observe the end-to-end outcome of the joint effects of all
                                         prove that the 𝜖−EXP3 algorithm is a no-regret policy that achieves                         stages, but cannot know how actions taken in each individual stage
                                         sublinear regret. Simulation results show that the 𝜖−EXP3 algo-                             contribute to the end-to-end outcome. Finally, an agent can only
                                         rithm significantly outperforms existing no-regret online learning                          learn the outcome of its chosen action, which is typically referred
                                         algorithms for the traditional multi-armed bandit problem.                                  to as the bandit feedback in the literature.
                                                                                                                                         We note that the traditional multi-armed bandit problem is a
                                         ACM Reference Format:
                                                                                                                                     special case of multi-stage systems when there is only one stage.
                                         I-Hong Hou. 2024. Distributed No-Regret Learning for Multi-Stage Systems
                                                                                                                                     The main challenge of the traditional multi-armed bandit problem
                                         with End-to-End Bandit Feedback. In The Twenty-fifth International Sym-
                                         posium on Theory, Algorithmic Foundations, and Protocol Design for Mobile                   is to balance between learning the outcomes of each possible action
                                         Networks and Mobile Computing (MOBIHOC ’24), October 14–17, 2024, Athens,                   (exploration) and choosing the action with the best historic out-
                                         Greece. ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3641512.                  comes (exploitation). The general problem of multi-stage systems
                                         3686369                                                                                     is even more challenging because agents in the next stage are also
                                                                                                                                     learning agents and their ability to learn depends on actions taken
                                         1     INTRODUCTION                                                                          by the agent in the previous stage. In the example of mobile edge
                                                                                                                                     computing, an edge server can only process a job and learn the out-
                                         In many modern applications, a job consists of multiple stages that
                                                                                                                                     come when it receives a job from the mobile user. When a mobile
                                         need to be performed by different agents, and the decision made
                                                                                                                                     user receives a poor outcome from an edge server, it may be because
                                         in each stage can impact the performance of the job. For example,
                                                                                                                                     the edge server has yet to learn the optimal action and chooses a
                                         consider a mobile edge computing application where a mobile user
                                                                                                                                     bad neural network, rather than because the edge server has no
                                         offloads video analytic jobs to nearby edge servers, and each edge
                                                                                                                                     good options. To ensure that all edge servers can learn the optimal
                                         server is equipped with multiple video analytic neural networks
                                                                                                                                     actions, the mobile user needs to educate edge servers by forward-
                                         with different precision and latency. To process a video analytic
                                                                                                                                     ing a sufficient number of jobs to each of them. Thus, the mobile
                                         job, the mobile user needs to first decide which edge server to
                                                                                                                                     user is facing an exploration-exploitation-education trilemma.
                                         forward this job to. After the mobile user forwards the job to an
                                                                                                                                         To study the online learning problem in multi-stage systems, we
                                         Permission to make digital or hard copies of part or all of this work for personal or       propose an analytical model that captures both the distributed deci-
                                         classroom use is granted without fee provided that copies are not made or distributed       sion making and the end-to-end bandit feedback. We first consider
                                         for profit or commercial advantage and that copies bear this notice and the full citation
                                         on the first page. Copyrights for third-party components of this work must be honored.
                                                                                                                                     the simplified case when each agent can observe the outcomes of all
                                         For all other uses, contact the owner/author(s).                                            its actions, including those not taken. We show that we can achieve
                                         MOBIHOC ’24, October 14–17, 2024, Athens, Greece                                            sublinear regret by making all agents employ the Normalized Ex-
                                         © 2024 Copyright held by the owner/author(s).
                                         ACM ISBN 979-8-4007-0521-2/24/10.
                                                                                                                                     ponential Gradient (normalized-EG) algorithm independently in a
                                         https://doi.org/10.1145/3641512.3686369                                                     distributed fashion.
```


## Page 2

```text
MOBIHOC ’24, October 14–17, 2024, Athens, Greece                                                                                        I-Hong Hou


    Next, we study the multi-stage system with only end-to-end           all agents. Singla, Hassani, and Krause [27] has studied a distributed
bandit feedback, that is, an agent can only observe an outcome if        learning problem in two-stage systems. It is limited to the special
it receives a job and it can only observe the outcome of its chosen      case of two stages and requires the root node to have the ability to
action. To address the exploration-exploitation-education trilemma,      block feedback information.
we propose a simple distributed online learning algorithm called         Mobile edge computing. One emerging application of mobile
𝜖−EXP3. The 𝜖−EXP3 algorithm has two operation modes, a uni-             edge computing is cloud/edge robotics where a robot offloads its
form selection mode in which the agent chooses actions uniformly         computation tasks to nearby edge servers. An important challenge
at random to provide equal education to agents in the next stage,        for cloud/edge robotics is that the performance of a job depends
and an EXP3 mode where the agent employs a variation of the              on both the quality of the outcome and the end-to-end delay. To
EXP3 algorithm to balance the tradeoff between exploration and           enable flexible trade-off between quality and latency, Jiang et al.
exploitation. By randomly alternating between these two modes,           [15] has proposed a controller that dynamically select the suitable
the 𝜖−EXP3 algorithm explicitly address all three of exploration,        neural network configuration. Wu et al. [30] has modeled the prob-
exploitation, and education. We theoretically prove that, when ap-       lem of adaptive configuration as an integer programming problem
plying 𝜖−EXP3 on a system with 𝐿 stages, the regret accumulated          and proposed a heuristic for it. He et al. [14] has employed a rein-
                                𝐿
over 𝑇 rounds is at most 𝑂 (𝑇 𝐿+1 ) = 𝑜 (𝑇 ).                            forcement learning approach for adaptive configuration. Zhang et
    To understand the fundamental regret lower bounds and the            al. [32] has employed Lyapunov optimization to learn the optimal
role of education in multi-stage systems, we study a class of time-      configuration over time. These studies only study the decisions of
homogeneous oracle policies. These policies assume that each node        edge servers and they only consider stationary systems. Chinchali
can know the outcome of all actions before making a decision.            et al. [10] has proposed using deep reinforcement learning for the
Therefore, there is no need to explore and each node only faces an       offloading decisions of robots, but it does not consider the adap-
education-exploitation dilemma. We show that the regret of these         tive configuration of edge servers. To the best of our knowledge,
                         𝐿−1
policies is at least Θ(𝑇 𝐿 ), which is only slightly better than the     no existing work has jointly optimized the offloading decision of
regret of 𝜖−EXP3.                                                        robots and the adaptive configuration of edge servers in unknown
    The utility of the 𝜖−EXP3 algorithm is further evaluated by          and time-varying environments.
simulations. The simulation results show that the regret of the          Multi-hop networks. There have been significant interests in
                                            𝐿                            employing online learning or reinforcement learning techniques
𝜖−EXP3 algorithm indeed scales as 𝑂 (𝑇 𝐿+1 ). We also evaluate two
                                                                         for multi-hop networks, but few of them have been able to charac-
other policies that are no-regret policies for the traditional one-
                                                                         terize end-to-end delay and enforcing end-to-end deadline. Bhorkar
stage bandit problem. Surprisingly, we show that their regrets scale
                                                                         and Javidi [6] has proposed a no-regret learning policy for min-
as 𝜃 (𝑇 ) even when the system has only two stages. The simulation
                                                                         imizing end-to-end transmission cost. Park, Kang, and Joo [24]
results demonstrate that the education component is indeed critical
                                                                         has proposed a UCB-based algorithm for throughput-optimality in
in multi-stage systems.
                                                                         multi-hop wireless networks. Al Islam et al. [2] has considered the
    The rest of the paper is organized as follows. Section 2 surveys
                                                                         problem of end-to-end congestion control problem in multi-hop
existing studies on adversarial bandit problems, mobile edge com-
                                                                         networks as a multi-armed bandit problem. Zhang, Tang, and Wang
puting, and multi-hop networks. Section 3 introduces our system
                                                                         [31] has studied the problem of relay selection to minimize energy
model and problem definition. Section 4 studies the simplified case
                                                                         consumption in two-hop networks. None of the aforementioned
with complete one-hop feedback. Section 5 introduces and analyzes
                                                                         studies consider end-to-end delay or end-to-end deadline.
the 𝜖−EXP3 algorithm for systems with end-to-end bandit feedback.
                                                                             Mao, Koksal, and Shroff [20], Deng, Zhao, and Hou [11], and
Section 6 establishes a regret lower bound for time-homogeneous
                                                                         Gu, Liu, Shen [12] have all studied online scheduling and routing
oracle policies. Section 7 presents our simulation results. Finally,
                                                                         algorithms for multi-hop networks with end-to-end deadlines, but
Section 8 concludes the paper.
                                                                         they require precise knowledge on the capacity and latency of each
                                                                         link. HasanzadeZonuzy, Kalathil, and Shakkottai [13] has proposed
2    RELATED WORK                                                        a model-based reinforcement learning algorithm for real-time multi-
No-regret bandit learning. The multi-armed bandit problem has            hop networks but it only works for stationary systems. Both Lin
attracted significant research interests because it elegantly captures   and van der Schaar [17] and Shiang, and van der Schaar [26] employ
the trade-off between exploration and exploitation. In adversarial       reinforcement learning to serve delay-sensitive traffic by modeling
environments, the celebrated EXP3 algorithm has been proved to           multi-hop networks as stationary MDPs with unknown kernels. To
                                   1
achieve a regret bound of 𝑂 (𝑇 2 ) [5]. This bound has later been        the best of our knowledge, no existing work has studied the regret
shown to be tight [3]. There have been many studies on variations        of delay-sensitive multi-hop networks in adversarial environments.
and improvements of the EXP3 algorithm [4, 22, 28, 29]. All these
studies only consider systems with one agent.
   There have been considerable recent efforts on cooperative learn-     3    SYSTEM MODEL
ing [1, 8, 9, 16, 18, 19, 21, 23] where agents help each other find      We represent a multi-stage system as a tree with depth 𝐿 + 1. We
the optimal action. These studies assume that the reward of an           denote the root node by 𝑟 and the set of leaf nodes by L. We use
agent only depends on the action of that agent. In contrast, our         C𝑖 to denote the set of children of a non-leaf node 𝑖. In each round
work allows different agents to have different sets of actions and       𝑡, the root node 𝑟 receives a job. It selects a child node 𝑓 [𝑟, 𝑡] ∈ C𝑟 ,
considers that the reward in each round depends on the actions of        possibly at random, and forwards the job to it. Likewise, every
```


## Page 3

```text
Distributed No-Regret Learning for Multi-Stage Systems with End-to-End Bandit Feedback                                MOBIHOC ’24, October 14–17, 2024, Athens, Greece


non-leaf node 𝑖 randomly selects a child node 𝑓 [𝑖, 𝑡] ∈ C𝑖 and, if                          Each non-leaf node 𝑖 employs a distributed online policy that
𝑖 receives a job in round 𝑡, forwards the job to 𝑓 [𝑖, 𝑡]. When the                      determines the probability of forwarding a job to a child node 𝑗
job reaches a leaf node 𝑗, it generates a cost of 𝑐 [ 𝑗, 𝑡] ∈ [0, 1]. The                in round 𝑡, denoted by 𝑥 [𝑖, 𝑗, 𝑡] := 𝑃𝑟𝑜𝑏 (𝑓 [𝑖, 𝑡] = 𝑗), in the event
                                                                                                                                               Í
value of 𝑐 [ 𝑗, 𝑡] is revealed to all nodes between the root and the                     that 𝑖 receives a job. We have 𝑥 [𝑖, 𝑗, 𝑡] ≥ 0 and 𝑗 ∈ C𝑖 𝑥 [𝑖, 𝑗, 𝑡] = 1.
leaf node 𝑗 through an end-to-end feedback message.                                      Node 𝑖 needs to determine the values of 𝑥 [𝑖, 𝑗, 𝑡] using only the
   We note that each node only has limited feedback information                          information available up to round 𝑡 − 1.
in this setting. In particular, if a node receives a job in round 𝑡,                         We now characterize the performance of a distributed online
then it will only know its own choice and the final cost. It has                         policy after it determines the values of 𝑥 [𝑖, 𝑗, 𝑡] and selects 𝑓 [𝑖, 𝑡] ac-
neither knowledge nor control on the choices made by its children.                       cordingly. Let 𝑦 [𝑖, 𝑡] be the random variable indicating the amount
This is to reduce coordination overhead and to protect privacy. If a                     of cost that would be incurred if node 𝑖 receives a job in round 𝑡,
node does not receive a job in a round, then it will not receive any                     under the probability distribution of 𝑓 [𝑖, 𝑡]. By definition, we have
feedback information.                                                                    𝑦 [ 𝑗, 𝑡] = 𝑐 [ 𝑗, 𝑡] for each leaf node 𝑗 ∈ L. For each non-leaf node 𝑖,
   To see how our model can be used to capture mobile edge com-                          𝑦 [𝑖, 𝑡] can be calculated   h recursivelyi through 𝑦 [𝑖, 𝑡] = 𝑦 [𝑓 [𝑖, 𝑡], 𝑡].
puting, we can consider the example shown in Fig. 1(a). In this                          Also, let 𝑤 [𝑖, 𝑡] := 𝐸 𝑦 [𝑖, 𝑡] H𝑡 −1 be the conditional expected
system, a robot chooses one of two edge servers to offload its video
                                                                                         amount of cost incurred if node 𝑖 receives a job in round 𝑡, given
analytic jobs. Each edge server has two neural networks to choose
                                                                                         all events up to round 𝑡 − 1, denoted by H𝑡 −1 . The value of 𝑤 [𝑖, 𝑡]
from. This system can be modeled as a tree with 𝐿 = 2 as shown
                                                                                         can then be calculated recursively by 𝑤 [ 𝑗, 𝑡] = 𝑐 [ 𝑗, 𝑡] for each
in Fig. 1(b). In Fig. 1(b), the robot is the root that chooses between                                                  Í
                                                                                         leaf node 𝑗 and 𝑤 [𝑖, 𝑡] = 𝑗 ∈ C𝑖 𝑥 [𝑖, 𝑗, 𝑡]𝑤 [ 𝑗, 𝑡] for each non-leaf
child 𝐴 and child 𝐵. Each of these child nodes corresponds to an
                                                                                         node 𝑖. The total expected cost incurred by the distributed online
edge server, and each child node chooses between two leaf nodes.
                                                                                         policy hover a itime horizon  h of 𝑇i rounds can then be written as
Each leaf node is labeled by 𝑋 : 𝑛, where 𝑋 indicates the edge server                    Í𝑇                   Í𝑇
chosen by the robot, and 𝑛 indicates the neural network chosen                             𝑡 =1 𝐸 𝑦 [𝑟, 𝑡] =     𝑡 =1 𝑤 [𝑟, 𝑡] .
                                                                                                                     𝐸
by the edge server. The cost of a leaf node is chosen to reflect the                         We compare the cost of a distributed online policy against a
delay and the quality of the outcome of the video analytic job.                          stationary policy where each node selects the same child node in
                                                                                         each round 𝑡, i.e., 𝑓 [𝑖, 𝑡] ≡ 𝑓𝑖 , ∀𝑡. Under a stationary policy, all jobs
                                                                                         will reach the same leaf node 𝑗 ∗ , and hence the total cost incurred by
                                 Neural network 1
                                                                                         the stationary policy is 𝑇𝑡=1 𝑐 [ 𝑗 ∗, 𝑡]. The optimal stationary policy
                                                                                                                     Í
                           Server A          Server B
                                                                                         is the one that has a minimum cost among all stationary policies
                                                                                                                    Í
                                                                                         and its cost is min 𝑗 ∈ L 𝑇𝑡=1 𝑐 [ 𝑗, 𝑡]. We therefore define the regret of
                                                                                                                            Í       h      i            Í
                                                                                         a distributed online policy as 𝑇𝑡=1 𝐸 𝑦 [𝑟, 𝑡] − min 𝑗 ∈ L 𝑇𝑡=1 𝑐 [ 𝑗, 𝑡].
                                 Neural network 2                                        Our goal is to design a no-regret policy whose regret is sublinear in
                             (a) System illustration                                     𝑇 under all possible vectors of 𝑐 [ 𝑗, 𝑡]:
                                                                                            Definition 1.h A distributed  online policy is said to be a no-regret
                                              The robot chooses
                                                                                                                  i
                                                                                         policy if 𝑇𝑡=1 𝐸 𝑦 [𝑟, 𝑡] − min 𝑗 ∈ L 𝑇𝑡=1 𝑐 [ 𝑗, 𝑡] = 𝑜 (𝑇 ).
                                                                                                  Í                           Í
                                               one edge server


                                                                                         4    PRELIMINARY: POLICY WITH COMPLETE
                                                                                              ONE-HOP FEEDBACK
                                                                                         In this section, we first study the simplified case where each non-leaf
                     Server A chooses     Server B chooses                               node has complete one-hop feedback from its children. Specifically,
                    one neural network   one neural network
                                                                                         each non-leaf node 𝑖, regardless whether it receives a job or not, will
                                  (b) Tree model                                         be able to learn the values of 𝑦 [ 𝑗, 𝑡] for each of its children 𝑗 ∈ C𝑖
                                                                                         after 𝑖 chooses 𝑓 [𝑖, 𝑡]. Node 𝑖 can then use these values to update
                                                                                         the values of 𝑥 [𝑖, 𝑗, 𝑡 + 1]. We emphasize that the communication
Figure 1: A mobile edge computing system and its tree model
                                                                                         overhead between a child node 𝑗 and its parent node contains
                                                                                         only one single scalar 𝑦 [ 𝑗, 𝑡] in each round. Hence, the feedback
   This model can also be used to capture multi-hop networks. In
                                                                                         information that a non-leaf node has is still limited. For example,
multi-hop networks, the root is the source that generates packets
                                                                                         a non-leaf node has neither knowledge nor control over actions
to be delivered. Each non-leaf node corresponds to the path used to
                                                                                         taken by its children.
transfer a packet to an intermediate router. The choice of that non-
                                                                                            We consider that each non-leaf node 𝑖 independently employs
leaf node corresponds to choosing the next-hop by the intermediate
                                                                                         the Normalized Exponential Gradient (normalized-EG) algorithm, a
router. Each leaf node is a complete path from the source to the
                                                                                         special case of the Online Mirror Descent algorithm and the Follow-
destination and its cost can be chosen to reflect end-to-end delay.
                                                                                         the-Regularized-Leader algorithm. Under the normalized-EG algo-
We note that we do not require the topology of the multi-hop
                                                                                         rithm, each non-leaf node 𝑖 maintains a variable 𝜃 [𝑖, 𝑗, 𝑡] for each
networks to be trees. Even when the topology of a network is not a
                                                                                         𝑗 ∈ C𝑖 by setting 𝜃 [𝑖, 𝑗, 1] = 0 and 𝜃 [𝑖, 𝑗, 𝑡] = 𝜃 [𝑖, 𝑗, 𝑡 −1] −𝑦 [ 𝑗, 𝑡 −1]
tree, the set of all loop-free paths from the source to the destination                                                                    𝜂𝑖 𝜃 [𝑖,𝑗,𝑡 ]
can still be represented as a tree.                                                      for all 𝑡 > 1. It then chooses 𝑥 [𝑖, 𝑗, 𝑡] = Í 𝑒 𝜂𝑖 𝜃 [𝑖,𝑘,𝑡 ] in each round
                                                                                                                                        𝑘 ∈C𝑖 𝑒
```


## Page 4

```text
MOBIHOC ’24, October 14–17, 2024, Athens, Greece                                                                                                                             I-Hong Hou


𝑡, where 𝜂𝑖 is a constant whose value will be determined later. The                                        If each non-leaf node has as most 𝐷 children, then,
                                                                                               Theorem 1. √︃
normalized-EG algorithm is an online policy because 𝜃 [𝑖, 𝑗, 𝑡] can                                                  log | C |
                                                                                           by setting 𝜂𝑖 =       𝑖
                                                                                                                   , ∀𝑖, the expected cost incurred by the root
be calculated only based on 𝑦 [ 𝑗, 1], 𝑦 [ 𝑗, 2], . . . , 𝑦 [ 𝑗, 𝑡 − 1]. A formal                              𝑇
                                                                                           node 𝑟 is upper-bounded by:
description of the normalized-EG algorithm is presented in Alg. 1.
                                                                                                          𝑇
                                                                                                         ∑︁  h         i       𝑇
                                                                                                                              ∑︁               √︁
                                                                                                            𝐸 𝑦𝑛 [𝑟, 𝑡] ≤ min    𝑐 [ 𝑗, 𝑡] + 2𝐿 𝑇 log 𝐷.                            (1)
Algorithm 1 Distributed Normalized Exponential Gradient                                                  𝑡 =1
                                                                                                                                    𝑗∈L
                                                                                                                                          𝑡 =1
    1: 𝜂𝑖 ← a pre-determined constant
                                                                                              Proof. We will prove the theorem by establishing the following
    2: 𝜃 [𝑖, 𝑗] ← 0, ∀𝑗 ∈ C𝑖
                                                                                           statement by induction:   If ai node 𝑖 is (𝐿 − ℎ)-hops from the root
    3: for each round 𝑡 do                                                                              Í      h             Í                 √︁
                         𝜂𝑖 𝜃 [𝑖,𝑗 ]                                                       node 𝑟 , then 𝑇𝑡=1 𝐸 𝑦𝑛 [𝑖, 𝑡] ≤ 𝑇𝑡=1 𝑦∗ [𝑖, 𝑡] + 2ℎ 𝑇 log 𝐷. Please
    4:   𝑥 [𝑖, 𝑗] ← Í 𝑒 𝜂𝑖 𝜃 [𝑖,𝑘 ] , ∀𝑗 ∈ C𝑖
                      𝑘 ∈C𝑖 𝑒                                                              see Appendix A for details.                                        □
    5:   Select a child 𝑓 [𝑖] with 𝑃𝑟𝑜𝑏 (𝑓 [𝑖] = 𝑗) = 𝑥 [𝑖, 𝑗]
    6:for each 𝑗 ∈ C𝑖 do                                                                   5     POLICY WITH END-TO-END BANDIT
    7:   Obtain 𝑦 [ 𝑗] from child 𝑗
 8:      𝜃 [𝑖, 𝑗] ← 𝜃 [𝑖, 𝑗] − 𝑦 [ 𝑗]
                                                                                                 FEEDBACK
 9:   end for                                                                              In this section, we consider the case where each non-leaf node only
10:   𝑦 [𝑖] ← 𝑦 [𝑓 [𝑖]]                                                                    has bandit feedback. Specifically, if a node does not receive a job in
11:   Report 𝑦 [𝑖] to the parent node                                                      round 𝑡, then it will not get any feedback. If a node receives a job
12: end for                                                                                and forwards it to a child node 𝑗 = 𝑓 [𝑖, 𝑡], then it will only learn the
                                                                                           value of 𝑦 [ 𝑗, 𝑡]. As discussed in earlier sections, online policies with
                                                                                           end-to-end bandit feedback faces a trilemma between exploration,
   The regret of the normalized-EG algorithm has been extensively                          i.e., choosing a child to learn its cost, exploitation, i.e., choosing a
studied for the special case when 𝐿 = 1. We will further show that                         child to incur low cost, and education, i.e., choosing a child so that
the normalized-EG algorithm is a no-regret policy for the general                          it has a chance to learn and improve its policy.
case 𝐿 > 1. It is important to note that the values of 𝑦 [ 𝑗, 𝑡] observed                      We propose a simple distributed online learning policy to ad-
by 𝑖 under the normalized-EG algorithm can be different from those                         dress the exploration-exploitation-education trilemma called the
under the optimal stationary policy. This is because the values                            𝜖−EXP3 algorithm. Under the 𝜖−EXP3 algorithm, each non-leaf
of 𝑦 [ 𝑗, 𝑡] depend on the decisions made by children nodes 𝑗. To                          node 𝑖 maintains a variable 𝜃 [𝑖, 𝑗, 𝑡] for each 𝑗 ∈ C𝑖 , which it will
distinguish between these two policies, we let 𝑦𝑛 [ 𝑗, 𝑡] be the values                    use to determine 𝑥 [𝑖, 𝑗, 𝑡]. When a node 𝑖 sends a job to a child node
of 𝑦 [ 𝑗, 𝑡] under the normalized-EG algorithm and let 𝑦∗ [ 𝑗, 𝑡] be                       𝑗 = 𝑓 [𝑖, 𝑡], node 𝑖 also includes a variable 𝑣 [ 𝑗, 𝑡] indicating the prob-
those under the optimal stationary policy.                                                 ability that the child node 𝑗 receives a job in round 𝑡. Since a node 𝑗
   Since the normalized-EG algorithm is updated with respect to                            will receive a job if its parent node 𝑖 receives a job and node 𝑖 chooses
𝑦𝑛 [ 𝑗, 𝑡], we let Y𝑛 [𝑖, 𝑡] := {𝑦𝑛 [ 𝑗, 𝜏], ∀𝑗 ∈ C𝑖 , 𝜏 ∈ [1, 𝑡]} be the                  𝑗, the value of 𝑣 [ 𝑗, 𝑡] can be calculated by 𝑣 [ 𝑗, 𝑡] = 𝑣 [𝑖, 𝑡]𝑥 [𝑖, 𝑗, 𝑡].
sequences of costs of all children of 𝑖 up to round 𝑡 and have the                             We now discuss how a non-leaf node 𝑖 decides 𝑓 [𝑖, 𝑡] in each
following from existing studies:                                                           round 𝑡. There are two modes for choosing 𝑓 [𝑖, 𝑡] and node 𝑖 ran-
   Lemma 1 ([25], Theorem 2.22). If 𝑦𝑛 [ 𝑗, 𝜏] ≥ 0 for all 𝑗 ∈ C𝑖 and                      domly decides which mode to operate in in each 𝑡. Each node 𝑖 is
𝜏 ∈ [1,𝑇 ], then the expected total cost incurred by 𝑖 given Y𝑛 [𝑖] is                     assigned two pre-determined constants 𝜖𝑖 and 𝜂𝑖 . With probability
upper-bounded by:                                                                          𝜖𝑖 , node 𝑖 is in the uniform selection mode and it chooses 𝑓 [𝑖, 𝑡]
                                                                                           uniformly at random from its children, that is, 𝑃𝑟𝑜𝑏 (𝑓 [𝑖, 𝑡] = 𝑗) =
          𝑇                                   𝑇
                                                               log |C𝑖 |                   1/|C𝑖 |, ∀𝑗 ∈ C𝑖 . With probability 1 − 𝜖𝑖 , node 𝑖 is in the EXP3 mode
         ∑︁    h                   i         ∑︁
              𝐸 𝑦𝑛 [𝑖, 𝑡] Y𝑛 [𝑖, 𝑡] ≤ min         𝑦𝑛 [ 𝑗, 𝑡] +                                                                                     𝜂𝑖 𝜃 [𝑖,𝑗,𝑡 ]
         𝑡 =1
                                      𝑗 ∈ C𝑖
                                             𝑡 =1
                                                                  𝜂𝑖                       and it chooses 𝑓 [𝑖, 𝑡] = 𝑗 with probability Í 𝑒 𝜂𝑖 𝜃 [𝑖,𝑘,𝑡 ] . We use
                                                                                                                                                              𝑘 ∈C𝑖 𝑒
                                                  𝑇 ∑︁
                                                 ∑︁                                        𝑚[𝑖, 𝑡] ∈ {𝑈 , 𝐸} to denote the mode of node 𝑖, where 𝑈 is the uni-
                                        + 𝜂𝑖                   𝑥 [𝑖, 𝑗, 𝑡]𝑦𝑛 [ 𝑗, 𝑡] 2 .   form selection mode and 𝐸 is the EXP3 mode. Combining these two
                                                 𝑡 =1 𝑗 ∈ C𝑖                                                                                        𝜂𝑖 𝜃 [𝑖,𝑗,𝑡 ]
                                                                                           modes and we have 𝑥 [𝑖, 𝑗, 𝑡] = 𝜖𝑖 | C1 | + (1 − 𝜖𝑖 ) Í 𝑒 𝜂𝑖 𝜃 [𝑖,𝑘,𝑡 ] .
                                                                                                                                                                   𝑘 ∈C𝑖 𝑒
Moreover, if 𝑦𝑛 [ 𝑗, 𝜏] ∈ [0, 1], ∀𝑗 ∈ C𝑖 , 𝜏 ∈ [1,𝑇 ], then setting 𝜂𝑖 =
                                                                                                                                                  𝑖
√︃                                                                                             After choosing 𝑓 [𝑖, 𝑡] for each node 𝑖, we can set 𝑦𝜖 [𝑖, 𝑡] = 𝑐 [𝑖, 𝑡]
   log | C𝑖 |
      𝑇       yields:                                                                      for each leaf node and set 𝑦𝜖 [𝑖, 𝑡] = 𝑦𝜖 [𝑓 [𝑖, 𝑡], 𝑡] for each non-leaf
                                                                                           node, where the subscript 𝜖 is to highlight that this corresponds
          𝑇                              𝑇
                                                                                           to the values of 𝑦 [ 𝑗, 𝑡] under the 𝜖−EXP3 algorithm. We note that,
         ∑︁  h                   i      ∑︁               √︁
            𝐸 𝑦𝑛 [𝑖, 𝑡] Y𝑛 [𝑖, 𝑡] ≤ min    𝑦𝑛 [ 𝑗, 𝑡] + 2 𝑇 log |C𝑖 |.
                                        𝑗 ∈ C𝑖                                             even if node 𝑖 does not receive a job in round 𝑡, the value of 𝑦𝜖 [𝑖, 𝑡]
         𝑡 =1                                    𝑡 =1
                                                                                           is still well-defined, but node 𝑖 does not know its value.
□
                                                                                               Finally, we discuss how node 𝑖 determines 𝜃 [𝑖, 𝑗, 𝑡]. Node 𝑖 ini-
   Under the optimal stationary policy, each node will choose to                           tializes 𝜃 [𝑖, 𝑗, 1] = 0 for all children 𝑗. If node 𝑖 receives a job
forward the job to the child that incurs the least cost through all 𝑇                      in round 𝑡, then it learns the value of 𝑦𝜖 [𝑓 [𝑖, 𝑡], 𝑡]. Node 𝑖 sets
                         Í                          Í
rounds. Hence, we have 𝑇𝑡=1 𝑦∗ [𝑖, 𝑡] = min 𝑗 ∈ C𝑖 𝑇𝑡=1 𝑦∗ [ 𝑗, 𝑡]. We                                       𝑦 [ 𝑓 [𝑖,𝑡 ],𝑡 ] | C |
                                                                                           𝑧 [𝑓 [𝑖, 𝑡], 𝑡] = 𝜖 𝑣 [𝑖,𝑡 ] 𝑖 , if 𝑚[𝑖, 𝑡] = 𝑈 , and sets 𝑧 [𝑓 [𝑖, 𝑡], 𝑡] =
now prove that the normalized-EG algorithm is still a no-regret                            𝑦𝜖 [ 𝑓 [𝑖,𝑡 ],𝑡 ] 𝑘 ∈C𝑖 𝑒 𝜂𝑖 𝜃 [𝑖,𝑘,𝑡 ]
                                                                                                            Í
policy for the multi-stage system:                                                                                                 , if 𝑚[𝑖, 𝑡]       = 𝐸. Node 𝑖 sets 𝑧 [ 𝑗, 𝑡] = 0
                                                                                                    𝑣 [𝑖,𝑡 ]𝑒 𝜂𝑖 𝜃 [𝑖,𝑗,𝑡 ]
```


## Page 5

```text
Distributed No-Regret Learning for Multi-Stage Systems with End-to-End Bandit Feedback                                     MOBIHOC ’24, October 14–17, 2024, Athens, Greece


for all 𝑗 ≠ 𝑓 [𝑖, 𝑡]. On the other hand, if node 𝑖 does not receive                         Lemma 2. For any non-leaf node 𝑖,
a job in round 𝑡, then it sets 𝑧 [ 𝑗, 𝑡] = 0, ∀𝑗 ∈ C𝑖 . Finally, it sets                              h                                 i
𝜃 [𝑖, 𝑗, 𝑡 + 1] = 𝜃 [𝑖, 𝑗, 𝑡] − 𝑧 [ 𝑗, 𝑡], ∀𝑗 ∈ C𝑖 .                                                 𝐸 𝑧 [ 𝑗, 𝑡] Y𝜖 [𝑖, 𝑡], Z [𝑖, 𝑡 − 1] = 𝑦𝜖 [ 𝑗, 𝑡],                   (2)
    Alg. 2 describes the 𝜖−EXP3 algorithm in detail, where we stream-                    and
line some of the steps for easier implementation.                                                      h                                      i
                                                                                                     𝐸 𝑧 [ 𝑗, 𝑡] 2 Y𝜖 [𝑖, 𝑡], Z [𝑖, 𝑡 − 1]

Algorithm 2 𝜖-EXP3
                                                                                                                            Í          𝜂𝑖 𝜃 [𝑖,𝑘,𝑡 ] 
                                                                                                     
                                                                                                                              𝑘 ∈ C𝑖 𝑒                 𝑦𝜖 [ 𝑗, 𝑡] 2
                                                                                                    = 𝜖𝑖 |C𝑖 | + (1 − 𝜖𝑖 )                                          .    (3)
 1: 𝜂𝑖 , 𝜖𝑖 ← pre-determined constants                                                                                          𝑒 𝜂𝑖 𝜃 [𝑖,𝑗,𝑡 ]         𝑣 [𝑖, 𝑡]
 2: 𝜃 [𝑖, 𝑗] ← 0, ∀𝑗 ∈ C𝑖
                                                                                            Proof. Please see Appendix B.                                                 □
 3: for each round 𝑡 do
 4:    if Node 𝑖 receives a job and 𝑣 [𝑖, 𝑡] from its parent then                           Next, we show that, if node 𝑖 is in the EXP3 mode at round 𝑡,
                                              𝜂𝑖 𝜃 [𝑖,𝑗 ]
 5:       𝑥 [𝑖, 𝑗] ← 𝜖𝑖 | C1 | + (1 − 𝜖𝑖 ) Í 𝑒 𝜂𝑖 𝜃 [𝑖,𝑘 ] , ∀𝑗 ∈ C𝑖                     then its expected cost is the same as the expected cost of running
                           𝑖                𝑘 ∈C𝑖 𝑒
                                                                                         the normalized-EG algorithm against the sequence 𝑧 [ 𝑗, 𝑡].
 6:       𝑣 [ 𝑗, 𝑡] ← 𝑣 [𝑖, 𝑡]𝑥 [𝑖, 𝑗], ∀𝑗 ∈ C𝑖
 7:       Randomly select 𝑚[𝑖] ∈ {𝑈 , 𝐸} with 𝑃𝑟𝑜𝑏 (𝑚[𝑖] = 𝑈 ) = 𝜖𝑖                          Lemma 3. By considering a sequence 𝑦𝑛 [ 𝑗, 𝜏] = 𝑧 [ 𝑗, 𝜏], ∀𝑗 ∈
                                                                                         C𝑖 , 𝜏 ∈ [1,𝑇 ] for the normalized-EG algorithm,
 8:       if 𝑚[𝑖] = 𝑈 then                                                                                 h                                              i
 9:          Select a child 𝑓 [𝑖] ∈ C𝑖 uniformly at random                                               𝐸 𝑦𝜖 [𝑖, 𝑡] 𝑚[𝑖, 𝑡] = 𝐸, Y𝜖 [𝑖, 𝑡], Z [𝑖, 𝑡 − 1]
10:          Forward the job and 𝑣 [𝑓 [𝑖], 𝑡] to child 𝑓 [𝑖] and obtain                                    h h                               ii
             𝑦𝜖 [𝑓 [𝑖]] from 𝑓 [𝑖]                                                                    =𝐸 𝐸 𝑦𝑛 [𝑖, 𝑡] Y𝑛 [𝑖, 𝑡] = Z [𝑖, 𝑡] ,
                                          𝑦 [ 𝑓 [𝑖 ] ] | C |
11:          𝜃 [𝑖, 𝑓 [𝑖]] ← 𝜃 [𝑖, 𝑓 [𝑖]] − 𝜖 𝑣 [𝑖,𝑡 ] 𝑖                                  where the outer expectation on the right hand side is taken with respect
12:          Return 𝑦𝜖 [𝑖] ← 𝑦𝜖 [𝑓 [𝑖]] to the parent                                    to 𝑧 [ 𝑗, 𝑡].
13:       else
14:          Select a child 𝑓 [𝑖] ∈ C𝑖 with 𝑃𝑟𝑜𝑏 (𝑓 [𝑖] = 𝑗) =                              Proof. Please see Appendix C.                                    □
                  𝜂𝑖 𝜃 [𝑖,𝑗 ]
              Í 𝑒 𝜂 𝜃 [𝑖,𝑘 ]                                                                                                                  Í     h         i
               𝑘 ∈C𝑖 𝑒 𝑖                                                                    Our next step is to bound the difference between 𝑇𝑡=1 𝐸 𝑦𝜖 [𝑖, 𝑡]
15:          Forward the job and 𝑣 [𝑓 [𝑖], 𝑡] to child 𝑓 [𝑖] and obtain                                       Í
                                                                                         and min 𝑗 ∈ C𝑖 𝑇𝑡=1 𝑦𝜖 [ 𝑗, 𝑡] under any given given sequence of
             𝑦𝜖 [𝑓 [𝑖]] from 𝑓 [𝑖]
                                           𝑦𝜖 [ 𝑓 [𝑖 ] ]
                                                           Í        𝜂𝑖 𝜃 [𝑖,𝑘 ]          𝑦𝜖 [ 𝑗, 1], . . . , 𝑦𝜖 [ 𝑗,𝑇 ], for all 𝑗 ∈ C𝑖 .
                                                           𝑘 ∈C 𝑒
16:       𝜃 [𝑖, 𝑓 [𝑖]] ← 𝜃 [𝑖, 𝑓 [𝑖]] −                 𝑖
                                          𝑣 [𝑖,𝑡 ]𝑒 𝜂𝑖 𝜃 [𝑖,𝑗 ]                             Lemma 4. If each non-leaf node has at most 𝐷 children, then
17:       Return 𝑦𝜖 [𝑖] ← 𝑦𝜖 [𝑓 [𝑖]] to the parent
18:     end if                                                                                         𝑇
                                                                                                      ∑︁  h                    i
19:   end if                                                                                             𝐸 𝑦𝜖 [𝑖, 𝑡] Y𝜖 [𝑖, 𝑡]
                                                                                                      𝑡 =1
20: end for
                                                                                                                𝑇                                       𝑇
                                                                                                               ∑︁                           log 𝐷      ∑︁      𝐷
                                                                                                   ≤ min              𝑦𝜖 [ 𝑗, 𝑡] + 𝜖𝑖 𝑇 +         + 𝜂𝑖               ,
                                                                                                      𝑗 ∈ C𝑖
                                                                                                               𝑡 =1
                                                                                                                                              𝜂𝑖       𝑡 =1
                                                                                                                                                            𝑣 [𝑖, 𝑡]
    Remark 1. The reason that the 𝜖−EXP3 algorithm has two different                     for all non-leaf node 𝑖. Moreover, if the depth of the tree is 𝐿 + 1, then
modes to choose 𝑓 [𝑖, 𝑡] is to address the exploration-exploitation-                                        𝐿
education trilemma. When node 𝑖 is in the uniform selection mode, its                    setting 𝜂𝑖 = 𝑇 − 𝐿+1 for all 𝑖 and setting 𝜖𝑖 to be 0 if 𝐶𝑖 ⊂ L, and
                                                                                                1
goal is to provide equal education to all its children. Hence, it selects                𝐷𝑇 − 𝐿+1 otherwise yields
𝑓 [𝑖, 𝑡] uniformly at random so that each child node has the same                                          𝑇
                                                                                                          ∑︁  h                   i       𝑇
                                                                                                                                         ∑︁
chance of receiving a job and learning from its outcome. When node                                           𝐸 𝑦𝜖 [𝑖, 𝑡] Y𝜖 [𝑖, 𝑡] − min    𝑦𝜖 [ 𝑗, 𝑡]
                                                                                                                                               𝑗 ∈ C𝑖
𝑖 is in the EXP3 mode, its goal is to balance the trade-off between                                       𝑡 =1                                          𝑡 =1
exploration and exploitation. Hence, it employs a very similar way
                                                                                                          (                        𝐿
                                                                                                               (𝐷 + log 𝐷)𝑇       𝐿+1   ,   if 𝐶𝑖 ⊂ L,
of choosing 𝑓 [𝑖, 𝑡] as the EXP3 algorithm. The value of 𝜖𝑖 determines                                ≤                             𝐿
the portion of time that node 𝑖 dedicate to education. On the other                                            (2𝐷 + log 𝐷)𝑇 𝐿+1            else.
hand, the value of 𝜂𝑖 determines the trade-off between exploration                          Proof. Please see Appendix D.                                                 □
and exploitation when node 𝑖 is in the EXP3 mode, where larger 𝜂𝑖
means more emphasis on exploitation. The values of 𝜖𝑖 and 𝜂𝑖 will be                        Remark 2. An explanation for the choice of 𝜖𝑖 is in order. We set
determined later.                                                                        𝜖𝑖 = 0 if all children of node 𝑖 are leaf nodes. Since leaf nodes do not
                                                                                         have any children to choose from, they have nothing to learn and do
    We now analyze the regret of 𝜖−EXP3. Our first step is to es-
                                                                                         not need education. Hence, node 𝑖 can operate exclusively in the EXP3
tablish some properties of 𝑧 [ 𝑗, 𝑡]. We let Y𝜖 [𝑖, 𝑡] := {𝑦𝜖 [ 𝑗, 𝜏], ∀𝑗 ∈
                                                                                         mode. On the other hand, if node 𝑖 has some children that are non-leaf
C𝑖 , 𝜏 ∈ [1, 𝑡]} be the sequences of costs of all children of 𝑖 up to
                                                                                         nodes, then node 𝑖 needs to educate these children. Hence, it operates
round 𝑡 and let Z [𝑖, 𝑡] := {𝑧 [ 𝑗, 𝜏], ∀𝑗 ∈ C𝑖 , 𝜏 ∈ [1, 𝑡]} be all the
                                                                                         in the uniform selection mode with a constant probability.
values of 𝑧 [ 𝑗, 𝜏] that has been observed by 𝑖 up to round 𝑡. We then
have the following:                                                                         We will now prove that the 𝜖−EXP3 policy is a no-regret policy.
```


## Page 6

```text
MOBIHOC ’24, October 14–17, 2024, Athens, Greece                                                                                               I-Hong Hou


   Theorem 2. If the depth of the tree is 𝐿 + 1 and each non-leaf               Algorithm 3 Anytime 𝜖-EXP3
node 𝑖 has at most 𝐷 children, then, by using the same settings of 𝜂𝑖               1: for m = 0, 1, 2, . . . do
and 𝜖𝑖 as in Lemma 4, the regret of 𝜖-EXP3 is at most ((2𝐿 − 1)𝐷 +                  2:   Set 𝜖𝑖 and 𝜂𝑖 according to Theorem 2, but replace 𝑇 with 2𝑚
           𝐿
𝐿 log 𝐷)𝑇 𝐿+1 = 𝑜 (𝑇 ).                                                             3:   Run Algorithm 2 on the 2𝑚 rounds 𝑡 = 2𝑚 , 2𝑚 +1, . . . , 2𝑚+1 −1
                                                                                    4: end for
     Proof. We will prove the theorem by establishing the following
statement:h If a inode     𝑖 is (𝐿 − ℎ)-hops from the root node 𝑟 , then                                                                     2𝐿
                                                                                     Theorem 3. The regret of Algorithm 3 is at most 2𝐿𝐿+1 ((2𝐿 −
Í𝑇                       Í𝑇                                        𝐿
   𝑡 =1 𝐸 𝑦𝑛 [𝑖, 𝑡] ≤ 𝑡 =1 𝑦 ∗ [𝑖, 𝑡] + ((2ℎ − 1)𝐷 +ℎ log 𝐷)𝑇
                                                                  𝐿+1 , where
                                                                                                                                          2 𝐿+1 −1
𝑦∗ [𝑖, 𝑡] is the cost under the optimal stationary policy.                      1)𝐷 + 𝐿 log 𝐷)𝑇 𝐿+1 .
                                                                                                     𝐿

    We prove the statement by induction. First, consider the case
ℎ = 1, that is, the node 𝑖 is (𝐿 − 1)-hops from 𝑟 . Since the tree has            Proof. The proof is very similar to that in [25, Section 2.3.1],
depth 𝐿 + 1, either 𝑖 is a leaf node or all children of 𝑖 are leaf nodes.       and is hence omitted.                                           □
If 𝑖 is a leaf node, then 𝑦𝑛 [𝑖, 𝑡] = 𝑦∗ [𝑖, 𝑡] = 𝑐 [𝑖, 𝑡] ∈ [0, 1] and the
statement holds. If all children of 𝑖 are leaf nodes, then we have              6        REGRET LOWER BOUND AND THE NEED
𝑦𝑛 [ 𝑗, 𝑡] = 𝑦∗ [ 𝑗, 𝑡] = 𝑐 [ 𝑗, 𝑡] for all 𝑗 ∈ C𝑖 . Hence, by Lemma 4,                  FOR EDUCATION
                                                                                                                                               𝐿−1
                     𝑇
                    ∑︁  h            𝑇
                                  i ∑︁  h                    i                  In this section, we establish a regret lower bound of Ω(𝑇 𝐿 ) for a
                       𝐸 𝑦𝜖 [𝑖, 𝑡] =   𝐸 𝑦𝜖 [𝑖, 𝑡] Y𝜖 [𝑖, 𝑡]                    class of time-homogeneous oracle policies. Under this class of policies,
                    𝑡 =1                       𝑡 =1                             each node knows the outcomes of each non-leaf child, 𝑦 [𝑖, 𝑡], before
                              𝑇
                             ∑︁                                                 selecting a child to forward a job to. Since the outcomes of each
                                                               𝐿
                ≤ min               𝑦𝜖 [ 𝑗, 𝑡] + (𝐷 + log 𝐷)𝑇 𝐿+1               non-leaf child is known in advance, there is no need for exploration
                    𝑗 ∈ C𝑖
                             𝑡 =1                                               and each node only faces an education-exploitation dilemma. As
                     𝑇
                    ∑︁                                                          we establish a regret lower bound for this class of policies, we also
                                                         𝐿
                =          𝑦∗ [𝑖, 𝑡] + (𝐷 + log 𝐷)𝑇 𝐿+1 ,                       establish the need for education.
                    𝑡 =1
and the statement holds.
   We now assume that the statement holds when ℎ = 𝑔 and
                                                                                                                   …
consider a node 𝑖 that is (𝐿 − (𝑔 + 1))-hops from 𝑟 . Either 𝑖 is
a leaf node or all children of 𝑖 are (𝐿 − 𝑔)-hops from 𝑟 . If 𝑖 is a
leaf node, then the statement clearly holds. If 𝑖 is notha leaf node,i
                                                                                                                   …
                                                      Í
then, by the induction hypothesis, we have 𝑇𝑡=1 𝐸 𝑦𝜖 [ 𝑗, 𝑡] ≤
Í𝑇                                           𝐿
  𝑡 =1 𝑦 ∗ [ 𝑗, 𝑡] + ((2𝑔 − 1)𝐷 + 𝑔 log 𝐷)𝑇
                                            𝐿+1 , for all 𝑗 ∈ C𝑖 . We can
                                                                                Figure 2: System illustration for establishing a lower bound
then use Lemma 4 to establish the following:
                𝑇
               ∑︁  h            𝑇
                             i ∑︁  h h                    ii                        We consider a system with depth 𝐿 + 1 as shown in Fig. 2. There
                  𝐸 𝑦𝜖 [𝑖, 𝑡] =   𝐸 𝐸 𝑦𝜖 [𝑖, 𝑡] Y𝜖 [𝑖, 𝑡]                       are 𝐿 non-leaf nodes, numbered as 1, 2, . . . , 𝐿, and 𝐿 + 1 leaf nodes,
               𝑡 =1                       𝑡 =1                                  numbered as 𝐿 + 1, 𝐿 + 2, . . . , 2𝐿 + 1. Each non-leaf node 𝑖 has two
                h             𝑇
                             ∑︁               i                                 children. For each 𝑖 ≤ 𝐿 − 1, one child of node 𝑖 is the leaf node
                                                                𝐿
           ≤𝐸 min                   𝑦𝜖 [ 𝑗, 𝑡] + (2𝐷 + log 𝐷)𝑇 𝐿+1              𝐿 + 𝑖 and the other child is the non-leaf node 𝑖 + 1. For node 𝐿, both
                    𝑗 ∈ C𝑖
                             𝑡 =1
                                                                                children, node 2𝐿 and node 2𝐿 + 1, are leaf nodes. When a leaf node
                         𝑇
                        ∑︁                                           𝐿          𝑗 received a job, it generates a cost of 1 with probability 𝑝 𝑗 and a
           ≤ min               𝑦∗ [ 𝑗, 𝑡] + ((2𝑔 − 1)𝐷 + 𝑔 log 𝐷)𝑇 𝐿+1          cost of 0 with probability 1 − 𝑝 𝑗 . Given a small positive constant
               𝑗 ∈ C𝑖
                        𝑡 =1
                                           𝐿                                    𝛿 < 1/2𝐿 , we set one of 𝑝 2𝐿 and 𝑝 2𝐿+1 to be (1 − 2𝐿 𝛿)/2 and the
               + (2𝐷 + log 𝐷)𝑇 𝐿+1                                              other to be (1 + 2𝐿 𝛿)/2, and then set 𝑝 𝑗 = (1 − (2𝐿 − 2 𝑗 −𝐿−1 )𝛿)/2
                𝑇
               ∑︁                                                               for all the other leaf nodes 𝑗 = 𝐿 + 1, 𝐿 + 2, . . . , 2𝐿 − 1. Hence, we
                                                                     𝐿
           =          𝑦∗ [𝑖, 𝑡] + ((2𝑔 + 1)𝐷 + (𝑔 + 1) log 𝐷)𝑇 𝐿+1 ,            have (1 − 2𝐿 𝛿)/2                                              𝐿
                                                                                                           < 𝑝 𝐿+2 𝐿< · · · < 𝑝 2𝐿−1 < (1 − 2 𝛿)/2 and
                                                                                                   < 𝑝 𝐿+1
                                                                                          Í𝑇
               𝑡 =1                                                             min 𝑗 ∈ L 𝑡 =1 𝐸 𝑐 [ 𝑗, 𝑡] = (1 − 2 𝛿)𝑇 /2. The regret of the system
and the statement holds. By induction, the statement holds for all              is
ℎ.                                                                                 𝑇
                                                                                  ∑︁                                  𝑇 
                                                                                                                     ∑︁                             
                                                                                                                                  
                                                  Í
   Since the root node 𝑟 is 0-hop from itself and 𝑇𝑡=1 𝑦∗ [𝑟, 𝑡] =                    𝐸 𝑦 [1, 𝑡] − (1 − 2𝐿 𝛿)𝑇 /2 =        𝐸 𝑦 [1, 𝑡] − (1 − 2𝐿 𝛿)/2 . (4)
         Í𝑇                                                                      𝑡 =1                                𝑡 =1
min 𝑗 ∈ L 𝑡 =1 𝑐 𝑗,𝑡 , the theorem holds.                        □
                                                                                   We now discuss the policies employed by each non-leaf node.
  Finally, we note that the 𝜖−EXP3 algorithm requires the knowl-                Since both children of node 𝐿 are leaf nodes, node 𝐿 does not need
edge of 𝑇 to set 𝜖𝑖 and 𝜂𝑖 . When 𝑇 is not known in advance, we                 to consider education. We consider that node 𝐿 can run an arbitrary
can employ the doubling trick to design an anytime algorithm as                 online learning algorithm with bandit feedback. For all other non-
shown in Algorithm 3. This anytime algorithm is also a no-regret                leaf nodes 𝑖 = 1, 2, . . . , 𝐿 − 1, we assume that they employ a time-
policy:                                                                         homogeneous oracle policy defined as follows:
```


## Page 7

```text
Distributed No-Regret Learning for Multi-Stage Systems with End-to-End Bandit Feedback                                         MOBIHOC ’24, October 14–17, 2024, Athens, Greece


   Definition 2. Let 𝑖 1 and 𝑖 2 be the two children of node 𝑖, then a                   Therefore, at any time 𝑡 < 𝑇𝛿 , we have
time-homogeneous oracle policy is one that chooses a child to forward                                              
                                                                                                          𝐸 𝑦 [1, 𝑡] − (1 − 2𝐿 𝛿)/2 ≥ 𝛿/2.                                  (5)
a job to at time 𝑡 with the following assumptions:
     • A1: Node 𝑖 can obtain the expected cost of each child, 𝐸 𝑦 [𝑖 1, 𝑡]                   Moreover, since node 𝐿 can only receive a job if, for each 𝑖 ≤ 𝐿−1,
                                                                          

        and 𝐸 𝑦 [𝑖 2, 𝑡] , before making the forwarding decision.                        node 𝑖 selects node 𝑖 + 1, which happens with probability at most
                        

     • A2: Node 𝑖 makes its forwarding decision solely based on                          𝑞𝑖 , we have
                                                                                                                               𝑁𝛿
        𝐸 𝑦 [𝑖 1, 𝑡] − 𝐸 𝑦 [𝑖 2, 𝑡] . Specifically, let 𝜁 := 𝐸 𝑦 [𝑖 1, 𝑡] −
                                                                                                              
                                                                                                                  𝐸 𝑇𝛿 ] ≥ Î𝐿−1      .                       (6)
                                                                                                                              𝑖=1 𝑞𝑖
        𝐸 𝑦 [𝑖 2, 𝑡] , then node 𝑖 will forward the job to the child with
                   
                                                                                            Next, we
                                                                                                    analyze the system behavior after time 𝑇𝛿 . For any time
        the higher expected cost with probability 𝑃𝑖 (𝜁 ), and to the other              𝑡 > 𝑇𝛿 , 𝐸 𝑦 [𝐿, 𝑡] ≥ min{𝑝 2𝐿 , 𝑝 2𝐿+1 } = (1 − 2𝐿 𝛿)/2. Consider the
        child with probability 1 − 𝑃𝑖 (𝜁 ), where 𝑃𝑖 (·) is an arbitrary                                                                               
                                                                                         forwarding decision of node 𝐿 − 1. Since 𝐸 𝑦 [2𝐿 − 1, 𝑡] = 𝑝 2𝐿−1 =
        decreasing function chosen by node 𝑖.                                                                                    
                                                                                         (1 − (2𝐿 − 2𝐿−2 )𝛿)/2 ≤ 𝐸 𝑦 [𝐿, 𝑡] + 2𝐿−3𝛿, the probability that
   We note that A1 provides a node with much more information                            node 𝐿 − 1 selects node 𝐿 is at most 1 − 𝑃𝐿 (2𝐿−3𝛿) = 1 − 𝑞𝐿−1 .
than is possible in multi-stage systems with bandit feedback, where                      Using a simple induction argument, we can further show that the
a node can only obtain the cost of a child if it forwards a job to                       probability that node 𝑖 selects node 𝑖 + 1 is at most 1 − 𝑞𝑖 , for all
the child, and only after it makes the forwarding decision. Thus,                        𝑖 ≤ 𝐿 − 1. Hence, the probability that node 𝐿 receives a job is at
                                                                                                Î𝐿−1
intuitively, the regret of policies with A1 serves as a natural lower                    most 𝑖=1     (1 −𝑞𝑖 ). If node 𝐿 does not receive a job, which happens
                                                                                                                            Î𝐿−1
bound for the regret of policies with end-to-end bandit feedback.                        with probability at least 1 − 𝑖=1        (1 − 𝑞𝑖 ), then the expected cost
The purpose of A2 is to highlight that a node 𝑖 only knows the                           is at least min 𝑗 ∈ {𝐿+1,𝐿+2,...,2𝐿−1} 𝑝 𝑗 = (1 − (2𝐿 − 1)𝛿)/2. We then
expected costs, but not the internal variables of its children.                          have, at any time 𝑡 > 𝑇𝛿 ,
   We also note that policies with A1 do not need to explore, since it
                                                                                                                                    𝐿−1
knows the expected costs of all children in advance. Hence, policies                                                              Ö            
with A1 only face an education-exploitation dilemma. The only                                       𝐸 𝑦 [1, 𝑡] − (1 − 2𝐿 𝛿)/2 ≥ 1 −     (1 − 𝑞𝑖 ) 𝛿/2.                      (7)
                                                                                                                                                      𝑖=1
reason that a policy may select a child with a higher expected cost,
by choosing 𝑃𝑖 (𝜂) > 0, is to educate its children.                                        Combining Eq. (5), (6), and (7) and we have the following regret
   We first establish a bound for the expected cost of node 𝐿, whose                     bound
children are both leaf nodes. Let 𝑁𝐿 (𝑡) be the number of times that                                  𝑇 
                                                                                                     ∑︁                           
node 𝐿 has received a job from its parent at time 𝑡. Since node 𝐿                                         𝐸 𝑦 [1, 𝑡] − (1 − 2𝐿 𝛿)/2
can only learn the costs of its children when it receives a job, node                                 𝑡 =1
𝐿 cannot determine which of its two children has the smaller 𝑝 𝑗                                          𝑇𝛿
                                                                                                         ∑︁                    𝑇
                                                                                                                              ∑︁            𝐿−1
                                                                                                                                            Ö
                                                                                                                                                              
when 𝑁𝐿 (𝑡) is small. The following lemma formalizes this intuition.                                ≥𝐸              𝛿/2 +              1−         (1 − 𝑞𝑖 ) 𝛿/2
                                                                                                             𝑡 =1           𝑡 =𝑇𝛿 +1        𝑖=1
  Lemma 5. There exists a positive integer 𝑁𝛿 such that, for all 𝑡
                                                                                                                                    𝐿−1
with 𝑁𝐿 (𝑡) < 𝑁𝛿 , 𝐸 𝑦 [𝐿, 𝑡] > (1 − (2𝐿 − 2𝐿−1 )𝛿)/2.                                               𝛿   𝑁𝛿             𝑁𝛿          Ö            
                                                                                                    ≥ [ Î𝐿−1    + (𝑇 − Î𝐿−1    ) 1−     (1 − 𝑞𝑖 ) ].                        (8)
                                                                                                     2       𝑞𝑖             𝑞𝑖      𝑖=1
                                                                                                                𝑖=1                     𝑖=1
   Proof. This is a direct result of Lemma 3.6 in [7].                       □
                                                                                                                                           𝑁𝛿             𝑁𝛿
                                                                                            It is then straightforward to show that 𝛿2 [ Î𝐿−1    +(𝑇 − Î𝐿−1      ) 1−
   We now establish a regret lower bound for the system in Fig. 2.                                                                        𝑖=1 𝑞𝑖          𝑖=1 𝑞𝑖
                                                                                                                  𝐿−1                                   1
                                                                                                                                                      −
                                                                                           𝑖=1 (1 − 𝑞𝑖 ) ] = Ω(𝑇
                                                                                                                   𝐿 ). Moreover, setting 𝑞𝑖 = Θ(𝑇 𝐿 ), for all
                                                                                         Î𝐿−1            
                                                                 𝐿−1
   Theorem 4. For the system in Fig. 2, the regret is Ω(𝑇 𝐿 ) for any                                             𝑁𝛿
                                                                                         𝑖 ≤ 𝐿 − 1, makes 𝛿2 [ Î𝐿−1              𝑁𝛿
                                                                                                                        + (𝑇 − Î𝐿−1
                                                                                                                                              Î𝐿−1
                                                                                                                                        ) 1 − 𝑖=1
                                                                                                                                                               
                                                                                                                                                     (1 − 𝑞𝑖 ) ] =
bandit learning policy employed by node 𝐿 and any time-homogeneous                                                          𝑖=1 𝑞𝑖                𝑖=1 𝑞𝑖
oracle policies employed by nodes 1, 2, . . . , 𝐿 − 1.
                                                                                              𝐿−1
                                                                                         Θ(𝑇 𝐿 ).                                                                            □

   Proof. Let 𝑇𝛿 be the time at which 𝑁𝐿 (𝑡) = 𝑁𝛿 . By Lemma 5,                           Before closing the section, we note that the lower-bound analysis
𝐸 𝑦 [𝐿, 𝑡] > (1 − (2𝐿 − 2𝐿−1 )𝛿)/2 for any 𝑡 < 𝑇𝛿 .                                      in this section is limited to time-homogeneous policies. We make
    We first study the system behavior before time 𝑇𝛿 . Consider                         this assumption to explicitly prevent a parent node from using
the forwarding decision of node 𝐿 − 1 at any time 𝑡 < 𝑇𝛿 . Node                         history to imply internal variables of its children. Extending our
𝐿 − 1 has two children. One is the leaf node 2𝐿 − 1 with 𝐸 𝑦 [2𝐿 −                       analysis to time-varying policies will be interesting future work.
     
1, 𝑡] = 𝑝 2𝐿−1 = (1 − (2𝐿 − 2𝐿−2 )𝛿)/2. The other is the non-leaf
                                                                                         7    SIMULATION RESULTS
                        
node 𝐿 with 𝐸 𝑦 [𝐿, 𝑡] > (1 − (2𝐿 − 2𝐿−1 )𝛿)/2 = 𝑝 2𝐿−1 + 2𝐿−3𝛿.
By A2, the probability that node 𝐿 − 1 selects node 𝐿 is at most                       We present our simulation results in this section. We simulate two
𝑞𝐿−1 := 𝑃𝐿−1 (2𝐿−3𝛿). We also have 𝐸 𝑦 [𝐿 − 1, 𝑡] ≥ 𝑝 2𝐿−1 =                             different scenarios. The first scenario is based on trees whose leaf
(1 − (2𝐿 − 2𝐿−2 )𝛿)/2.                                                                   nodes generate Bernoulli costs. While this scenario is artificially
    We further analyze the forwarding decision of node 𝑖 < 𝐿 − 1                         constructed and may not correspond to real-world applications, its
at any time 𝑡 < 𝑇𝛿 . Using a simple induction argument, it can                           simulation results provide important insights on how online algo-
be shown that the probability that    node 𝑖 selects node 𝑖 + 1 is at                  rithms behave in distributed multi-stage systems. The second sce-
most 𝑞𝑖 := 𝑃𝑖 (2𝑖 −2𝛿), and 𝐸 𝑦 [𝑖, 𝑡] ≥ 𝑝 𝐿+𝑖 = (1 − (2𝐿 − 2𝑖 −1 )𝛿)/2.                 nario is based on mobile edge computing. We compare our 𝜖−EXP3,
```


## Page 8

```text
MOBIHOC ’24, October 14–17, 2024, Athens, Greece                                                                                                    I-Hong Hou


                 (a) 𝐷 = 2, 𝐿 = 2, 𝑝𝑚𝑖𝑛 = 0.2                        (b) 𝐷 = 2, 𝐿 = 3, 𝑝𝑚𝑖𝑛 = 0.4                    (c) 𝐷 = 2, 𝐿 = 4, 𝑝𝑚𝑖𝑛 = 0.6


                 (d) 𝐷 = 4, 𝐿 = 2, 𝑝𝑚𝑖𝑛 = 0.2                        (e) 𝐷 = 4, 𝐿 = 3, 𝑝𝑚𝑖𝑛 = 0.4                    (f) 𝐷 = 4, 𝐿 = 4, 𝑝𝑚𝑖𝑛 = 0.6


                                            Figure 3: Time-average regrets under various system parameters


with parameters from Lemma 4, against the standard EXP3 algo-                            Simulation results are shown in Fig. 3, with the error bars indi-
rithm, where each node runs the EXP3 algorithm independently                          cating standard deviations. It can be observed that the time-average
from each other, and the Broad-OMD algorithm [29]. Both EXP3                          regret of 𝜖−EXP3 approaches 0 over time in all cases. We note
and Broad-OMD are no-regret policies for the special case when                        that the convergence rate of 𝜖−EXP3 becomes much slower as 𝐿
𝐿 = 1.                                                                                becomes larger. This is consistent with Theorem √      2, which shows
                                                                                      that the time-average regret scales as 𝑂 (1/ 𝐿+1 𝑇 ).√To verify that the
7.1     Trees with Bernoulli Costs                                                    time-average regret of 𝜖−EXP3 scales as 𝑂 (1/ 𝐿+1 𝑇 ), we also plot
We consider systems that can be represented as trees with depth 𝐿+1.                  the asymptotic trend in Fig. 3. The value√ of the asymptotic trend for
Each non-leaf node has 𝐷 children. Each leaf node 𝑗 is associated                     a particular 𝑇 is calculated as 𝑅𝐷,𝐿 / 𝐿+1 𝑇 , where 𝑅𝐷,𝐿 is chosen so
with a parameter 𝑝 𝑗 ∈ [0, 1]. Whenever a leaf node 𝑗 receives a job,                 that the value of the asymptotic trend and the time-average regret
its cost 𝑐 [ 𝑗, 𝑡] is 1 with probability 𝑝 𝑗 and 0 with probability 1 − 𝑝 𝑗 .         of 𝜖−EXP3 are the same when 𝑇 = 5 × 106 , that is, at the mid-point
The system is run over 𝑇 rounds. Initially, the values of 𝑝 𝑗 is chosen               of the x-axis in the figures. It can be observed that 𝜖−EXP3 is close
so that max 𝑗 𝑝 𝑗 = 1 and min 𝑗 𝑝 𝑗 = 𝑝𝑚𝑖𝑛 . At round 𝑡 = 𝑇 /100, the                 to the asymptotic trend. This demonstrates √ that the time-average
leaf with 𝑝 𝑗 = 1 has its value of 𝑝 𝑗 changed into 0. Fig. 4 illustrates             regret 𝜖−EXP3 indeed scales as 𝑂 (1/ 𝐿+1 𝑇 ).
an example. For a given set of parameters 𝐷, 𝐿,𝑇 , and [𝑝 𝑗 ], we                        On the other hand, it can also be observed that the time-average
simulate the system for 20 independent runs and calculate                 the        regrets of both EXP3 and Broad-OMD converge to 𝑝𝑚𝑖𝑛 in all set-
                                                                     
                           Í𝑇                         Í𝑇                              tings in Fig. 3. This result shows that neither of them is a no-regret
time-average regret          𝑡 =1 𝑦 [𝑟, 𝑡] − min 𝑗 ∈ L 𝑡 =1 𝑐 [ 𝑗, 𝑡] /𝑇 under        policy in multi-stage systems. To understand why the standard
all evaluated policies.                                                               EXP3 algorithm is not a no-regret policy, consider the system il-
                                                                                      lustrated in Fig. 4. Before round 𝑡 = 100 𝑇 , the optimal strategy for
                                                                                      node 1 is to forward the job to node 4 with 𝑝 4 = 0.6. The optimal
                                                                                      strategy for the root is to forward the job to node 2, who then for-
                                                                                      wards the job to node 6 with 𝑝 6 = 0.2. Hence, at round 𝑡 = 100  𝑇 and
                                                                                      under the EXP3 algorithm, the root will choose node 2 with a high
                                                                                      probability and node 1 will choose node 4 with a high probability.
                                                                                      Now, consider the first time after round 100𝑇 when the root forwards
                                                                                      a job to node 1. Since node 1 is unaware that 𝑝 3 has become 0, it
                                                                                      chooses node 4 with a high probability and will likely incur a high
                      1, 𝑖𝑓 𝑡 < 𝑇/100   𝑝# = 0.6   𝑝" = 0.4   𝑝! = 0.2
             𝑝$ = (                                                                   cost. This high cost will cause the root to exponentially reduce
                      0, 𝑖𝑓 𝑡 ≥ 𝑇/100
                                                                                      the probability of choosing node 1 in the future, making it even
                                                                                      harder for node 1 to explore and learn the fact that 𝑝 3 has become 0.
      Figure 4: A system with 𝐷 = 2, 𝐿 = 2 and 𝑝𝑚𝑖𝑛 = 0.2                             This is why the EXP3 algorithm suffers from a time-average regret
```


## Page 9

```text
Distributed No-Regret Learning for Multi-Stage Systems with End-to-End Bandit Feedback                                                                          MOBIHOC ’24, October 14–17, 2024, Athens, Greece


                   1                                                             1                                               is sufficiently large. The Broad-OMD algorithm has similar perfor-
                                                                                                 Prob(root chooses node 1)
                  0.8                                                           0.8
                                                                                                 Prob(node 1 chooses node 3)
                                                                                                                                 mance as 𝜖−EXP3 when 𝐷 = 2, but is much worse than 𝜖−EXP3
    Probability


                                                                  Probability
                  0.6                                                           0.6
                                                                                                                                 when 𝐷 = 3.
                  0.4                                                           0.4

                  0.2             Prob(root chooses node 1)                     0.2

                   0
                                  Prob(node 1 chooses node 3)
                                                                                 0                                               7.3     Multi-hop Networks
                        0   100   200     300      400      500                       0    100   200     300      400      500
                                    t(*1000)                                                       t(*1000)                      We consider multi-hop networks as illustrated in Fig. 7(a). In this
                   (a) The behavior of 𝜖 − EXP3                                       (b) The behavior of EXP3                   system, the source (node 𝑆) is sending packets to the destination
                                                                                                                                 (node 𝐷) through a number of inter-connected relay nodes. Upon
                                                                                                                                 receiving a packet, a node needs to decide which node to forward
Figure 5: Transient behaviors of the system in Fig. 4 with                                                                       the packet to. The delay of transmitting over a link is an exponential
𝑇 = 5 × 106 .                                                                                                                    function with mean 𝜆1 . Some links have a constant 𝜆 while other
                                                                                                                                 links have a 𝜆 that increases over time. We consider that the source
of roughly 𝑝 6 = 0.2. In contrast, our 𝜖−EXP3 algorithm ensures                                                                  requires a strict end-to-end deadline guarantee of one unit time. If
that the root always chooses node 1 with at least a constant proba-                                                              the end-to-end delay of a packet is more than one unit time, then a
bility in each round. This persistent education enables node 1 to                                                                deadline violation occurs.
eventually discover that 𝑝 3 has become 0.                                                                                          Let 𝐿 be the number of relay nodes that a packet needs to visit
   To demonstrate the behavior discussed in the above paragraph,                                                                 before reaching the destination. We have tested this system for
we conduct a simulation to show the transient behaviors of the two                                                               different values of 𝐿. Simulation results are shown in Fig. 7. It can
algorithms. Specifically, we test the system shown in Fig. 4 with                                                                be observed that the 𝜖−EXP3 algorithm is either optimal or near-
𝑇 = 5 × 106 . The value of 𝑝 3 is initially 1, and becomes 0 at round                                                            optimal in all settings.
5 × 104 . For each algorithm, we record the probability that the root 𝑟
would choose node 1 and the probability that node 1 would choose                                                                 8     CONCLUSION
node 3. Simulation results are shown in Fig. 5, where each data                                                                  In this paper, we study multi-stage systems with end-to-end ban-
point represent the average of the previous 1000 rounds. Under the                                                               dit feedback. The fundamental challenge of learning the optimal
EXP3 algorithm, the probability that the root would choose node 1                                                                policy of agents in each stage is a newly introduced exploration-
at round 5 × 104 is less than 0.05%. Since node 1 rarely receives any                                                            exploitation-education trilemma. We propose a simple distribute
jobs, it cannot improve its performance, which, in turn, makes the                                                               policy, the 𝜖−EXP3 algorithm, that explicitly addresses this trilemma.
root even less likely to choose node 1. At round 5 × 105 , probability                                                           Moreover, we theoretically prove that the 𝜖−EXP3 algorithm is a no-
that the root would choose node 1 has become less than 0.02%.                                                                    regret policy. Simulation results show that the 𝜖−EXP3 algorithm
In contrast, the 𝜖−EXP3 algorithm offers persistent education to                                                                 significantly outperforms existing policies.
node 1. Hence, after round 5 × 104 , node 1 quickly finds that 𝑝 3
has improved and increases its probability of choosing node 3. As                                                                9     ACKNOWLEDGEMENT
a result, the root also starts increasing its probability of choosing                                                            This material is based upon work supported in part by NSF under
node 1 after round 105 .                                                                                                         Award Numbers ECCS-2127721 and CCF-2332800 and in part by
                                                                                                                                 the U.S. Army Research Laboratory and the U.S. Army Research
7.2                 Mobile Edge Computing                                                                                        Office under Grant Number W911NF-22-1-0151.
We consider a mobile edge computing system. In this system, there
is a mobile robot that generates video analytic jobs for real-time                                                               REFERENCES
processing. The robot is connected to 𝐷 edge servers with different                                                               [1] Mridul Agarwal, Vaneet Aggarwal, and Kamyar Azizzadenesheli. 2022. Multi-
communication media. To process a job, each edge server has 𝐷                                                                         agent multi-armed bandits with limited communication. The Journal of Machine
different neural networks to choose from. Different neural networks                                                                   Learning Research 23, 1 (2022), 9529–9552.
                                                                                                                                  [2] ABM Alim Al Islam, SM Iftekharul Alam, Vijay Raghunathan, and Saurabh Bagchi.
have different precision and different processing time. There is also                                                                 2012. Multi-armed bandit congestion control in multi-hop infrastructure wireless
a communication latency of each link. The delay of transmitting                                                                       mesh networks. In 2012 IEEE 20th International Symposium on Modeling, Analysis
                                                                                                                                      and Simulation of Computer and Telecommunication Systems. IEEE, 31–40.
over a link is an exponential function with mean 𝜆1 . Some links have                                                             [3] Jean-Yves Audibert and Sébastien Bubeck. 2009. Minimax Policies for Adversarial
a constant 𝜆 while other links have a 𝜆 that increases over time.                                                                     and Stochastic Bandits.. In COLT, Vol. 7. 1–122.
This models the time-varying congestion on these links. Fig. 6(a)                                                                 [4] Jean-Yves Audibert and Sébastien Bubeck. 2010. Regret bounds and minimax
                                                                                                                                      policies under partial monitoring. The Journal of Machine Learning Research 11
illustrates the system when 𝐷 = 2.                                                                                                    (2010), 2785–2836.
    The robot requires a strict deadline of one time unit for each                                                                [5] Peter Auer, Nicolo Cesa-Bianchi, Yoav Freund, and Robert E Schapire. 2002. The
job. If the end-to-end latency, that is, the sum of communication                                                                     nonstochastic multiarmed bandit problem. SIAM journal on computing 32, 1
                                                                                                                                      (2002), 48–77.
latency and processing time, exceeds one time unit, then a deadline                                                               [6] AA Bhorkar and T Javidi. 2010. No regret routing for ad-hoc wireless networks.
violation occurs and the cost is one. If the end-to-end latency is less                                                               In 2010 Conference Record of the Forty Fourth Asilomar Conference on Signals,
                                                                                                                                      Systems and Computers. IEEE, 676–680.
than one time unit, then the cost is the miss rate of the employed                                                                [7] Sébastien Bubeck, Nicolo Cesa-Bianchi, et al. 2012. Regret analysis of stochastic
neural network.                                                                                                                       and nonstochastic multi-armed bandit problems. Foundations and Trends® in
    We have conducted 20 independent runs for each 𝑇 . Simulation                                                                     Machine Learning 5, 1 (2012), 1–122.
                                                                                                                                  [8] Nicolò Cesa-Bianchi, Tommaso Cesari, and Claire Monteleoni. 2020. Cooperative
results are shown in Fig. 6. It can be observed that the 𝜖−EXP3                                                                       online learning: Keeping your neighbors updated. In Algorithmic learning theory.
algorithm significantly outperforms the EXP3 algorithm when 𝑇                                                                         PMLR, 234–250.
```


## Page 10

```text
MOBIHOC ’24, October 14–17, 2024, Athens, Greece                                                                                                                          I-Hong Hou


                         Processing time = 0.2; Miss rate = 10%


                             Delay =                Delay =
                            𝐸𝑋𝑃(𝜆 !)               𝐸𝑋𝑃(𝜆 " )


                         Processing time = 0.5; Miss rate = 0.5%


                   (a) System illustration when 𝐷 = 2                                     (b) Results for 𝐷 = 2                              (c) Results for 𝐷 = 3


                                                       Figure 6: Setting and result of a mobile edge computing system


                  Links with constant 𝜆
                  Links with increasing 𝜆

                       (a) System illustration                                      (b) Results for 𝐿 = 2                                  (c) Results for 𝐿 = 3


                                                                   Figure 7: Setting and result of multi-hop networks


 [9] Ronshee Chawla, Abishek Sankararaman, Ayalvadi Ganesh, and Sanjay Shakkot-                   [21] David Martínez-Rubio, Varun Kanade, and Patrick Rebeschini. 2019. Decentral-
     tai. 2020. The gossiping insert-eliminate algorithm for multi-agent bandits. In                   ized cooperative stochastic bandits. Advances in Neural Information Processing
     International conference on artificial intelligence and statistics. PMLR, 3471–3481.              Systems 32 (2019).
[10] Sandeep Chinchali, Apoorva Sharma, James Harrison, Amine Elhafsi, Daniel                     [22] Gergely Neu. 2015. Explore no more: Improved high-probability regret bounds
     Kang, Evgenya Pergament, Eyal Cidon, Sachin Katti, and Marco Pavone. 2021.                        for non-stochastic bandits. Advances in Neural Information Processing Systems 28
     Network offloading policies for cloud robotics: a learning-based approach. Au-                    (2015).
     tonomous Robots 45, 7 (2021), 997–1012.                                                      [23] Conor Newton, Ayalvadi Ganesh, and Henry Reeve. 2022. Asymptotic Optimality
[11] Han Deng, Tao Zhao, and I-Hong Hou. 2019. Online routing and scheduling                           for Decentralised Bandits. ACM SIGMETRICS Performance Evaluation Review 49,
     with capacity redundancy for timely delivery guarantees in multihop networks.                     2 (2022), 51–53.
     IEEE/ACM Transactions on Networking 27, 3 (2019), 1258–1271.                                 [24] Daehyun Park, Sunjung Kang, and Changhee Joo. 2021. A learning-based dis-
[12] Yan Gu, Bo Liu, and Xiaojun Shen. 2021. Asymptotically Optimal Online Sched-                      tributed algorithm for scheduling in multi-hop wireless networks. Journal of
     uling With Arbitrary Hard Deadlines in Multi-Hop Communication Networks.                          Communications and Networks 24, 1 (2021), 99–110.
     IEEE/ACM Transactions on Networking 29, 4 (2021), 1452–1466.                                 [25] Shai Shalev-Shwartz. 2012. Online learning and online convex optimization.
[13] Aria HasanzadeZonuzy, Dileep Kalathil, and Srinivas Shakkottai. 2020. Rein-                       Foundations and Trends® in Machine Learning 4, 2 (2012), 107–194.
     forcement learning for multi-hop scheduling and routing of real-time flows. In               [26] Hsien-Po Shiang and Mihaela van der Schaar. 2010. Online learning in autonomic
     2020 18th International Symposium on Modeling and Optimization in Mobile, Ad                      multi-hop wireless networks for transmitting mission-critical applications. IEEE
     Hoc, and Wireless Networks (WiOPT). IEEE, 1–8.                                                    Journal on Selected Areas in Communications 28, 5 (2010), 728–741.
[14] Zhaoliang He, Yuan Wang, Chen Tang, Zhi Wang, Wenwu Zhu, Chenyang Guo,                       [27] Adish Singla, Hamed Hassani, and Andreas Krause. 2018. Learning to inter-
     and Zhibo Chen. 2022. AdaConfigure: Reinforcement Learning-Based Adap-                            act with learning agents. In Proceedings of the AAAI Conference on Artificial
     tive Configuration for Video Analytics Services. In International Conference on                   Intelligence, Vol. 32.
     Multimedia Modeling. Springer, 245–257.                                                      [28] Taishi Uchiya, Atsuyoshi Nakamura, and Mineichi Kudo. 2010. Algorithms for
[15] Junchen Jiang, Ganesh Ananthanarayanan, Peter Bodik, Siddhartha Sen, and Ion                      adversarial bandit problems with multiple plays. In International Conference on
     Stoica. 2018. Chameleon: scalable adaptation of video analytics. In Proceedings of                Algorithmic Learning Theory. Springer, 375–389.
     the 2018 Conference of the ACM Special Interest Group on Data Communication.                 [29] Chen-Yu Wei and Haipeng Luo. 2018. More adaptive algorithms for adversarial
     253–266.                                                                                          bandits. In Conference On Learning Theory. PMLR, 1263–1291.
[16] Peter Landgren, Vaibhav Srivastava, and Naomi Ehrich Leonard. 2021. Distributed              [30] Kun Wu, Yibo Jin, Weiwei Miao, Zeng Zeng, Zhuzhong Qian, Jingmian Wang,
     cooperative decision making in multi-agent multi-armed bandits. Automatica                        Mingxian Zhou, and Tuo Cao. 2021. Soudain: Online Adaptive Profile Con-
     125 (2021), 109445.                                                                               figuration for Real-time Video Analytics. In 2021 IEEE/ACM 29th International
[17] Zhichu Lin and Mihaela van der Schaar. 2010. Autonomic and distributed joint                      Symposium on Quality of Service (IWQOS). IEEE, 1–10.
     routing and power control for delay-sensitive applications in multi-hop wireless             [31] Jian Zhang, Jian Tang, and Feng Wang. 2020. Cooperative relay selection for load
     networks. IEEE Transactions on Wireless Communications 10, 1 (2010), 102–113.                     balancing with mobility in hierarchical WSNs: A multi-armed bandit approach.
[18] Keqin Liu and Qing Zhao. 2010. Distributed learning in multi-armed bandit with                    IEEE Access 8 (2020), 18110–18122.
     multiple players. IEEE transactions on signal processing 58, 11 (2010), 5667–5681.           [32] Sheng Zhang, Can Wang, Yibo Jin, Jie Wu, Zhuzhong Qian, Mingjun Xiao, and
[19] Udari Madhushani, Abhimanyu Dubey, Naomi Leonard, and Alex Pentland. 2021.                        Sanglu Lu. 2021. Adaptive Configuration Selection and Bandwidth Allocation for
     One more step towards reality: Cooperative bandits with imperfect communica-                      Edge-Based Video Analytics. IEEE/ACM Transactions on Networking 30, 1 (2021),
     tion. Advances in Neural Information Processing Systems 34 (2021), 7813–7824.                     285–298.
[20] Zhoujia Mao, Can Emre Koksal, and Ness B Shroff. 2014. Optimal online
     scheduling with arbitrary hard deadlines in multihop communication networks.
     IEEE/ACM Transactions on Networking 24, 1 (2014), 177–189.
```


## Page 11

```text
Distributed No-Regret Learning for Multi-Stage Systems with End-to-End Bandit Feedback                                           MOBIHOC ’24, October 14–17, 2024, Athens, Greece


A     PROOF OF THEOREM 1                                                                 chooses 𝑓 [𝑖, 𝑡] = 𝑗, whose probability depends on 𝑚[𝑖, 𝑡]. Hence,
     Proof. We will prove the theorem by establishing the following                             h                                 i
statement:h If a inode     𝑖 is (𝐿 − ℎ)-hops from the root node 𝑟 , then                      𝐸 𝑧 [ 𝑗, 𝑡] Y𝜖 [𝑖, 𝑡], Z [𝑖, 𝑡 − 1]
Í𝑇                       Í𝑇                     √︁
   𝑡 =1 𝐸  𝑦𝑛 [𝑖, 𝑡]   ≤    𝑡 =1 𝑦 ∗ [𝑖, 𝑡] + 2ℎ 𝑇 log 𝐷.
                                                                                                  h                                                i
                                                                                             =𝜖𝑖 𝐸 𝑧 [ 𝑗, 𝑡] 𝑚[𝑖, 𝑡] = 𝑈 , Y𝜖 [𝑖, 𝑡], Z [𝑖, 𝑡 − 1]
     We prove the statement by induction. First, consider the case                                           h                                             i
ℎ = 1, that is, the node 𝑖 is (𝐿 − 1)-hops from 𝑟 . Since the tree has                         + (1 − 𝜖𝑖 )𝐸 𝑧 [ 𝑗, 𝑡] 𝑚[𝑖, 𝑡] = 𝐸, Y𝜖 [𝑖, 𝑡], Z [𝑖, 𝑡 − 1]
depth 𝐿 + 1, either 𝑖 is a leaf node or all children of 𝑖 are leaf nodes.
If 𝑖 is a leaf node, then 𝑦𝑛 [𝑖, 𝑡] = 𝑦∗ [𝑖, 𝑡] = 𝑐 [𝑖, 𝑡] ∈ [0, 1] and the                                    1 𝑦𝜖 [ 𝑗, 𝑡]|C𝑖 |
                                                                                             =𝜖𝑖 𝑣 [𝑖, 𝑡]
statement holds. If all children of 𝑖 are leaf nodes, then we have                                            |C𝑖 | 𝑣 [𝑖, 𝑡]
𝑦𝑛 [ 𝑗, 𝑡] = 𝑦∗ [ 𝑗, 𝑡] = 𝑐 [ 𝑗, 𝑡] for all 𝑗 ∈ C𝑖 . Hence, by Lemma 1,                                               𝑒 𝜂𝑖 𝜃 [𝑖,𝑗,𝑡 ]     𝑦𝜖 [ 𝑗, 𝑡] 𝑘 ∈ C𝑖 𝑒 𝜂𝑖 𝜃 [𝑖,𝑘,𝑡 ]
                                                                                                                                                    Í
                                                                                               + (1 − 𝜖𝑖 )𝑣 [𝑖, 𝑡] Í
                                                                                                                             𝜂 𝜃 [𝑖,𝑘,𝑡 ]       𝑣 [𝑖, 𝑡]𝑒 𝜂𝑖 𝜃 [𝑖,𝑗,𝑡 ]
                                                                                                                    𝑘 ∈ C𝑖 𝑒 𝑖
         𝑇
        ∑︁  h            𝑇
                      i ∑︁  h                    i                                           =𝑦𝜖 [ 𝑗, 𝑡],
           𝐸 𝑦𝑛 [𝑖, 𝑡] =   𝐸 𝑦𝑛 [𝑖, 𝑡] Y𝑛 [𝑖, 𝑡]
        𝑡 =1                            𝑡 =1
                                                                                         and
                  𝑇
                 ∑︁                   √︁
     ≤ min              𝑦𝑛 [ 𝑗, 𝑡] + 2 𝑇 log |C𝑖 |                                           h                                   i
        𝑗 ∈ C𝑖
                 𝑡 =1                                                                      𝐸 𝑧 [ 𝑗, 𝑡] 2 Y𝜖 [𝑖, 𝑡], Z [𝑖, 𝑡 − 1]
                  𝑇                               𝑇
                                                                                               h                                                  i
                 ∑︁                   √︁         ∑︁              √︁                       =𝜖𝑖 𝐸 𝑧 [ 𝑗, 𝑡] 2 𝑚[𝑖, 𝑡] = 𝑈 , Y𝜖 [𝑖, 𝑡], Z [𝑖, 𝑡 − 1]
     ≤ min              𝑦∗ [ 𝑗, 𝑡] + 2 𝑇 log 𝐷 =    𝑦∗ [𝑖, 𝑡] + 2 𝑇 log 𝐷,
        𝑗 ∈ C𝑖                                                                                              h                                             i
                 𝑡 =1                                   𝑡 =1
                                                                                            + (1 − 𝜖𝑖 )𝐸 𝑧 [ 𝑗, 𝑡] 2 𝑚[𝑖, 𝑡] = 𝐸, Y𝜖 [𝑖, 𝑡], Z [𝑖, 𝑡 − 1]

and the statement holds.                                                                                  1 𝑦𝜖 [ 𝑗, 𝑡] 2 |C𝑖 | 2
                                                                                          =𝜖𝑖 𝑣 [𝑖, 𝑡]
   We now assume that the statement holds when ℎ = 𝑔 and con-                                            |C𝑖 |  𝑣 [𝑖, 𝑡] 2
sider a node 𝑖 that is (𝐿 − (𝑔 + 1))-hops from 𝑟 . Either 𝑖 is a leaf node                                         𝑒 𝜂𝑖 𝜃 [𝑖,𝑗,𝑡 ]       𝑦𝜖 [ 𝑗, 𝑡] 2 ( 𝑘 ∈ C𝑖 𝑒 𝜂𝑖 𝜃 [𝑖,𝑘,𝑡 ] ) 2
                                                                                                                                                       Í
or all children of 𝑖 are (𝐿 −𝑔)-hops from 𝑟 . If 𝑖 is a leaf node, then the                  + (1 − 𝜖𝑖 )𝑣 [𝑖, 𝑡] Í
                                                                                                                           𝜂 𝜃 [𝑖,𝑘,𝑡 ]          𝑣 [𝑖, 𝑡] 2𝑒 2𝜂𝑖 𝜃 [𝑖,𝑗,𝑡 ]
                                                                                                                 𝑘 ∈ C𝑖 𝑒 𝑖
statement clearly holds. If 𝑖 ish not a leaf
                                           i node, then, by the induction                                        Í          𝜂𝑖 𝜃 [𝑖,𝑘,𝑡 ] 
                        Í𝑇                     Í𝑇                √︁
hypothesis, we have 𝑡 =1 𝐸 𝑦𝑛 [ 𝑗, 𝑡] ≤ 𝑡 =1 𝑦∗ [ 𝑗, 𝑡] + 2𝑔 𝑇 log 𝐷,
                                                                                           
                                                                                                                  𝑘 ∈ C𝑖 𝑒                  𝑦𝜖 [ 𝑗, 𝑡] 2
                                                                                          = 𝜖𝑖 |C𝑖 | + (1 − 𝜖𝑖 )                                         .
for all 𝑗 ∈ C𝑖 . Since 𝑦𝑛 [ 𝑗, 𝑡] ∈ [0, 1], we can use Lemma 1 to estab-                                             𝑒 𝜂𝑖 𝜃 [𝑖,𝑗,𝑡 ]         𝑣 [𝑖, 𝑡]
lish the following:                                                                                                                                                                  □

                    𝑇
                   ∑︁  h            𝑇
                                 i ∑︁  h h                    ii                         C     PROOF OF LEMMA 3
                      𝐸 𝑦𝑛 [𝑖, 𝑡] =   𝐸 𝐸 𝑦𝑛 [𝑖, 𝑡] Y𝑛 [𝑖, 𝑡]
                   𝑡 =1                        𝑡 =1
                                                                                             Proof. Since both 𝜖−EXP3 and normalized-EG update 𝜃 [𝑖, 𝑗, 𝑡]
                    h             𝑇             i                                        by 𝜃 [𝑖, 𝑗, 𝑡 + 1] = 𝜃 [𝑖, 𝑗, 𝑡] − 𝑧 [ 𝑗, 𝑡], they have the same values of
                                                                                         𝜃 [𝑖, 𝑗, 𝑡] on every sample path.
                                 ∑︁                   √︁
               ≤𝐸 min                   𝑦𝑛 [ 𝑗, 𝑡] + 2 𝑇 log |C𝑖 |
                        𝑗 ∈ C𝑖                                                               By the design of the 𝜖−EXP3 algorithm, we have
                                 𝑡 =1
                      𝑇
                     ∑︁  h          i   √︁                                                                     h                                               i
               ≤ min    𝐸 𝑦𝑛 [ 𝑗, 𝑡] + 2 𝑇 log 𝐷                                                              𝐸 𝑦𝜖 [𝑖, 𝑡] 𝑚[𝑖, 𝑡] = 𝐸, Y𝜖 [𝑖, 𝑡], Z [𝑖, 𝑡 − 1]
                   𝑗 ∈ C𝑖
                            𝑡 =1
                             𝑇
                                                                                                               ∑︁                     𝑒 𝜂𝑖 𝜃 [𝑖,𝑗,𝑡 ]
                            ∑︁                          √︁                                               =             𝑦𝜖 [ 𝑗, 𝑡] Í                       .
               ≤ min               𝑦∗ [ 𝑗, 𝑡] + 2(𝑔 + 1) 𝑇 log 𝐷                                                                             𝜂 𝜃 [𝑖,𝑘,𝑡 ]
                                                                                                              𝑗 ∈ C𝑖                𝑘 ∈ C𝑖 𝑒 𝑖
                   𝑗 ∈ C𝑖
                            𝑡 =1
                    𝑇
                   ∑︁                         √︁                                           Under the normalized-EG algorithm, we have 𝑦𝑛 [𝑖, 𝑡] = 𝑧 [ 𝑗, 𝑡]
               =          𝑦∗ [𝑖, 𝑡] + 2(𝑔 + 1) 𝑇 log 𝐷,                                                      𝜂𝑖 𝜃 [𝑖,𝑗,𝑡 ]
                   𝑡 =1                                                                  with probability Í 𝑒 𝜂𝑖 𝜃 [𝑖,𝑘,𝑡 ] . Hence,
                                                                                                                       𝑘 ∈C𝑖 𝑒

                                                                                                    h h                               ii
and the statement holds. By induction, the statement holds for all                                 𝐸 𝐸 𝑦𝑛 [𝑖, 𝑡] Y𝑛 [𝑖, 𝑡] = Z [𝑖, 𝑡]
ℎ.                                                              h        i
                                                          Í
     Since the root node 𝑟 is 0-hop from itself, we have 𝑇𝑡=1 𝐸 𝑦𝑛 [𝑟, 𝑡] ≤
                                                                                                     ∑︁            𝑒 𝜂𝑖 𝜃 [𝑖,𝑗,𝑡 ]      h                                  i
                                                                                                 =                                     𝐸 𝑧 [ 𝑗, 𝑡] Y𝜖 [𝑖, 𝑡], Z [𝑖, 𝑡 − 1]
                       √︁                                  √︁                                                 Í           𝜂 𝜃 [𝑖,𝑘,𝑡 ]
                                                                                                                 𝑘 ∈ C𝑖 𝑒 𝑖
Í𝑇                                         Í𝑇
   𝑡 =1 𝑦 ∗ [𝑟, 𝑡] + 2𝐿 𝑇 log 𝐷 = min 𝑗 ∈ L 𝑡 =1 𝑐 𝑗,𝑡 + 2𝐿 𝑇 log 𝐷.    □                            𝑗 ∈ C𝑖
                                                                                                     ∑︁                       𝑒 𝜂𝑖 𝜃 [𝑖,𝑗,𝑡 ]
                                                                                                 =            𝑦𝜖 [ 𝑗, 𝑡] Í                        .           (∵ Eq. (2))
B    PROOF OF LEMMA 2                                                                                                                𝜂 𝜃 [𝑖,𝑘,𝑡 ]
                                                                                                     𝑗 ∈ C𝑖                 𝑘 ∈ C𝑖 𝑒 𝑖
  Proof. Under the 𝜖−EXP3 algorithm, 𝑧 [ 𝑗, 𝑡] ≠ 0 only when
node 𝑖 receives a job, which happens with probability 𝑣 [𝑖, 𝑡], and 𝑖                    This completes the proof.                                                                   □
```


## Page 12

```text
MOBIHOC ’24, October 14–17, 2024, Athens, Greece                                                                                                                                               I-Hong Hou


D     PROOF OF LEMMA 4
                                                                                        𝜂𝑖 𝜃 [𝑖,𝑗,𝑡 ]
    Proof. The normalized-EG algorithm sets 𝑥 [𝑖, 𝑗, 𝑡] = Í 𝑒                                𝜂 𝜃 [𝑖,𝑘,𝑡 ]   .                 𝑇
                                                                                                                             ∑︁  h                   i       𝑇
                                                                                                                                                            ∑︁
                                                                                     𝑘 ∈C𝑖 𝑒 𝑖                                  𝐸 𝑦𝜖 [𝑖, 𝑡] Y𝜖 [𝑖, 𝑡] − min    𝑦𝜖 [ 𝑗, 𝑡]
By Lemma 1 and the fact that 𝑦𝜖 [ 𝑗, 𝑡] ∈ [0, 1], ∀𝑗, 𝑡, we have                                                             𝑡 =1
                                                                                                                                                                  𝑗 ∈ C𝑖
                                                                                                                                                                           𝑡 =1
                                                                                                                             (        𝐿                  𝐿
       𝑇
      ∑︁  h h                               ii                                                                                   𝑇   𝐿+1   log 𝐷 + 𝐷𝑇   𝐿+1   ,                   if 𝐶𝑖 ⊂ L,
         𝐸 𝐸 𝑦𝑛 [𝑖, 𝑡] Y𝑛 [𝑖, 𝑡] = Z [𝑖, 𝑡]                                                                              ≤                 𝐿    𝐿                      𝐿
                                                                                                                                 𝐷𝑇 𝐿+1 + 𝑇 𝐿+1 log 𝐷 + 𝐷𝑇 𝐿+1                    else.
      𝑡 =1
      h         𝑇
               ∑︁            i log |C |                                                                         This completes the proof.                                                              □
                                      𝑖
    ≤𝐸 min          𝑧 [ 𝑗, 𝑡] +
        𝑗 ∈ C𝑖                    𝜂 𝑖
               𝑡 =1

                                    𝑒 𝜂𝑖 𝜃 [𝑖,𝑗,𝑡 ]
                𝑇 ∑︁
               ∑︁                                        h            i
      + 𝜂𝑖                                              𝐸 𝑧 [ 𝑗, 𝑡] 2
                             Í             𝜂 𝜃 [𝑖,𝑘,𝑡 ]
               𝑡 =1 𝑗 ∈ C𝑖        𝑘 ∈ C𝑖 𝑒 𝑖
                𝑇                                     𝑇
               ∑︁                     log |C𝑖 |      ∑︁     |C𝑖 |
    ≤ min             𝑦𝜖 [ 𝑗, 𝑡] +              + 𝜂𝑖                      (∵ Lemma 2)
      𝑗 ∈ C𝑖
               𝑡 =1
                                         𝜂𝑖          𝑡 =1
                                                          𝑣 [𝑖, 𝑡]
                𝑇                                 𝑇
               ∑︁                     log 𝐷      ∑︁      𝐷
    ≤ min             𝑦𝜖 [ 𝑗, 𝑡] +          + 𝜂𝑖               .
      𝑗 ∈ C𝑖
               𝑡 =1
                                        𝜂𝑖       𝑡 =1
                                                      𝑣 [𝑖, 𝑡]

Combining the above inequality with Lemma 3 and we have
                  𝑇
                 ∑︁  h                                               i
                    𝐸 𝑦𝜖 [𝑖, 𝑡] 𝑚[𝑖, 𝑡] = 𝐸, Y𝜖 [𝑖, 𝑡], Z [𝑖, 𝑡 − 1]
                 𝑡 =1
                           𝑇                                𝑇
                          ∑︁                    log 𝐷      ∑︁      𝐷
               ≤ min             𝑦𝜖 [ 𝑗, 𝑡] +         + 𝜂𝑖               .
                 𝑗 ∈ C𝑖
                          𝑡 =1
                                                  𝜂𝑖       𝑡 =1
                                                                𝑣 [𝑖, 𝑡]

Moreover, since 𝑦𝜖 [ 𝑗, 𝑡] ∈ [0, 1]∀𝑗, 𝑡, we clearly have
                  𝑇
                 ∑︁  h                                                i
                    𝐸 𝑦𝜖 [𝑖, 𝑡] 𝑚[𝑖, 𝑡] = 𝑈 , Y𝜖 [𝑖, 𝑡], Z [𝑖, 𝑡 − 1]
                 𝑡 =1
                           𝑇
                          ∑︁
             ≤ min               𝑦𝜖 [ 𝑗, 𝑡] + 𝑇 .
                 𝑗 ∈ C𝑖
                          𝑡 =1

   Since 𝑚[𝑖, 𝑡] = 𝑈 with probability 𝜖𝑖 and 𝑚[𝑖, 𝑡] = 𝐸 with proba-
bility 1 − 𝜖𝑖 , we now have
        𝑇
       ∑︁  h                                  i
          𝐸 𝑦𝜖 [𝑖, 𝑡] Y𝜖 [𝑖, 𝑡], Z [𝑖, 𝑡 − 1]
        𝑡 =1
                  𝑇                                                      𝑇
                 ∑︁                                          log 𝐷      ∑︁      𝐷
     ≤ min              𝑦𝜖 [ 𝑗, 𝑡] + 𝜖𝑖 𝑇 + (1 − 𝜖𝑖 )(             + 𝜂𝑖               ),
        𝑗 ∈ C𝑖
                 𝑡 =1
                                                               𝜂𝑖       𝑡 =1
                                                                             𝑣 [𝑖, 𝑡]

and hence
                  𝑇
                 ∑︁  h                    i
                    𝐸 𝑦𝜖 [𝑖, 𝑡] Y𝜖 [𝑖, 𝑡]
                 𝑡 =1
                           𝑇                                        𝑇
                          ∑︁                            log 𝐷      ∑︁      𝐷
               ≤ min             𝑦𝜖 [ 𝑗, 𝑡] + 𝜖𝑖 𝑇 +          + 𝜂𝑖               ,
                 𝑗 ∈ C𝑖
                          𝑡 =1
                                                          𝜂𝑖       𝑡 =1
                                                                        𝑣 [𝑖, 𝑡]

which proves the first part of the theorem.
                                             𝐿
   Moreover, if we choose 𝜂𝑖 = 𝑇 − 𝐿+1 for all 𝑖 and set 𝜖𝑖 to be 0
                           1                                           1
if 𝐶𝑖 ⊂ L, and 𝐷𝑇 𝐿+1 otherwise, then 𝑥 [𝑖, 𝑗, 𝑡] ≥ | 𝜖C𝑖 | ≥ 𝑇 − 𝐿+1
                        −
                                                              𝑖
for all 𝑖, 𝑗, 𝑡. Since 𝑣 [𝑟, 𝑡] = 1 for the root node 𝑟 and each non-leaf
                                                                     𝐿−1
node 𝑖 is at most (𝐿 − 1)−hops from 𝑟 , we have 𝑣 [𝑖, 𝑡] ≥ 𝑇 − 𝐿+1 .
Putting these into the above inequality and we have
```
