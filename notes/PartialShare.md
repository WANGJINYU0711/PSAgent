# Analysis of Partial-Share (PS) Algorithm

Converted from PDF using `pdftotext -layout` for Codex-friendly reading.

> Note: Formula layout and line breaks are OCR/text-extraction based, so a few equations may need manual cleanup.

## Page 1

```text
Analysis of Partial-Share (PS) Algorithm

              Derivation Note




                     1
```

## Page 2

```text
Algorithm 1 Adaptive Partial-Share (PS) Algorithm (Shared branch with IPS update)
 1: Input: Tree T , shared leaf set Lshr , learning rate η, exploration base rate ϵ.
 2: Notation: For a node i, let L(i) denote the set of leaf descendants of i.
 3: Pre-processing: Label a non-leaf node i as Safe if L(i) ⊆ Lshr , and Risky otherwise.
 4: Initialize:
 5:   θℓ ← 0, ∀ℓ ∈ L
 6:   For every edge (i, j) with j ∈ Children(i):
                                  X
                 Sshr [i, j] ←            exp(ηθℓ ),     Sunshr [i, j] ← 1{L(j) ∩ Lunshr ̸= ∅}.
                                ℓ∈L(j)∩Lshr


 7:   Interpret Sunshr [i, j] as a maintained unshared aggregate state, updated online along sampled un-
    shared paths.
 8: for each round t = 1, . . . , T do
 9:    1. Adaptive Selection
10:    i ← root, Πt (root) ← 1
11:    Initialize sampled path Pt ← ∅
12:    while i is not a leaf do
13:        for each j ∈ Children(i) do

                                                       Sshr [i, j] + Sunshr [i, j]
                             Ptexp (j | i) ← P                                             
                                                 k∈Children(i) Sshr [i, k] + Sunshr [i, k]

14:         end for
                                   ϵi ← ϵ · 1{i is Risky and Children(i) ̸⊆ L}
15:         for each j ∈ Children(i) do
                                                                              ϵi
                                 pt (j | i) ← (1 − ϵi )Ptexp (j | i) +
                                                                         |Children(i)|

16:          end for
17:          Sample jt ∼ pt (· | i)
18:          Append (i, jt ) to Pt
19:          Πt (jt ) ← Πt (i) pt (jt | i)
20:          i ← jt
21:      end while
22:      Let sampled leaf be ℓt ← i. Observe terminal cost ct .
23:      2. Hybrid Update
24:      if ℓt ∈ Lshr then                                                                     ▷ Shared branch
25:          valold ← exp(ηθℓt )
26:          θℓt ← θℓt − Πtc(ℓt t )
27:          ∆t ← exp(ηθℓt ) − valold
28:          Back-propagation: Propagate ∆t to Sshr along sampled ancestors:

                       Sshr [u, v] ← Sshr [u, v] + ∆t ,       ∀(u, v) ∈ Pt such that ℓt ∈ L(v)

29:       else                                                                          ▷ Unshared branch
30:          Back-propagation: Propagate multiplicative updates to Sunshr along sampled risky ances-
      tors:
                                                                       
                                                             ct
               Sunshr [u, v] ← Sunshr [u, v] · exp −η                     , ∀(u, v) ∈ Pt with u risky
                                                      Πt (u) pt (v | u)

31:    end if
32: end for




                                                          2
```

## Page 3

```text
Algorithm 2 Full-Share Algorithm (Recursive Aggregation)
 1: Input: Tree T , learning rate η
 2: Initialize:
 3:   θℓ ← 0, ∀ℓP∈ L.
 4:   W [u] ← ℓ∈subtree(u) exp(η · θℓ ), ∀u ∈ T .
 5: for each round t = 1, . . . , T do
 6:    Selection:
 7:    i ← root
 8:    while i is not a leaf do
 9:        Select child j from Children(i) with probability:
                                               P
                                                ℓ∈subtree(j) exp(η · θℓ )    W [j]
                                     P (j|i) = P                           =
                                                 ℓ∈subtree(i) exp(η · θℓ )   W [i]

10:         i←j
11:     end while
12:     Let leaf ℓt ← i.
13:     Observe cost ct at leaf ℓt .
14:     Update:
15:     θold ← θℓt
16:     θℓt ← θℓt − Πtc(ℓt t )
17:     Propagate Change:
18:     Calculate weight change at leaf:

                                          ∆ = exp(η · θℓt ) − exp(η · θold )

19:     curr ← ℓt
20:    while curr ̸= root do
21:       W [curr] ← W [curr] + ∆
22:       curr ← P arent(curr)
23:    end while
24:    W [root] ← W [root] + ∆
25: end for




                                                        3
```

## Page 4

```text
Definitions
For any risky non-leaf node i, let Ci be the set of its children with di = |Ci |. Let Πt (i) denote the
probability that node i is reached in round t. For any child j ∈ Ci , we define ρj,t as the conditional
probability that a leaf selected from the subtree of j is an unshared leaf:

                                        ρj,t := P(leaf ∈ Lunshr | Jt = j).

We define the Effective Unshared Mass of node i at round t as
                                               X
                                      Λt (i) =     ρj,t .                                                (1)
                                                          j∈Ci

   We also define the local exploration rate at node i as
                                             (
                                               0, if Ci ⊆ L,
                                       ϵi :=                                                             (2)
                                               ϵ, otherwise.


Lemma PS-2: Moments of the Estimator at a Risky Node
Lemma 1 (Exact Second Moment Bound at a Risky Node). For any risky non-leaf node i, let z[j, t]
be the estimated cost for child j constructed by the PS algorithm. Let Ht−1 be the history up to round
t − 1. Then the conditional weighted second moment of the estimators satisfies:
                   X                               X                               1     X
                          pt (j | i) Et z[j, t]2 ≤
                                               
                                                     pt (j | i) (1 − ρj,t ) +         2
                                                                                              ρj,t .     (3)
                                                                                Πt (i)
                   j∈Ci                           j∈Ci                                   j∈Ci
                                                  |           {z         }      |       {z       }
                                                         Shared Mass            Unshared Variance


Lemma PS-3: Equivalence of Virtual and Effective Expected
Costs
Lemma 2 (Expectation Bridge). Let ỹ[k, t] ≜ Et [z[k, t] | Ht−1 ] be the expected estimator value
(effective cost) for child k. Then the expected virtual loss (used by Hedge) satisfies:
                                                          
                                   X
                              Et     pt (j|i)z[j, t] Ht−1  = Ek∼pt (·|i) [ỹ[k, t]] .        (4)
                                    j∈Ci



Theorem PS-1: Single-Layer Regret Bound at a Risky Node
Theorem 1 (Single-Layer Regret of PS at a Risky Node). For any risky non-leaf node i, the cumulative
regret with respect to the optimal child j ∗ satisfies
            T               T                               T
                                                                                                      !
          X                X                      ln di
                                                           X          X                       Λ t (i)
               E[ct (i)] −     ct (j ∗ ) ≤ ϵi T +       +η     Πt (i)   pt (k|i)(1 − ρk,t ) +           . (5)
           t=1             t=1
                                                    η      t=1
                                                                                              Πt (i)
                                                                       k∈Ci




                                                          4
```

## Page 5

```text
A      Proofs
A.1     Proof of Lemma 1
Proof. We analyze the conditional second moment Et [z[j, t]2 ] according to whether the selected leaf inside
the subtree of child j is shared or unshared. Recall that

                                            ρj,t = P(leaf ∈ Lunshr | Jt = j).

    For a fixed child j ∈ Ci ,

                 Et [z[j, t]2 ] = (1 − ρj,t ) · Et [z[j, t]2 | Shared] + ρj,t · Et [z[j, t]2 | Unshared]
                                                                    "                     2            #
                                                       2               y[j, t]I(Jt = j)
                                = (1 − ρj,t ) · y[j, t] + ρj,t · Et                            Unshared
                                                                         pt (j|i)Πt (i)
                                                                    1
                              ≤ (1 − ρj,t ) · 1 + ρj,t ·                      Et [I(Jt = j)]
                                                            (pt (j|i)Πt (i))2
                                                       ρj,t
                              = (1 − ρj,t ) +                    ,
                                                 pt (j|i)Πt (i)2

where we used y[j, t] ≤ 1 and Et [I(Jt = j)] = pt (j|i).
   Multiplying both sides by pt (j|i) and summing over j ∈ Ci gives
                        X                                X                                  1    X
                              pt (j|i)Et [z[j, t]2 ] ≤          pt (j|i)(1 − ρj,t ) +          2
                                                                                                   ρj,t .
                                                                                         Πt (i)
                       j∈Ci                              j∈Ci                                     j∈Ci

This proves the claim.

A.2     Proof of Lemma 2
Proof. By linearity of expectation,
                                                   
                                   X                    X
                              Et    pt (j|i)z[j, t] =   pt (j|i)Et [z[j, t]]
                                        j∈Ci                           j∈Ci
                                                                       X
                                                                   =          pt (j|i)ỹ[j, t].
                                                                       j∈Ci

By definition, the right-hand side is exactly Ek∼pt (·|i) [ỹ[k, t]].
   To make ỹ[k, t] explicit, distinguish two cases:

    • Case 1: Shared subtree under child k. Then the estimator is deterministic:

                                                  ỹ[k, t] = Et [z[k, t]] = y[k, t].

    • Case 2: Unshared subtree under child k. The estimator is IPS:
                                                                 y[k, t]
                                                z[k, t] =                   I(Jt = k).
                                                             pt (k|i)Πt (i)

      Hence
                                                                                  y[k, t]      y[k, t]
                                  ỹ[k, t] = Et [z[k, t]] = pt (k|i) ·                       =         .
                                                                              pt (k|i)Πt (i)   Πt (i)

Therefore the bridge identity holds.




                                                                  5
```

## Page 6

```text
A.3    Proof of Theorem 1
Proof. Apply the standard Hedge regret bound to the estimator sequence {z[j, t]}j∈Ci ,t≤T :
                    T X                        T                          T X
                    X                          X
                                                      ∗         ln di    X
                               pt (j|i)z[j, t] ≤     z[j , t] +       +η      pt (j|i)z[j, t]2 .       (6)
                    t=1 j∈Ci                     t=1
                                                                  η      t=1 j∈Ci

   Taking total expectation over the event that node i is reached, Lemma 1 gives
                                                                                
                   XT X                         T
                                                X               X
                E         pt (j|i)z[j, t]2  =   Πt (i) · Et    pt (j|i)z[j, t]2 
                     t=1 j∈Ci                      t=1             j∈Ci
                                                                                                  
                                                   T
                                                   X              X                      Λ  t (i)
                                                ≤     Πt (i)      pt (j|i)(1 − ρj,t ) +         2
                                                                                                   
                                                  t=1
                                                                                         Π t (i)
                                                              j∈Ci
                                                                                                
                                                   T
                                                  X            X                         Λt  (i)
                                                =     Πt (i)      pt (j|i)(1 − ρj,t ) +         .
                                                  t=1
                                                                                         Π t (i)
                                                               j∈Ci


   Combining the above inequality with Lemma 2, and noting that the exploration mixture contributes
an additive ϵi T term, we obtain
           T               T                             T
                                                                                                  !
          X               X
                                   ∗           ln di    X          X                       Λt (i)
              E[ct (i)] −     ct (j ) ≤ ϵi T +       +η     Πt (i)   pt (k|i)(1 − ρk,t ) +          ,
          t=1             t=1
                                                 η      t=1
                                                                                           Πt (i)
                                                                      k∈Ci

which proves (5).




                                                          6
```

## Page 7

```text
B      Recursive Root Regret Bound for Risky-PS
Definitions
For any node i, define the cumulative expected loss of the algorithm starting from i by
                                                       T
                                                       X
                                        Yη [i, T ] ≜         E[ yη [i, t] ].
                                                       t=1

   Let Y∗ [i, T ] denote the cumulative loss of the optimal stationary policy starting from node i, defined
recursively as                               (PT
                                                 t=1 ct (i),      if i ∈ L,
                                Y∗ [i, T ] =
                                              minj∈Ci Y∗ [j, T ], otherwise.
    For any non-leaf node i, define its local branching factor and optimal-child set by

                                 di ≜ |Ci |,       Ci∗ ≜ arg min Y∗ [j, T ].
                                                                   j∈Ci

    We further define the dangerous-child count of node i as

                             DU (i) ≜ |{j ∈ Ci : subtree(j) ∩ Lunshr ̸= ∅}| .

    To make the risky depth precise, define the risky height of each node i recursively by
                                   
                                   
                                    0,               if i is safe,
                                   
                                   
                          R(i) ≜ 1,                   if i is risky and Ci ⊆ L,
                                   
                                   1 + max R(j), if i is risky and C ̸⊆ L.
                                   
                                   
                                                                               i
                                          j∈Ci

We then define the risky skeleton depth of the whole tree as

                                                 R ≜ R(r),

where r is the root. Equivalently, R is the maximum number of risky nodes that can appear on any
root-to-leaf path before the path enters a fully safe suffix.
   For risky nodes, we use the R-adjusted parameters
                                                         (
                             risk        R
                                      − R+1
                                                           0,           Ci ⊆ L,
                            ηi    = T       ,     ϵi =             1
                                                                − R+1
                                                           di T       , Ci ̸⊆ L.

    For a safe node i, let
                                                 Ni ≜ |L(i)|
be the number of leaves in the safe subtree rooted at i. We upper bound its local contribution by the
standard Ni -armed Exp3 rate over at most T rounds, using
                                                   r
                                            shr      ln Ni
                                           ηi =            .
                                                      T Ni

Lemma RR-1 (Safe-Subtree Exp3 Equivalence).                      For any safe node i, let

                                                 Ni ≜ |L(i)|

denote the number of leaves in the safe subtree rooted at i. Then the recursive selection rule of the
full-share algorithm restricted to the subtree of i is equivalent to running an Ni -armed Exp3 forecaster
directly on the leaf set L(i).
    More precisely, if the leaf weights are

                                    wt (ℓ) ≜ exp ηishr θℓ,t ,
                                                           
                                                                ℓ ∈ L(i),


                                                       7
```

## Page 8

```text
and the recursive aggregate at any node u is
                                                               X
                                            Wt [u] ≜                      wt (ℓ),
                                                           ℓ∈subtree(u)


then for any leaf ℓ ∈ L(i), the probability that the recursive aggregation procedure starting from node i
selects ℓ at round t is exactly

                                                    wt (ℓ)             exp(ηishr θℓ,t )
                           Pt (ℓ | i) = P                     ′
                                                                 =P                shr ′
                                                                                           .          (7)
                                                ℓ′ ∈L(i) wt (ℓ )   ℓ′ ∈L(i) exp(ηi θℓ ,t )

   Consequently, viewing the safe subtree as an Ni -armed Exp3 problem over at most T rounds, we
obtain                                                     p
                                Yη [i, T ] ≤ Y∗ [i, T ] + 2 T Ni ln Ni .                      (8)

Proof. Since node i is safe, every leaf in subtree(i) is shared. The recursive aggregation algorithm
selects a child j of an internal node u with probability

                                                                    Wt [j]
                                                   Pt (j | u) =            .
                                                                    Wt [u]

Fix any leaf ℓ ∈ L(i), and let
                                          i = u0 → u1 → · · · → um = ℓ
be the unique path from node i to leaf ℓ. The probability that the recursive procedure starting at i
finally reaches ℓ is the product of the transition probabilities along this path:
                                                m−1                           m−1
                                                Y                             Y   Wt [ur+1 ]
                                 Pt (ℓ | i) =          Pt (ur+1 | ur ) =                     .
                                                 r=0                          r=0
                                                                                   Wt [ur ]

This product telescopes:
                                                                   Wt [ℓ]
                                                    Pt (ℓ | i) =          .
                                                                   Wt [i]
Since a leaf has no descendants except itself,

                                          Wt [ℓ] = wt (ℓ) = exp(ηishr θℓ,t ),

and since i is the root of the current safe subtree,
                                          X               X
                                Wt [i] =       wt (ℓ′ ) =   exp(ηishr θℓ′ ,t ).
                                           ℓ′ ∈L(i)                ℓ′ ∈L(i)

Therefore,
                                                               exp(ηishr θℓ,t )
                                        Pt (ℓ | i) = P                     shr ′
                                                                                   ,
                                                           ℓ′ ∈L(i) exp(ηi θℓ ,t )

which is exactly the arm-selection probability of an exponential-weights/Exp3 forecaster run directly on
the leaf set L(i).
   Hence, the safe subtree rooted at i can be viewed as an Ni -armed adversarial bandit problem. Since
node i can be visited in at most T rounds, a valid upper bound is obtained by applying the standard
Exp3 regret bound over horizon T with learning rate
                                                    r
                                              shr      ln Ni
                                             ηi =            .
                                                       T Ni
This gives                                                       p
                                      Yη [i, T ] − Y∗ [i, T ] ≤ 2 T Ni ln Ni ,
which proves the claim.                                                                                □




                                                              8
```

## Page 9

```text
Lemma RR-2 (R-Adjusted Local Risky Bounds). For any risky node i, the single-layer regret
satisfies the following fixed-parameter bounds:
                                                             R
                Yη [i, T ] ≤ min Yη [j, T ] + DU (i) + ln di T R+1 , if Ci ⊆ L,       (8)
                           j∈Ci
                                                                 R
               Yη [i, T ] ≤ min Yη [j, T ] + di + DU (i) + ln di T R+1 ,                  if Ci ̸⊆ L.    (9)
                           j∈Ci


Proof. This follows from the single-layer Partial-Share bound (Theorem PS-1) after substituting the
R-adjusted risky parameters
                                                   (
                             risk     R
                                   − R+1             0,           Ci ⊆ L,
                            ηi = T       ,    ϵi =           1
                                                          − R+1
                                                     di T       , Ci ̸⊆ L,

together with |Ci | = di and the structural bound Λt (i) ≤ DU (i). When Ci ⊆ L, the exploration term
                                                                                                    R
vanishes, yielding (8). Otherwise, the extra exploration contribution produces the additional di T R+1
term, yielding (9).                                                                                  □

Theorem RR-1 (Recursive Root Regret Bound of Risky-PS). Define recursively, for every
node i,          
                 
                 
                  0,                                     i ∈ L,
                 
                 
                   √
                 2 T Ni ln Ni ,                          i is safe,
                 
                 
          ∆(i) ≜                     R                                           (10)
                 
                   DU (i) + ln di T R+1 ,                i is risky and Ci ⊆ L,
                 
                 
                 
                                      R
                  di + DU (i) + ln di T R+1 + min∗ ∆(j), i is risky and Ci ̸⊆ L.
                 
                 
                                                                  j∈Ci

   Then for every node i,
                                            Yη [i, T ] ≤ Y∗ [i, T ] + ∆(i).                             (11)
In particular, for the root node r,

                                  Regroot (T ) ≜ Yη [r, T ] − Y∗ [r, T ] ≤ ∆(r).                        (12)

Proof. We proceed by structural induction on the subtree rooted at i.
  Base case: If i ∈ L, then by definition

                                                   Yη [i, T ] = Y∗ [i, T ],

and since ∆(i) = 0, (11) holds.
   Safe case: If i is safe, then (11) follows directly from Lemma RR-1:
                                                       p
                            Yη [i, T ] ≤ Y∗ [i, T ] + 2 T Ni ln Ni = Y∗ [i, T ] + ∆(i).

   Risky leaf-parent case: If i is risky and Ci ⊆ L, then Lemma RR-2 gives
                                                                          R
                             Yη [i, T ] ≤ min Yη [j, T ] + DU (i) + ln di T R+1 .
                                            j∈Ci

Since all children are leaves,
                                     min Yη [j, T ] = min Y∗ [j, T ] = Y∗ [i, T ].
                                     j∈Ci                j∈Ci

Therefore,
                                                                 R
                        Yη [i, T ] ≤ Y∗ [i, T ] + DU (i) + ln di T R+1 = Y∗ [i, T ] + ∆(i).
   Risky non-leaf case: Suppose now that i is risky and Ci ̸⊆ L. By Lemma RR-2,
                                                                           R
                         Yη [i, T ] ≤ min Yη [j, T ] + di + DU (i) + ln di T R+1 .                      (13)
                                        j∈Ci

Take any optimal child j ∗ ∈ Ci∗ . By definition of Ci∗ ,

                                                Y∗ [j ∗ , T ] = Y∗ [i, T ].                             (14)


                                                              9
```

## Page 10

```text
By the induction hypothesis applied to node j ∗ ,

                                           Yη [j ∗ , T ] ≤ Y∗ [j ∗ , T ] + ∆(j ∗ ).                      (15)

Also,
                                                min Yη [j, T ] ≤ Yη [j ∗ , T ].                          (16)
                                                j∈Ci

Substituting (15) and (16) into (13), and then using (14), yields
                                                                                   R
                           Yη [i, T ] ≤ Y∗ [i, T ] + ∆(j ∗ ) + di + DU (i) + ln di T R+1 .

Since this holds for any j ∗ ∈ Ci∗ , taking the minimum over Ci∗ gives
                                                                          R
                Yη [i, T ] ≤ Y∗ [i, T ] + min∗ ∆(j) + di + DU (i) + ln di T R+1 = Y∗ [i, T ] + ∆(i).
                                      j∈Ci


Thus the induction closes, and (11) holds for all nodes i. Applying it to the root proves (12).             □

Sanity checks.

    • All-share. If the whole tree is safe, then r is safe and
                                                      p
                                             ∆(r) = 2 T |L| ln |L|,

        which is independent of the depth L.
    • All-unshare. If all leaves are unshared, then every internal node is risky and DU (i) = di for every
      internal node i. Hence the recursive expansion yields
                                                                                  
                                  X                 X                 X                 L
                      ∆(r) =          di +                    di +          ln di  T L+1 ,
                                   i∈Pint (r)          i∈Pint (r)\{ileaf-parent }     i∈Pint (r)


        where Pint (r) denotes the set of internal nodes on the optimal root-to-leaf path, and ileaf-parent is
        its last internal node. In the homogeneous case di ≡ D, this simplifies to
                                                                    L
                                          ∆(r) = (2L − 1)D + L ln D T L+1 ,

        which exactly recovers the original worst-case endpoint.


C       A Structure-Dependent Lower Bound for Risky Partial-Share
In this section, we establish a regret lower bound for a class of time-homogeneous oracle policies under a
tree whose longest risky prefix has length R. This lower bound shows that even when all side branches
are fully shared safe subtrees, the need to educate a deepest risky learner still induces a polynomial regret
barrier.

Risky-prefix system. We consider a tree constructed as follows. There are R risky internal nodes,
denoted by v1 , v2 , . . . , vR , where v1 is the root. For each i ≤ R − 1, node vi has two children: (i) a
fully-shared safe subtree Si , and (ii) the next risky node vi+1 . The node vR has two leaf children u and
u′ , both of which are unshared. By construction, once a path enters any Si , all remaining descendants
are shared and hence safe; therefore the longest risky prefix is exactly R.
     Each visited leaf incurs a Bernoulli cost. Fix a small positive constant δ < 1/2R . Let

                                                                1 − 2R δ
                                                        p∗ ≜             .
                                                                   2
We set one of pu , pu′ to p∗ and the other to

                                                            1 + 2R δ
                                                                     .
                                                               2


                                                                10
```

## Page 11

```text
For each safe subtree Si , all of its leaves are shared and have the same Bernoulli mean

                                      1 − (2R − 2i−1 )δ
                             pSi ≜                      ,        i = 1, 2, . . . , R − 1.
                                              2
Hence, for any t,
                                                E[y[Si , t]] = pSi ,
regardless of which shared leaf inside Si is reached. In particular,
                                                                 δ
                                                  pS1 − p∗ =       .
                                                                 2
   The regret of the system is
                           T                              T 
                                                                            1 − 2R δ
                           X                             X                           
                                 E[y[v1 , t]] − p∗ T =       E[y[v1 , t]] −            .                (7)
                           t=1                           t=1
                                                                               2

    We consider the following policy class: the deepest risky node vR may employ an arbitrary bandit
learning algorithm with feedback only when it receives a job, whereas nodes v1 , . . . , vR−1 employ time-
homogeneous oracle policies in the sense of Definition 2. That is, before deciding between Si and vi+1 ,
node vi knows the expected costs of its two children, and if the absolute gap equals ζ, it forwards the job
to the higher-cost child with probability Pi (ζ) and to the other child with probability 1 − Pi (ζ), where
Pi (·) is an arbitrary decreasing function.
    We note that the fully-shared structure of each Si is precisely where the share assumption enters the
lower bound: the side branch is safe, and its expected cost is completely revealed to its parent. Thus,
the only nontrivial learning challenge lies on the risky prefix v1 → v2 → · · · → vR .
Lemma 3. There exists a positive integer Nδ such that, for all t with NR (t) < Nδ ,

                                                         1 − (2R − 2R−1 )δ
                                        E[y[vR , t]] >                     ,
                                                                 2
where NR (t) denotes the number of jobs that node vR has received from its parent up to time t.
Proof. This is the same bottom-level two-armed bandit hardness argument as in Lemma 5 of the original
proof, with L replaced by R. Since node vR can only learn the means of u and u′ from the rounds in
which it actually receives a job, there exists a finite threshold Nδ such that before receiving Nδ samples,
its expected cost cannot be reduced below 1 − (2R − 2R−1 )δ /2.
Theorem 2. For the risky-prefix system above, the regret is
                                               R−1 
                                            Ω T R

for any bandit learning policy employed by node vR and any time-homogeneous oracle policies employed
by nodes v1 , . . . , vR−1 . Consequently, the same lower bound applies a fortiori to Risky Partial-Share,
which has no more information than this oracle class.
Proof. Let Tδ be the time at which NR (t) = Nδ . By Lemma 3,

                                             1 − (2R − 2R−1 )δ
                            E[y[vR , t]] >                              for all t < Tδ .
                                                     2
    We first study the system behavior before time Tδ . Consider the forwarding decision of node vR−1
at any time t < Tδ . Node vR−1 has two children: one is the shared safe subtree SR−1 with

                                                               1 − (2R − 2R−2 )δ
                                 E[y[SR−1 , t]] = pSR−1 =                        ,
                                                                       2
and the other is the non-leaf node vR with

                                             1 − (2R − 2R−1 )δ
                            E[y[vR , t]] >                     = pSR−1 + 2R−3 δ.
                                                     2


                                                          11
```

## Page 12

```text
By Assumption A2, the probability that node vR−1 selects the higher-cost child vR is at most

                                            qR−1 ≜ PR−1 (2R−3 δ).

We also have
                                                      1 − (2R − 2R−2 )δ
                                E[y[vR−1 , t]] ≥ pSR−1 =                .
                                                               2
   We further analyze the forwarding decision of node vi for i < R − 1 at any time t < Tδ . Using a
simple induction argument, it can be shown that the probability that node vi selects node vi+1 is at most

                                                qi ≜ Pi (2i−2 δ),

and
                                                            1 − (2R − 2i−1 )δ
                                    E[y[vi , t]] ≥ pSi =                      .
                                                                    2
Therefore, at any time t < Tδ , we have

                                                           1 − 2R δ  δ
                                          E[y[v1 , t]] −            ≥ .                                   (8)
                                                              2      2
   Moreover, since node vR can only receive a job if, for each i ≤ R − 1, node vi selects node vi+1 , which
happens with probability at most qi , we have
                                                        Nδ
                                              E[Tδ ] ≥ QR−1                .                              (9)
                                                                i=1   qi

   Next, we analyze the system behavior after time Tδ . For any time t > Tδ ,

                                                                               1 − 2R δ
                                   E[y[vR , t]] ≥ min{pu , pu′ } =                      .
                                                                                  2
Consider the forwarding decision of node vR−1 . Since

                                                1 − (2R − 2R−2 )δ
                     E[y[SR−1 , t]] = pSR−1 =                     ≤ E[y[vR , t]] + 2R−3 δ,
                                                        2
the probability that node vR−1 selects node vR is at most

                                       1 − PR−1 (2R−3 δ) = 1 − qR−1 .

Using a simple induction argument, we can further show that the probability that node vi selects node
vi+1 is at most 1 − qi , for all i ≤ R − 1. Hence, the probability that node vR receives a job is at most
                                                  R−1
                                                  Y
                                                        (1 − qi ).
                                                  i=1

If node vR does not receive a job, which happens with probability at least
                                                     R−1
                                                     Y
                                                1−         (1 − qi ),
                                                     i=1

then the job must enter one of the safe subtrees S1 , . . . , SR−1 , and the expected cost is at least

                                                                 1 − (2R − 1)δ
                                      min     pSi = pS1 =                      .
                                    1≤i≤R−1                            2
We then have, at any time t > Tδ ,
                                                                      R−1
                                                                                           !
                                              1 − 2R δ                Y                        δ
                               E[y[v1 , t]] −          ≥        1−             (1 − qi )         .       (10)
                                                 2                     i=1
                                                                                               2



                                                           12
```

## Page 13

```text
    Combining (8), (9), and (10), we obtain the following regret bound:
          T 
                                            "T          T        R−1
                                                                              ! #
                             1 − 2R δ          δ
                                      
         X                                   X    δ     X         Y            δ
              E[y[v1 , t]] −            ≥E          +         1−     (1 − qi )
         t=1
                                2            t=1
                                                  2              i=1
                                                                               2
                                                      t=Tδ +1
                                            "                           !       R−1
                                                                                             !#
                                          δ       Nδ              Nδ            Y
                                        ≥     QR−1 + T − QR−1                1−     (1 − qi )   .                   (11)
                                          2      i=1 qi          i=1 qi         i=1

    It is then straightforward to show that
                       "                           !    R−1
                                                                     !#
                     δ     Nδ                Nδ         Y                  R−1 
                         QR−1 + T − QR−1             1−     (1 − qi )   =Ω T R .
                     2    i=1 qi            i=1 qi      i=1

Moreover, setting                                    
                                         qi = Θ T −1/R ,                   ∀ i ≤ R − 1,
makes                 "                                         !                          !#
                                                                          R−1
                    δ     Nδ                         Nδ                   Y                        R−1 
                        QR−1 +            T − QR−1                   1−          (1 − qi )      =Θ T R .
                    2    i=1 qi                    i=1     qi              i=1
This completes the proof.


D       All-Share Lower Bound via Adversarial Bandit Reduction
Assume the whole tree is all-share, so that the root r is safe. Let
                                                          N ≜ |L(r)|
be the number of leaves. Then the recursive full-share selection rule induces an N -armed adversarial
bandit policy on the leaf set L(r). Consequently, there exists a universal constant c0 > 0 such that for any
δ ∈ (0, 1) satisfying the conditions of Theorem 17.4, there exists a cost sequence {ct (ℓ)}t≤T,ℓ∈L(r) ⊂ [0, 1]
such that                                            r              !
                                                                  1
                                  P Regroot (T ) ≥ c0 T N log          ≥ δ.                           (LB-1)
                                                                 2δ
In particular, there exists a universal constant c1 > 0 such that
                                                              √
                                      sup E[Regroot (T )] ≥ c1 T N .                                              (LB-2)
                                            c1:T


Proof.    For any leaf ℓ ∈ L(r), let
                                            r = u0 → u1 → · · · → um = ℓ
be the unique root-to-leaf path. Under the recursive aggregation rule,
                                                                         Wt [us+1 ]
                                              Pt (us+1 | us ) =                     ,
                                                                          Wt [us ]
hence the probability of selecting leaf ℓ at round t is
                                            m−1
                                            Y   Wt [us+1 ]   Wt [ℓ]       wt (ℓ)
                             Pt (ℓ | r) =                  =        =P              ′
                                                                                       .
                                            s=0
                                                 Wt [us ]    Wt [r]   ℓ′ ∈L(r) wt (ℓ )

Therefore, the recursive full-share algorithm on the all-share tree is equivalent, at the leaf level, to an
N -armed bandit policy over L(r).
    Now define rewards by
                                      xtℓ ≜ 1 − ct (ℓ),    ℓ ∈ L(r).
Then the reward regret and the cost regret coincide:
                             T
                             X                       T
                                                     X                            T
                                                                                  X
                     max           (xtℓ − xtAt ) =         ct (At ) − min                ct (ℓ) = Regroot (T ).
                    ℓ∈L(r)                                               ℓ∈L(r)
                             t=1                     t=1                           t=1

Applying Theorem 17.4 with k = N and n = T yields the high-probability lower bound (LB-1). Choosing
a fixed constant δ (e.g. δ = 1/4) gives (LB-2).                                                   □


                                                                    13
```
