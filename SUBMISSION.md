# HW6 Submission --- Grid World and FrozenLake

## AI-use declaration

I used ChatGPT as an AI assistant to help understand the assignment
requirements, review the Bellman equation calculations, debug and
explain the Python implementation, and organize the results in this
submission file.

The implementation was run locally. The reported experiment results,
value maps, policy maps, convergence values, and rule-check results come
from my program output. I reviewed the generated results before
including them here.

## Setup and reproduction

### Requirements

The project uses Python 3 and the following Python packages:

-   `gymnasium` --- FrozenLake environment for Part B.
-   `numpy` --- numerical calculations, random number generation, and
    Value Iteration.
-   `pypandoc` is not required to run the assignment; it was only used
    to prepare this Markdown file.

The exact installed package versions should be recorded from the project
virtual environment. To display them, run:

``` bash
python --version
python -m pip freeze
```

To save the installed dependencies:

``` bash
python -m pip freeze > requirements.txt
```

### Installation

From the project root, create and activate a virtual environment.

Windows PowerShell:

``` powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install gymnasium numpy
```

Windows Command Prompt:

``` cmd
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install gymnasium numpy
```

### Run commands

Run each part from the project root with the virtual environment
activated:

``` bash
python part_b.py
python part_c.py
```

The programs save their experiment results and supporting text files in
the `outputs/` directory.

------------------------------------------------------------------------

## Part A --- Paper calculations

**Paper scan/photo:**
`[Add the link or file path to the scan/photo of the handwritten calculations here.]`

The original grid is:

``` text
. M . G
. # I H
S . . .
```

The discount factor is (`\gamma `{=tex}= 0.9), and the initial values
are (V_0(s)=0).

The rewards are:

-   Start `S` and ordinary floor `.`: -1
-   Mud `M`: -3
-   Ice `I`: -1
-   Goal `G`: +10
-   Hole `H`: -10
-   Wall `#`: not a state

When the agent moves from an ice tile, the intended direction has
probability 0.8, and each perpendicular direction has probability 0.1.
Movement from other nonterminal tiles is deterministic. If movement
would leave the grid or enter a wall, the agent stays in the current
tile. The reward is based on the tile where the agent ends up. Goal and
hole are terminal states.

### V1 for every nonterminal tile

Using

\[ V_1(s)=`\max`{=tex}*a
`\sum`{=tex}*{s'}P(s'\|s,a)`\left[R(s,a,s')+\gamma V_0(s')\right]`{=tex},
\]

and (V_0(s)=0), the V1 values are:

``` text
-1.00  -1.00  10.00   0.00
-1.00    #    -1.00   0.00
-1.00  -1.00  -1.00  -1.00
```

The goal and hole are terminal, so their displayed values remain 0. The
wall is not a state.

### Requested V2 calculations

The second iteration is calculated with:

\[
Q_2(s,a)=`\sum`{=tex}\_{s'}P(s'\|s,a)`\left[R(s,a,s')+0.9V_1(s')\right]`{=tex},
`\qquad`{=tex} V_2(s)=`\max`{=tex}\_a Q_2(s,a). \]

**Mud at (1,2)**

  Action        Q2
  -------- -------
  Up         -3.90
  Right       8.00
  Down       -3.90
  Left       -1.90

Therefore, (V_2=8.00), and the best action is **Right (→)**.

**Ordinary floor at (1,3)**

  Action        Q2
  -------- -------
  Up          8.00
  Right      10.00
  Down       -1.90
  Left       -3.90

Therefore, (V_2=10.00), and the best action is **Right (→)**.

**Ice at (2,3)**

  Action        Q2
  -------- -------
  Up          5.21
  Right      -7.39
  Down       -2.71
  Left       -0.91

Therefore, (V_2=5.21), and the best action is **Up (↑)**.

There are no ties in these three requested calculations.

------------------------------------------------------------------------

## Part B --- FrozenLake

### Fixed policy, written before the experiments

The action encoding used by the FrozenLake environment is:

-   `0` = Left
-   `1` = Down
-   `2` = Right
-   `3` = Up

The fixed policy is:

``` python
FIXED_POLICY = {
    0: 1,
    1: 2,
    2: 0,
    3: 0,
    4: 1,
    6: 1,
    8: 2,
    9: 2,
    10: 1,
    13: 2,
    14: 2
}
```

The policy was defined before running the experiments and was not
updated during the episodes. It follows a route toward the goal while
avoiding the known hole states where possible. In the non-slippery
environment, the intended route is:

``` text
0 → 4 → 8 → 9 → 10 → 14 → 15
```

This route reaches the goal in six steps when movement is deterministic.

### Results

Each condition was tested in three batches of 50 episodes, for a total
of 150 episodes per condition.

  --------------------------------------------------------------------------------
  Slippery?   Policy       Batch Goals /50 Holes /50      Time Success %      Mean
                            seed                        limits               steps
                                                           /50           
  ----------- -------- --------- --------- --------- --------- --------- ---------
  False       Random          11         0        50         0        0%      6.46

  False       Random          22         1        49         0        2%      8.52

  False       Random          33         0        50         0        0%      7.50

  False       Fixed           11        50         0         0      100%      6.00

  False       Fixed           22        50         0         0      100%      6.00

  False       Fixed           33        50         0         0      100%      6.00

  True        Random          11         0        50         0        0%      8.16

  True        Random          22         2        48         0        4%      8.98

  True        Random          33         2        48         0        4%      7.92

  True        Fixed           11         3        47         0        6%      4.68

  True        Fixed           22         3        47         0        6%      5.46

  True        Fixed           33         0        50         0        0%      4.08
  --------------------------------------------------------------------------------

### Episode results and example traces

The program generated these files:

-   `outputs/part_b_episodes.csv` --- individual episode results.
-   `outputs/part_b_batch_summary.csv` --- batch summaries.
-   `outputs/part_b_rendered_episode.txt` --- rendered example episode.
-   `outputs/part_b_success_trace.txt` --- successful episode trace, if
    one was recorded.
-   `outputs/part_b_failure_trace.txt` --- failure episode trace, if one
    was recorded.

### Explanation of the results

The fixed policy performed very well when `is_slippery=False`. It
reached the goal in all 150 episodes, with an average of 6 steps. The
random policy performed poorly, reaching the goal only once out of 150
episodes.

When `is_slippery=True`, both policies became less reliable because the
agent could move in an unintended direction. The fixed policy reached
the goal in 6 out of 150 episodes, while the random policy succeeded in
4 out of 150 episodes.

The results were consistent across the three batches for the
non-slippery fixed policy. However, the slippery fixed policy had
different results between batches, with success rates of 6%, 6%, and 0%.
This shows how random transitions can affect the outcome even when the
policy stays unchanged.

No episodes reached the 100-step time limit. All episodes ended either
at the goal or in a hole.

A single successful episode is not enough evidence of a good policy
because the result may happen by chance. Repeated experiments provide a
more reliable comparison.

These experiments are not training because neither policy is updated
using rewards or experience. The random policy selects random actions,
while the fixed policy always uses the same predefined action for each
state. The experiments only evaluate their performance.

------------------------------------------------------------------------

## Part C --- Custom solver

### Compare code with paper

The custom solver uses the same original map and reward/transition rules
as Part A. It uses synchronous Value Iteration with
(`\gamma=0.9`{=tex}), starting from zero values.

The solver produced the same V1 values as the paper calculations:

``` text
-1.00  -1.00  10.00   0.00
-1.00    #    -1.00   0.00
-1.00  -1.00  -1.00  -1.00
```

It also produced the same requested V2 values:

  State              Up   Right    Down    Left      V2 Best action
  ------------- ------- ------- ------- ------- ------- -------------
  Mud (1,2)       -3.90    8.00   -3.90   -1.90    8.00 Right
  Floor (1,3)      8.00   10.00   -1.90   -3.90   10.00 Right
  Ice (2,3)        5.21   -7.39   -2.71   -0.91    5.21 Up

There were no discrepancies between the paper calculations and the
custom solver for V1 or the requested V2 values.

### Values, policies and convergence

**Original map --- final values**

``` text
 4.20   8.00  10.00   0.00
 2.78    #     5.82   0.00
 1.54   2.82   4.24   2.82
```

**Original map --- optimal policy**

``` text
 >    >    >    G
 ^    #    ^    H
 >    >    ^    <
```

**Extended map**

The extended map is:

``` text
. M . G
. # I H
S D . W
```

`D` is Sand with reward -2, and `W` is Water with reward -5.

**Extended map --- final values**

``` text
 4.20   8.00  10.00   0.00
 2.78    #     5.82   0.00
 1.50   2.82   4.24   2.82
```

**Extended map --- optimal policy**

``` text
 >    >    >    G
 ^    #    ^    H
 ^    >    ^    <
```

**Convergence results**

  Map              Iterations    Final delta Converged
  -------------- ------------ -------------- -----------
  Original map             12   0.0000002165 Yes
  Extended map             11   0.0000002406 Yes

The stopping threshold was `0.000001`. The algorithm stopped when the
maximum absolute change in state values between two consecutive
iterations became smaller than this threshold.

The policy extraction checks actions in the order Up, Right, Down, Left.
If two actions have equal values, the first action in this order is
kept. No tie-breaking changed the three requested V2 answers.

The program generated the following map files:

-   `outputs/part_c_part_a_maps.txt` --- V1, final values, and policy
    arrows for the original map.
-   `outputs/part_c_extended_map_maps.txt` --- V1, final values, and
    policy arrows for the extended map.

### Simulation results

Each map was tested using three batches of 50 episodes.

  ------------------------------------------------------------------------------
  Map        Batch seed  Goals /50  Holes /50       Time Mean steps         Mean
                                              limits /50              discounted
                                                                          return
  ---------- ---------- ---------- ---------- ---------- ---------- ------------
  Original           11         47          3          0       5.04     2.229704

  Original           22         44          6          0       4.96     1.442384

  Original           33         42          8          0       4.88     0.969992

  Extended           11         50          0          0       5.00     1.502000

  Extended           22         50          0          0       5.00     1.502000

  Extended           33         50          0          0       5.00     1.502000
  ------------------------------------------------------------------------------

The episode and batch data were saved in:

-   `outputs/part_c_episodes.csv`
-   `outputs/part_c_batch_summary.csv`

### Explain your results

The original and extended maps have the same V1 values because the first
Value Iteration update starts from zero values and uses the immediate
rewards. The final values and policy can differ after later iterations
because future rewards become important.

After adding Sand and Water, the optimal action in the bottom-left state
changed from Right to Up. Its final value changed from 1.54 to 1.50. The
other displayed final state values and policy arrows remained the same.

Sand has a reward of -2, while Water has a reward of -5. These negative
rewards make paths that enter those tiles less attractive. The solver
considers both immediate movement rewards and discounted future values,
so it can prefer a route that avoids a costly tile even if that route is
not the shortest in distance.

The original map reached the goal in 133 of 150 simulation episodes
(88.67%). The extended map reached the goal in all 150 episodes (100%).
These results show that the optimal policy for the extended map was more
successful in the tested simulations.

The simulations are stochastic on ice because the agent can move in an
unintended perpendicular direction. Therefore, even when the policy is
fixed, the actual episode path and discounted return can vary. The
computed state values represent expected discounted returns under the
transition model, while the simulations are individual sampled outcomes.

### Rule checks

All five automated rule checks passed:

  Rule check                                Result
  ----------------------------------------- --------
  Boundary behavior                         Passed
  Wall behavior                             Passed
  Ice transition probabilities sum to 1     Passed
  Mud, Sand, and Water rewards              Passed
  Terminal states give no further rewards   Passed

The program also saved the rule-check results in
`outputs/part_c_rule_checks.txt`.

------------------------------------------------------------------------

## Final output files

The following generated files should be present in the `outputs/`
directory:

### Part B

-   `part_b_episodes.csv`
-   `part_b_batch_summary.csv`
-   `part_b_rendered_episode.txt`
-   `part_b_success_trace.txt`
-   `part_b_failure_trace.txt`

### Part C

-   `part_c_episodes.csv`
-   `part_c_batch_summary.csv`
-   `part_c_rule_checks.txt`
-   `part_c_part_a_maps.txt`
-   `part_c_extended_map_maps.txt`

## Before final submission

1.  Add the actual scan/photo of the handwritten Part A calculations and
    replace the placeholder link above.
2.  Check the installed versions with `python --version` and
    `python -m pip freeze`, then save them to `requirements.txt`.
3.  Confirm that all listed output files exist in `outputs/`.
4.  Commit the Part A scan/calculations before the coding work if this
    is required by the assignment's submission rules.
5.  Commit the completed code, outputs, `requirements.txt`, and this
    `SUBMISSION.md`.
