# ContraBrains

The Design and Implementation of a ContraForce Agent based on Deep Reinforcement Learning

## Install

## Solution
### State Space Description
#### Informal Description
We use Gym Retro as the simulated environment for the game. In this environment, state space is the set of all possible emulator states, which doesn't have much discussion value.

Here, we focus on observation space. 
We use vision directly as the input: a $256\times 256$ pixels RGB image.

For simplicity, we down-sample the frame to $84\times 84$ pixels, convert it to grayscale and normalize intensity to $[0,1]$.

Besides, since we want to learn  temporal dependences, we stack the last 4 frames.

Also, we want to know how many scores we've earned and how many lives we have. We can use OCR to extract life and score information at specific locations in the screen. Gym Retro also provides apis to read life and game score information. To simplify the process, we choose to read remaining lives and scores directly from the API.

#### Formal Description
We model the game as a Partially Observable MDP:
$$M = \left\langle S, A, O, T, R\right\rangle$$
, where 
Symbol|Definition 
-|-
$S$|The set of all possible emulator states
$A=\{a_{1},…,a_{6}\}$| Discrete action set :$\{↑, ↓, ←, →, J,K\}$.
$O = [0,1]^{K\times H\times W}$ | Observtion space. Each observation $o$​ is a stack of the $K=4$ most recent grayscale frames, where $H=W=84$.
$T:S\times A \to \Delta(S)$ | Transitions $T(s' \mid s,a)$ is determined by the game emulator written determistically in ROM code. For us and the agent, it is a black box.
$R:S\times A \to \mathbb{R}$ | Reward function

## Appendix and Reference
### Project Scope
<a href="https://pics1.obs.myhuaweicloud.com/GraduateGW/Proposal_YidingPei.pdf" target="_blank">Click to Download</a>

