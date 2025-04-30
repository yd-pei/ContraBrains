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

## Implementation

Note: Because Gym-Retro already wires up the ROM’s transition and screen-capture code, and because PPO doesn’t require an explicit model of those processes, I don't need to implement state transition and observation for my project.

So far(April 29th), I've trained ContraForce on torch and gym-retro for about a few million frames. The results weren't particularly good, so I decided to train the same algorithm on Contra (ContraForce's previous game) as a possible alternative. This is how much I have done on two games:
### ContraForce
1. The original image is directly used as input, the game Score is used as output, and PPO training is used. In this game, however, the agent was stuck at a "local maximum". Enemies in ContraForce respawn a few seconds after being killed, and agents use this mechanism to keep trying to score points in the same place without moving forward.
2. To overcome the above situation, I reward the forward movement (right direction action) directly, but the agent will go right regardless of death. It also leads to irrational strategies.
3. After that, I tried to add the optical flow function to detect whether the agent moved or not, but the implementation was difficult and the effect was not good.
4. (Currently) Try to read the location information in memory to detect if the agent is making too many kills in the same location, and if so, reduce the reward in this part to encourage the agent to move forward. Although this violates the original assumption of only using images as input, existing methods do not seem to have a good solution.
### Contra
1. (Currently) I plan to directly use images and scores as observations and rewards, because the Contra game does not have the problems encountered in ContraForce, so the training process should be smooth.
## Appendix and Reference
### Project Scope
<a href="https://pics1.obs.myhuaweicloud.com/GraduateGW/Proposal_YidingPei.pdf" target="_blank">Click to Download</a>

