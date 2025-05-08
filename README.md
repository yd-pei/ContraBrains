# ContraBrains

The Design and Implementation of a ContraForce Agent based on Deep Reinforcement Learning
![Contra-Nes](https://www.retrogames.cz/games/022/NES-gameplay.gif)

## Content

- [Intall](#Install)
- [Description and Solution](#Description-and-Solution)
	-[Problem statement](#Problem-statement)
	-[Related solutions to similar problems](#Related-solutions-to-similar-problems)
	-[State Space Description](#State-Space-Description)
        -[Informal Description](#Informal-Description)
        -[Formal Description](#Formal-Description)
	-[Solution Method](#Solution-Method)
        -[PPO](#ppo)
        -[Training Pipeline](#training-pipeline)
- [Result](#Result)

## Install
### Prerequisites
- Linux System (tested on WSL2 Ubuntu 22.04, not supported on Win or OSX)
- Python Intepreter >= 3.9 (tested on python 3.11.11)
- Nvidia GPU
- Cuda >= 11
- conda

### Install dependencies
(May take a few minutes)
(If you can not build stable-retro locally, see [How to install stable-retro?](#How to install stable-retro?))

```
$ git clone https://github.com/yd-pei/ContraBrains.git
$ cd ./ContraBrains
$ bash ./setup.sh
```

### Use examples
May need to activate conda environment before run main.py
```
$ conda deactivate
$ conda activate contra_ppo
```
train
```$ python main.py train```

5 stochastic evaluation (default 5 episodes)
```$ python main.py evaluate```

10 stochastic evaluation
```$ python main.py evaluate -e 10```

1 deterministic evaluation (default 1 episode)
```$ python main.py evaluate --deterministic```



__For videos results, check __ `./evaluation_videos`.



## Description and Solution
### Problem statement

The core objective of this project is to design and implement a deep reinforcement learning based agent apable of completing the first level of the Contra-Nes game. Contra is a 2-dimensional side-scrolling action game originally developed for the NES platform. It features a complex environment that players must defeat opponents with limited lives.

__Note:__ __My initial goal was to play ContraForce, a sequel to the Contra game, but the model didn't perform well due to a few difficulties below, so I decided to port the algorithm to the Contra game. __

1. The reward of the ContraForce game is extremely sparse, and it is impossible to determine whether it is moving to the right from the score, which is also difficult to achieve visually. In contrast, the Contra game directly encapsulates the reward for moving to the right in the Retro environment.
1. ContraForce NPCs regenerate a few seconds after being killed, allowing the character to kill the same unit over and over again to earn bonus points. This leads to a locally optimal solution.

### Related solutions to similar problems

The mainstream solutions for Atari games are basically based on deep reinforcement learning. Here are some existing methods:
1. DQN, which combined Q-learning with convolutional neural networks and experience replay, was initially published to play Atari games with human-level performance at 2015. However, DQN was shown to be prone to overfitting, especially when trained on fixed-level environments.

2. A3C (Asynchronous Advantage Actor-Critic) were then proposed at 2016 to address stability and scalability. It uses multiple parallel agents to update shared policy and value
networks asynchronously to accelerate training. Nonetheless, A3C’s on-policy nature still struggled with generalization across levels.

3. Proximal Policy Optimization (PPO) emerged as a more stable and efficient on-policy method, introducing clipped policy updates to balance exploration and exploitation.

The above games were tested on Atari games, but Nes games are more difficult than Atari. The purpose of this project is to test the performance of PPO algorithm on Nes games.

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
$$
M = \left\langle S, A, O, T, R\right\rangle
$$
, where 
Symbol|Definition 
-|-
$S$|The set of all possible emulator states
$A=\{a_{1},…,a_{6}\}$| Discrete action set :$\{B,Left + B, Right + B, Down + B,A + B,Right + Up + B,Right + Down + B,Right + A + B\}$.
$O = [0,1]^{K\times H\times W}$ | Observtion space. Each observation $o$​ is a stack of the $K=4$ most recent grayscale frames, where $H=W=84$.
$T:S\times A \to \Delta(S)$ | Transitions $T(s' \mid s,a)$ is determined by the game emulator written determistically in ROM code. For us and the agent, it is a black box.
$R:S\times A \to \mathbb{R}$ | Reward function

Here, the reward function is composed of a series of heuristics functions:

Reward Component|heuristics function
-|-
Moving forward reward|$\Delta x_t = x_{\text{scroll},t} - x_{\text{scroll},t-1} - 0.01 $ <br> $ r_{\text{move},t} = \operatorname{clip}\!\bigl(\Delta x_t,\,-3,\,3\bigr)$
Killing reward| $\Delta s_t = s_t - s_{t-1}$ <br> $r_{\text{score},t} = \operatorname{clip}\!\bigl(\Delta s_t,\,0,\,2\bigr)$
Dead penalty|$r_{\text{life},t} =\begin{cases}  -15, & \text{if } \ell_t < \ell_{t-1} \\   \;\;0, & \text{otherwise} \end{cases}$
Win reward| $r_{\text{end},t} =\begin{cases}  +50, & \text{if } \ell_t > 0 \quad(\text{win})\\  -35, & \text{if } \ell_t = 0 \quad(\text{lose})\end{cases}$
Instant rewards | $ r^{\text{raw}}_t =   r_{\text{move},t} +   r_{\text{score},t} +   r_{\text{life},t} + r_{\text{end},t} $
scale up| $r_t = \frac{r^{\text{raw}}_t}{10}$
### Solution method
In this project, I formulate the game-playing task as a Markov Decision Process (MDP) and solve it using Proximal Policy Optimization (PPO). Here we focus on PPO and training pipeling.
#### PPO
I apply PPO due to its stability and suitability for high-dimensional action and observation spaces. PPO optimizes the policy by constraining updates within a trust region, thus ensuring incremental improvements and reducing performance instability.
In the project, two neural networks in our PPO architecture is maintained.

- __Actor Network(PPOAgent\.\_policy in /src/model.py):__ Maps observations $s_t$ to action probabilities $\pi_\theta(a_t|s_t)$. The actor network outputs a distribution over the 8 discrete actions defined previously.

- __Critic Network(PPOAgent\.\_value in /src/model.py)__

The PPO policy objective is given by:
$$
\begin{aligned}
L^{\text{CLIP}}(\theta)
&=
\mathbb{E}_t \left[
    \min \left(
        r_t(\theta) \, A_t,\;\;
        \operatorname{clip}\big(r_t(\theta),\, 1 - \epsilon,\, 1 + \epsilon \big) \, A_t
    \right)
\right]
\\[6pt]
\text{where} \quad
r_t(\theta) &= \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
\end{aligned}
$$
Here, $A_t$ denotes the advantage estimated by Generalized Advantage Estimation (GAE). 

#### Training Pipeline

In summary, the entire solution pipeline includes:

1. Preprocessing raw image frames to create a suitable, normalized observation space.
2. Discretizing the original NES action space into a simplified 8-action set.
3. Defining a shaped reward function guiding the agent toward game progression.
4. Employing PPO to train a robust policy that efficiently navigates the Contra-NES environment.

It should be noted that when training the model, I first trained the PPO model on the deterministic game environment for about 40 million frames. Finally, the final model is obtained by training another 40 million frames under uncertain environment.

The training hyperparameters and model details can be found in `/src/config.py` and `/src/model.py`.

## Result
Model for ContraForce has been trained on torch and gym-retro for about 40 million frames. The results weren't particularly good, so I decided to train the same algorithm on Contra (ContraForce's previous game) as an alternative. This is how much I have done on two games:
### ContraForce（abandoned）
1. The original image is directly used as input, the game Score is used as output, and PPO training is used. In this game, however, the agent was stuck at a "local maximum". Enemies in ContraForce respawn a few seconds after being killed, and agents use this mechanism to keep trying to score points in the same place without moving forward.
2. To overcome the above situation, I reward the forward movement (right direction action) directly, but the agent will go right regardless of death. It also leads to irrational strategies.
3. After that, I tried to add the optical flow function to detect whether the agent moved or not, but the implementation was difficult and the effect was not good.
### Contra
__Note: __ [Source of Randomness](#Source of Randomness)
1. In the determined game environment, the trained model can pass the game. <br>run  `$ python main.py evaluate --deterministic`  to check the result. (Because the game environment is deterministic, there is no need to repeat the run to verify.)
2. After adding randomness, the model's clearance rate becomes very low \(<5%\). <br> run`$ python main.py evaluate` and check the video under `/evaluation_videos`. <br>But for each game, ageng can still complete at least 50 percent of the content. And there is a greater than one in three chance of getting to the final Boss scenario.
#### Conclusion and Future work
It shows that PPO algorithm has a certain generalization for the game of Contra-Nes, but it is not enough to complete the first level in the case of disturbance. For Future work, it is worth trying to improve the Reward function and applying other deep reinforcement learning algorithms to Contra-Nes game.  
## Appendix and Reference
### Project Scope
<a href="https://pics1.obs.myhuaweicloud.com/GraduateGW/Proposal_YidingPei.pdf" target="_blank">Click to Download</a>

### How to install stable-retro?
You can use `pip install stable-retro` to install it. But after that, you need to copy folder `Contra-Nes` under` /ContraBrains/stable-retro/retro/data/stable/Contra-Nes` to `/{your conda package dir}/stable-retro/retro/data/stable/` .

### Source of Randomness
The source of randomness I used in Contra is the insertion of 2 frames of random action between every 50 frames. I did not use randomly changing character positions as a source of randomness in the game. While this was possible in ContraForce, in Contra I didn't find a good way to modify memory data about positions and often caused the game to crash and restart.

### Statement

Yiding Pei was responsible for setting up the environment, training and writing the report. Referencing some code from the [codebase](https://github.com/Hauf3n/PPO-Atari-PyTorch/blob/main/CartPole-Acrobot/PPO_Cartpole_Acrobot.ipynb).
