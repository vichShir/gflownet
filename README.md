# gflownet

gflownet is a library built upon [PyTorch](https://pytorch.org/) to easily train and extend [GFlowNets](https://arxiv.org/abs/2106.04399), also known as GFN or generative flow networks. GFlowNets are a machine learning framework for probabilistic and generative modelling, with a wide range of applications, especially in [scientific discovery](https://pubs.rsc.org/en/content/articlelanding/2023/dd/d3dd00002h) problems.

In a nutshell, GFlowNets can be regarded as a generative model designed to sample objects $x \in \mathcal{X}$ proportionally to a reward function $R(x)$. This results in the potential of sampling diverse objects with high rewards. For example, given the reward landscape depicted below, defined over a two-dimensional space, a well-trained GFlowNet will be able to sample from the four high-reward corners with high probability.

<p align="center">
  <img width="400" src="docs/source/figures/reward_landscape.png" />
</p>

GFlowNets rely on the principle of **compositionality** to generate samples. A meaningful decomposition of samples $x$ into multiple intermediate states $s_0\rightarrow s_1 \rightarrow \dots \rightarrow x$ can yield generalisable patterns. These patterns can then be learned by neural networks trained to model the value of transitions $F_{\theta}(s_t \rightarrow s_{t+1})$.

Consider the problem of generating [Tetris](https://en.wikipedia.org/wiki/Tetris)-like boards. A natural decomposition of the sample generation process would be to add one piece at a time, starting from an empty board. For any state representing a board with pieces, we could identify its valid parents and children, as illustrated in the figure below.

<p align="center">
  <img width="400" src="docs/source/figures/tetris_flows.png" />
</p>

We could define a reward function $R(x)$ as the number of cells occupied by pieces, for instance. The goal of training a GFlowNet on this task would be to discover (sample) diverse solutions (boards with pieces) with high rewards. This represents an intuitive yet complex problem where GFlowNets can be used, which is [implemented in this library](https://github.com/alexhernandezgarcia/gflownet/blob/ahg/313-update-readme-components/gflownet/envs/tetris.py). Many problems in scientific discoveries, such as the inverse design of proteins, molecules, or crystals share similarties with this intuitive task.

## Main Components of the GFlowNet Library

The GFlowNet library comprises four core components: environment, proxy, policy models (forward and backward), and GFlowNet agent.

### Environment

The environment defines the state space $\mathcal{S}$ and action space $\mathbb{A}$ of a particular problem, for example the Tetris task. To illustrate the environment, let's consider an even simpler environment currently implemented in the library: the [Scrabble](https://en.wikipedia.org/wiki/Scrabble) environment, inspired by the popular board game.

The Scrabble environment simulates a simple letter arrangement game where words are constructed by adding one letter at a time, up to a maximum sequence length (typically 7). Therefore, the action space is the set of all English letters plus a special end-of-sequence (EOS) action; and the state space is the set of all possible words with up to 7 letters. We can represent each `state` as a list of indices corresponding to the letters, padded with zeroes to the maximum length. For example, the state for the word "CAT" would be represented as `[3, 1, 20, 0, 0, 0, 0]`. Actions in the Scrabble environment are single-element tuples containing the index of the letter, plus the end-of-sequence (EOS) action `(-1,)`.

Using the gflownet library for a new task will typically require implementing your own environment. The library is particularly designed to make such extensions as easy as possible. In the documentation, we show how to do it step by step. You can also watch [this live-coding tutorial](https://www.youtube.com/watch?v=tMVJnzFqa6w&t=5h22m35s) on how to code the Scrabble environment.

### Proxy

We use the term "[proxy](gflownet/proxy/base.py)" to refer to the function or model that provides the rewards for the states of an environment. In other words, In the context of GFlowNets, the proxy can be thought of as a function $E(x)$ from which the reward is derived: $R(x) = g(E(x))$, where $g$ is a function that transforms the proxy values into non-zero rewards, that is "the higher the reward the better". For example, we can implement a proxy that simulates the scores of a word in the Scrabble game. That is, the [`ScrabbleScorer`](gflownet/proxy/scrabble.py) proxy computes the sum of the score of each letter of a word. For the word "CAT" that is $E(x) = 3 + 1 + 1 = 5$. While in many environments the proxy functions is a simple scorer, more complex settings like molecule or [crystal generation](gflownet/proxy/crystals/dave.py) may be use proxies that represent the energy or a property predicted by a pre-trained machine learning model.

Adapting the gflownet library for a new task will also likely require implementing your own proxy, which is usually fairly simple, as illustrated in the documentation.

### Policy models

The policy models are neural networks that model the forward and backward transitions between states, $F_{F_{\theta}}(s_t \rightarrow s_{t+1})$ (forward) and $F_{B_{\theta}}(s_{t+1} \rightarrow s_t)$ (backward). These models take a state as input and output a distribution over the actions in the action space. For continuous environments, the outputs are the parameters of a probability distribution to sample continuous-valued actions. For many tasks, simple multi-layer perceptrons with a few layers do the job, but technically any architecture could be used as policy model. 

### GFlowNet Agent

The GFlowNet Agent is the central component that ties all others together. It orchestrates the interaction between the environment, policies, and proxy, as well as other auxiliary components such as the Evaluator and the Logger. The GFlowNet can construct training batches by sampling trajectories, optimise the policy models via gradient descent, compute evaluation metrics, log data to [Weights & Biases](https://wandb.ai/), etc. The agent can be configured to optimise any of the following loss functions implemented in the library: [flow matching (FM)](https://arxiv.org/abs/2106.04399), [trajectory balance (TB)](https://arxiv.org/abs/2201.13259), [detailed balance (TB)](https://arxiv.org/abs/2201.13259) and [forward-looking (FL)](https://arxiv.org/abs/2302.01687).

## Installation

First, install uv:

```bash
pip -q install uv
```

Clone the repository:

```bash
git clone https://github.com/vichShir/gflownet.git
cd gflownet
```

Create the environment and install the dependencies:

```bash
uv venv ./gfn-env --python 3.10
uv pip install --python ./gfn-env/bin/python .
uv pip install --python ./gfn-env/bin/python seaborn
```

> To install any other packages: `uv pip install --python ./gfn-env/bin/python ipdb`

- This project **requires** Python 3.10 and CUDA 11.8.
- It is also **possible to install a CPU-only environment** that supports most features (see below).
- Setup is currently only supported on Ubuntu. It should also work on OSX, but you will need to handle the package dependencies.

## Quickstart: How to train a GFlowNet model for Tetris

Execute the following script to train a GFlowNet model for the Tetris environment:

```bash
bash run_train.sh
```

> It may take around 1 hour to process 5000 timesteps using CPU.

## Evaluation of a trained GFlowNet

To evaluate a trained GFlowNet and sample from it, we can use the script `run_eval.sh`. Check the `rundir` argument to the desired model checkpoint (`./logs` directory). For example:

```
bash run_eval.sh
```

> The script consider you trained the GFlowNet model with 5000 timesteps.

## Visualizing Tetris boards

To visualize the best Tetris board from a random or trained model, use the following scripts, respectively:

```
bash run_viz_random.sh
bash run_viz_trained.sh
```

> It will run 2000 plays and select the best play. Also, do not forget to specify the model checkpoint path (`./logs` directory).

## Contributors

Many wonderful scientists and developers have contributed to this repository: [Alex Hernandez-Garcia](https://github.com/alexhernandezgarcia), [Nikita Saxena](https://github.com/nikita-0209), [Alexandra Volokhova](https://github.com/AlexandraVolokhova), [Michał Koziarski](https://github.com/michalkoziarski), [Divya Sharma](https://github.com/sh-divya), [Pierre Luc Carrier](https://github.com/carriepl) and [Victor Schmidt](https://github.com/vict0rsch).

## Research

This repository has been used in at least the following research articles:

- Lahlou et al. [A theory of continuous generative flow networks](https://proceedings.mlr.press/v202/lahlou23a/lahlou23a.pdf). ICML, 2023.
- Hernandez-Garcia, Saxena et al. [Multi-fidelity active learning with GFlowNets](https://arxiv.org/abs/2306.11715). RealML at NeurIPS 2023.
- Mila AI4Science et al. [Crystal-GFN: sampling crystals with desirable properties and constraints](https://arxiv.org/abs/2310.04925). AI4Mat at NeurIPS 2023 (spotlight).
- Volokhova, Koziarski et al. [Towards equilibrium molecular conformation generation with GFlowNets](https://arxiv.org/abs/2310.14782). AI4Mat at NeurIPS 2023.

## Cite

Bibtex Format

```text
@misc{hernandez-garcia2024,
  author = {Hernandez-Garcia, Alex and Saxena, Nikita and Volokhova, Alexandra and Koziarski, Michał and Sharma, Divya and Viviano, Joseph D and Carrier, Pierre Luc and Schmidt, Victor},
  title  = {gflownet},
  url    = {https://github.com/alexhernandezgarcia/gflownet},
  year   = {2024},
}
```

Or [CFF file](./CITATION.cff)
