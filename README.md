# Sequential-Social-Dilemma

This repository is an open-source implementation of DeepMind's Sequential Social Dilemma (SSD) multi-agent game-theoretic environments [[1]](https://arxiv.org/abs/1702.03037). SSDs can be thought of as analogous to spatially and temporally extended Prisoner's Dilemma-like games. The reward structure poses a dilemma because individual short-term optimal strategies lead to poor long-term outcomes for the group.

This implementation seeks to extend the social dilemma scenario to incorporate new dynamics aimed at making the strategic scenario more realistic and to implement mechanisms to incentivise the emergence of cooperative behaviour in agents.

The implemented environments are structured to be compatible with [OpenAIs gym environments](https://github.com/openai/gym).

# Installation 

To install `cd` to the directory of the repository and run `pip install -e`.

# Usage 

The file `example.py` contains a simple usage example where you can modify the number of agents and the size of its field of vision. To run the example `cd` to the directory of the repository and run `python example.py`. You should see something like this:

![3_1-ezgif com-crop](https://github.com/user-attachments/assets/7fc1341c-b7d0-498c-b1c1-a0c393f794c3)

# Relevant papers 

1. Leibo, J. Z., Zambaldi, V., Lanctot, M., Marecki, J., & Graepel, T. (2017). [Multi-agent reinforcement learning in sequential social dilemmas](https://arxiv.org/abs/1702.03037). In Proceedings of the 16th Conference on Autonomous Agents and MultiAgent Systems (pp. 464-473).
2. Perolat, J., Leibo, J. Z., Zambaldi, V., Beattie, C., Tuyls, K., & Graepel, T. (2017). [A multi-agent reinforcement learning model of common-pool resource appropriation](https://proceedings.neurips.cc/paper_files/paper/2017/hash/2b0f658cbffd284984fb11d90254081f-Abstract.html). Advances in neural information processing systems, 30.
3. Macy, M. W., & Flache, A. (2002). [Learning dynamics in social dilemmas](https://www.pnas.org/doi/abs/10.1073/pnas.092080099). Proceedings of the National Academy of Sciences, 99(suppl_3), 7229-7236.


# Contributors

This project is based on the PyColab environment, with the base code for the environment architecture originally developed by [tiagoCuervo](https://github.com/tiagoCuervo/CommonsGame), which has provided a solid foundation for this work. Any modifications or extensions beyond the original code are on my own.
