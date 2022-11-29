# Chess engines project
Repository for AMS 325 final project.  

# Todo
- [ ] create custom environment to interact w agent & play chess game
- [ ] split DQN agent class into another Neural Network class
- [ ] move training step into DQN agent class method
- [ ] check huber loss & improve neural network architecture


# Table of Contents
- [Description](#description)
- [Setup](#setup)
- [Authors](#authors)
- [References](#references)


# Description
Implementing 2 chess engines:
1. Naive algorithm
    - datasets acquired from [lichess](https://database.lichess.org/)
    - optional dataset: [chess.com](https://www.chess.com/news/view/published-data-api#pubapi-endpoint-games-archive-list)
2. Deep learning neural network algorithm


# Setup 
```sh
$ git clone https://github.com/SungJKK/chess_engines && cd chess_engines
$ conda env create -n chess_engines -f environment.yml
$ conda activate chess_engines
```

- Note: if you are using mac arm-processors, follow the steps below
```sh
$ git clone https://github.com/SungJKK/chess_engines && cd chess_engines
$ CONDA_SUBDIR=osx-arm64 conda env create -n chess_engines -f environment.yml
$ conda activate chess_engines
$ conda config --env --set subdir osx-arm64
```


# Authors
[Sung Joong Kim](https://github.com/SungJKK) and [Bernard Tenreiro](https://github.com/BernardTenreiro)


# References
- [Guide to Reinforcement Learning with Python and Tensorflow](https://rubikscode.net/2021/07/13/deep-q-learning-with-python-and-tensorflow-2-0/)
- [Can Deep Reinforcement Learning Solve Chess?](https://towardsdatascience.com/can-deep-reinforcement-learning-solve-chess-b9f52855cd1e)
- [Hacking Chess with Decision Making Deep Reinforcement Learning](https://towardsdatascience.com/hacking-chess-with-decision-making-deep-reinforcement-learning-173ed32cf503)
- [AlphaZero paper](https://arxiv.org/abs/1712.01815)
- [AlphaZero chess representation](https://www.chessprogramming.org/AlphaZero)

- [python-chess python package](https://python-chess.readthedocs.io/en/latest/)
- [gym-chess python package](https://github.com/iamlucaswolf/gym-chess)
- [m1 mac tensorflow install](https://stackoverflow.com/questions/72964800/what-is-the-proper-way-to-install-tensorflow-on-apple-m1-in-2022) 
- [tensorflow documentation](https://www.tensorflow.org/api_docs)



