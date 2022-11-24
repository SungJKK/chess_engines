# Chess engines project
Repository for AMS 325 final project.  

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
- [m1 mac tensorflow install](https://stackoverflow.com/questions/72964800/what-is-the-proper-way-to-install-tensorflow-on-apple-m1-in-2022) 
