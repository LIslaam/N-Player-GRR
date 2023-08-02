# N-Player-GRR
Extension of Jon Lorraine et al. work "Lyapunov Exponents for Diversity in Differentiable Games" https://arxiv.org/pdf/2112.14570.pdf

To run, type
```Shell
python3 src/run_exp.py --game=small-ipd --optimiser=sgd
```
options:
--game: 'small-ipd', 'mp' (matching pennies), 'mix' (interpolated small IPD and matching pennies)
--optimiser: 'sgd', 'lola'
--dimensions: '2' or '3', two and three player GRR supported
