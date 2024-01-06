import itertools
from typing import Callable, Dict, Set, Tuple


def factorial(val: int) -> int:
    """Calculates factorial of val."""
    res = 1
    for i in range(1, val + 1):
        res *= i
    return res


def binom(n: int, k: int) -> int:
    """Calculates the binomial coefficient, n choose k."""
    return factorial(n) / (factorial(k) * factorial(n - k))


Player = str


def shapley(
    coalition: Set[Player], value_func: Callable[[Set[Player]], float], player: Player
) -> float:
    """Implements the Calculation of Shapley value.

    Args:
        Coalition: the set of all players.
        value_func: the value function that maps a set of players to a value.
        player: for which player is the Shapley value calculated for.
    """
    n = len(coalition)
    val = 0
    for size in range(n):
        total = 0  # sum of margins of a particular subset size.
        weight = 1 / binom(n - 1, size)
        for subset in itertools.combinations(coalition - {player}, size):
            subset = set(subset)
            margin = value_func(subset | {player}) - value_func(subset)
            total += margin
        val += total * weight
    return val / n
