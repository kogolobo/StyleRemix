import numpy as np
import pandas as pd

from collections import defaultdict
from typing import Dict, List

def std_to_weight(n_std: float) -> float:
    if n_std <= 1:
        return .7
    elif n_std > 1 and n_std <= 2:
        return .9
    elif n_std > 2 and n_std <= 3:
        return 1.2
    else:
        return 1.5
    
def choose_directions_genre_mean(row: pd.Series, top_n: int, available_types: List[str], all_types: List[str]) -> Dict[str, float]:
    indices = np.abs(row.values).argsort()[::-1][:(top_n + 4)]
    num_chosen = 0
    axes = row.index[indices]
    directions = {}
    type_chosen = False
    for axis in axes:
        effective_axis = axis
        n_std = np.abs(row[axis])
        weight = (-1.0) * np.sign(row[axis])

        if axis in all_types:
            types_to_choose = [type_name for type_name in available_types if type_name != axis]
            if len(types_to_choose) == 0 or type_chosen:
                continue
            effective_axis = np.random.choice(types_to_choose)
            type_chosen = True
            weight = 1.0


        weight *= std_to_weight(n_std)
        directions[effective_axis] = weight
        num_chosen += 1
        if num_chosen >= top_n:
            break

    return directions

def combine_directions(directions: List[str], all_types: List[str]) -> Dict[str, float]:
    direction_sums = defaultdict(float)
    direction_counts = defaultdict(int)
    type_chosen = None
    for slider in directions:
        for axis, weight in slider.items():
            if axis in all_types:
                if type_chosen is not None and type_chosen != axis:
                    continue
                type_chosen = axis
            direction_sums[axis] += weight
            direction_counts[axis] += 1
            
    combined_directions = {axis: weight / direction_counts[axis] for axis, weight in direction_sums.items()}
    return combined_directions