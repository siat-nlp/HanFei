import numpy as np
import pandas as pd


def check_ratio_valid(val_ratio: float, test_radio: float) -> bool:
    return (val_ratio >= 0.0) & (test_radio >= 0.0) & (val_ratio + test_radio < 1.0)


def split_data(data: list, val_ratio: float = 0.05, test_ratio: float = 0.05):
    if not check_ratio_valid(val_ratio, test_ratio):
        print(f'Error: radios: {val_ratio}, {test_ratio} not valid')
    # set random seed
    np.random.seed = 42
    data_size = len(data)
    shuffiled_dices = np.random.permutation(data_size)
    val_split_pos = int(data_size * val_ratio)
    test_split_pos = val_split_pos + int(data_size * test_ratio)
    data = pd.Series(data)
    _val = data.iloc[shuffiled_dices[:val_split_pos]].values.tolist()
    _test = data.iloc[shuffiled_dices[val_split_pos:test_split_pos]].values.tolist()
    _train = data.iloc[shuffiled_dices[test_split_pos:]].values.tolist()
    return _train, _val, _test
