from pytorch_lightning.utilities import rank_zero_only


@rank_zero_only
def print_rank_0(*args):
    print(*args)
