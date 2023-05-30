"""
@Desc:
@Reference:
@Notes:

"""


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        import argparse
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def are_same_strings(string1: str, string2: str):
    if not isinstance(string1, str) or not isinstance(string2, str):
        raise ValueError("input should be strings")
    return string1.strip().lower() == string2.strip().lower()


def rm_extra_spaces(string: str):
    return " ".join(string.strip().split())
