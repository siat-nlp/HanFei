"""
@Desc:
@Reference:
@Notes:
"""


def print_done(desc: str):
    def wrapper(func):
        def decorate(*args):
            print(f"{desc}...", end="")
            func(*args)
            print('- done')

        return decorate

    return wrapper
