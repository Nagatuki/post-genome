import os


def mkdir(path):
    try:
        os.makedirs(path)

    except FileExistsError as e:
        # ignore
        pass

    except Exception as e:
        raise e
