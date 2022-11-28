def giveData(aa):
    import argparse
    parser = argparse.ArgumentParser()
    for k, v in aa.items():
        n_k=f"--{k}"
        parser.add_argument(n_k, default=v)
    print(parser)
    ags = parser.parse_args(args=[])
    return ags
