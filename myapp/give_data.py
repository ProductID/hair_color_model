def giveData(aa):
    import argparse
    # parser = argparse.ArgumentParser(description='Barbershop')
    parser = argparse.ArgumentParser()
    for k, v in aa.items():
        parser.add_argument('--' + k, default=v)
    print(parser)
    ags = parser.parse_args(args=[])
    return ags
