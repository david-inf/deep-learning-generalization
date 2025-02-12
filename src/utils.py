
def N(x):
    # detach from computational graph
    # send back to cpu
    # numpy ndarray
    return x.detach().cpu().numpy()


# TODO: plotting routines
