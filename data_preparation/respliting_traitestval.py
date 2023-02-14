DIR = "/home/gpux1/Codes/EquiBind/data"
from commons.func import load_list_from_file
import random


def load_ids():
    inds = set()
    for p in ["timesplit_no_lig_overlap_train", "timesplit_no_lig_overlap_val", "timesplit_test"]:
        path = "%s/%s" % (DIR, p)
        ls = load_list_from_file(path)
        for ix in ls:
            inds.add(ix)
    inds = sorted(list(inds))
    return inds


def splitx(inds, rd_seed=1, nFold=20, iStart=0):
    random.seed(rd_seed)
    random.shuffle(inds)
    sz = len(inds)
    batch_size = sz // nFold
    train_ids = []
    val_ids = []
    test_ids = []

    startVal = iStart * batch_size
    endVal = min((iStart + 1) * batch_size, sz)

    if iStart < nFold - 1:
        startTest = (iStart + 1) * batch_size
        endTest = min((iStart + 2) * batch_size, sz)
    else:
        startTest = 0
        endTest = batch_size

    for i, v in enumerate(inds):
        if startVal <= i < endVal:
            l = val_ids
        elif startTest <= i < endTest:
            l = test_ids
        else:
            l = train_ids
        l.append(v)

    ls = [train_ids, val_ids, test_ids]
    paths = ["%s/%s_%s_%s" % (DIR, c, rd_seed, iStart) for c in ["train", "val", "test"]]
    for i in range(3):
        l = ls[i]
        path = paths[i]
        f = open(path, "w")
        l = "\n".join(l)
        f.write("%s"%l)
        f.close()

if __name__ == "__main__":
    inds = load_ids()
    splitx(inds)