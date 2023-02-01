import glob
import re

DATA_DIR = "/home/gpux1/Codes/EquiBind/data/PDBBind"


def split_pair(token):
    hyphen_pos = []
    for i, c in enumerate(token):
        if c == '-':
            hyphen_pos.append(i)
    hyphen_pos.append(len(token) - 1)
    num_list = []
    for i in range(len(hyphen_pos) - 1):
        num_str = token[hyphen_pos[i]:hyphen_pos[i + 1]]
        num_list.append(float(num_str))
    return num_list


def get_max_coord(path, skip_error=True, max_size=9):
    fin = open(path)
    mx = -1e10
    while True:
        line = fin.readline()
        if line == "":
            break
        if line.startswith("ATOM"):
            l = re.sub("\s+", " ", line)
            parts = l.split(" ")
            for i in [6, 7, 8]:
                p = parts[i]
                if p == "999.991000":
                    print("?????")
                    print(path)
                    exit(-1)
                if len(p) >= max_size and skip_error:
                    continue
                try:
                    v = float(p)
                    v = abs(v)
                    if v > mx:
                        mx = v
                except:
                    if skip_error:
                        continue
                    num_list = split_pair(p)
                    for v in num_list:
                        v = abs(v)
                        if v > mx:
                            mx = v
    fin.close()
    return mx,path


def get_max_all_files():
    paths = glob.glob("%s/*/*_processed.pdb" % DATA_DIR)
    mx = 1e-10
    i = 0
    for path in paths:
        i += 1
        if i % 10 == 0:
            print("\r%s" % i, end="")
        try:
            mx = max(mx, get_max_coord(path))
        except:
            print(path)

    fout = open("%s/../max_coord.info" % DATA_DIR, "w")
    fout.write("%f" % mx)
    fout.close()

def get_max_all_files_p():
    paths = glob.glob("%s/*/*_processed.pdb" % DATA_DIR)
    import numpy as np
    from multiprocessing import Pool
    with Pool(8) as pool:
        res = pool.map(get_max_coord, paths)

    mxs = []
    ps = []
    for re in res:
        mx, path = re
        mxs.append(mx)
        ps.append(path)
    mxs = np.asarray(mxs)
    iv = np.argmax(mxs)
    print(mxs[iv], ps[iv])

    fout = open("%s/../max_coord.info" % DATA_DIR, "w")
    fout.write("%f" % mxs[iv])
    fout.close()


def demo():
    # print(get_max_coord("%s/1a0t/1a0t_protein_processed.pdb" % DATA_DIR))
    # print(split_pair('-18.902-100.848'))
    # get_max_coord("ss.txt")
    pass


if __name__ == "__main__":
    # demo()
    # get_max_all_files()
    get_max_all_files_p()
