import time
from io import open

import numpy as np
import joblib
import os
import time

from rdkit import Chem

def read_mol(molecule_file):
    mol = None
    if molecule_file.endswith('.sdf'):

        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    else:
        print("File error")
    return mol



def trans_mol(mol, r, t, m = 0):
    # return mol
    # print("TPM" , type(mol), type(r), type(t), type(m))
    conf = mol.GetConformer()
    mol_coords = conf.GetPositions()
    # print(mol_coords)
    mol_coords = mol_coords - m
    new_coords = (r @ mol_coords.transpose()).transpose() + t
    # print(new_coords)
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (new_coords[i][0],new_coords[i][1],new_coords[i][2]))

    # writer = Chem.SDWriter('x.sdf')
    # for cid in range(mol.GetNumConformers()):
    #     mol.SetProp('ID', f'x_{cid}')
    #     writer.write(mol, confId=cid)
    # writer.close()

    return mol

def save_mol_sdf(mol, path):
    writer = Chem.SDWriter(path)
    writer.SetKekulize(False)
    # print(path)
    for cid in range(mol.GetNumConformers()):

        mol.SetProp('ID', f'x_{cid}')
        # if mol is not None: Chem.Kekulize(mol, clearAromaticFlags=True)
        writer.write(mol, confId=cid)


    writer.close()

def getCurrentTimeString(no_space=True):
    t = time.localtime()
    currentTime = time.strftime("%Y-%m-%d %H:%M:%S", t)
    if no_space:
        currentTime = currentTime.replace(" ", "_")
    return currentTime





def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
def copy_dir(c_dir, target_dir, exclusives=[".git", "tmp", ".idea", "__pycache__"], exclusive2=[]):

    if len(exclusive2) > 0:
        for e in exclusive2:
            exclusives.append(e)
    # command = "cp -R \"%s\"  \"%s\"" % (c_dir, target_dir)
    ex = " ".join(["--exclude \"%s\"" % v for v in exclusives])
    command = "rsync -a  \"%s\"  \"%s\" %s" % (c_dir, target_dir, ex)
    ensure_dir(target_dir)
    print("Cmd: ", command)
    os.system(command)

def convertHexToBinString888(hexString):
    # scale = 16  ## equals to hexadecimal
    # num_of_bits = 888
    return bin(int(hexString, 16))[2:].zfill(888)


def convertBinString888ToArray(binString888):
    ar = np.ndarray(888, dtype=float)
    ar.fill(0)
    for i in range(887, -1, -1):
        if binString888[i] == "1":
            ar[i] = 1
    return ar


def convertHex888ToArray(hex888):
    return convertBinString888ToArray(convertHexToBinString888(hex888))


def get_dict(d, k, v=-1):
    try:
        v = d[k]
    except:
        pass
    return v


def get_insert_key_dict(d, k, v=0):
    try:
        v = d[k]
    except:
        d[k] = v
    return v


def add_dict_counter(d, k, v=1):
    try:
        v0 = d[k]
    except:
        v0 = 0
    d[k] = v0 + v


def sort_dict(dd):
    kvs = []
    for key, value in sorted(dd.items(), key=lambda p: (p[1], p[0])):
        kvs.append([key, value])
    return kvs[::-1]


def sum_sort_dict_counter(dd):
    cc = 0
    for p in dd:
        cc += p[1]
    return cc


def get_update_dict_index(d, k):
    try:
        current_index = d[k]
    except:
        current_index = len(d)
        d[k] = current_index
    return current_index


def get_dict_index_only(d, k):
    try:
        current_index = d[k]
    except:
        current_index = -1

    return current_index


def load_list_from_file(path):
    list = []
    fin = open(path)
    while True:
        line = fin.readline()
        if line == "":
            break
        list.append(line.strip())
    fin.close()
    return list


def reverse_dict(d):
    d2 = dict()
    for k, v in d.items():
        d2[v] = k
    return d2


def save_obj(obj, path):
    joblib.dump(obj, path)


def load_obj(path):
    return joblib.load(path)


def loadMapFromFile(path, sep="\t", keyPos=0, valuePos=1):
    fin = open(path)
    d = dict()
    while True:
        line = fin.readline()
        if line == "":
            break
        parts = line.strip().split(sep)
        d[parts[keyPos]] = parts[valuePos]
    fin.close()
    return d


def loadMapSetFromFile(path, sep="\t", keyPos=0, valuePos=1, sepValue="", isStop=""):
    fin = open(path)
    dTrain = dict()

    if isStop != "":
        dTest = dict()

    d = dTrain

    while True:
        line = fin.readline()
        if line == "":
            break
        if isStop != "":
            if line.startswith(isStop):
                d = dTest
                continue
        parts = line.strip().split(sep)
        v = get_insert_key_dict(d, parts[keyPos], set())
        if sepValue == "":
            v.add(parts[valuePos])
        else:
            values = parts[valuePos]
            values = values.split(sepValue)
            for value in values:
                v.add(value)
    fin.close()
    if isStop != "":
        return dTrain, dTest
    return dTrain


def getTanimotoScore(ar1, ar2):
    c1 = np.sum(ar1)
    c2 = np.sum(ar2)
    bm = np.dot(ar1, ar2)
    return bm * 1.0 / (c1 + c2 - bm + 1e-10)

def getCosin(ar1, ar2):

    return np.dot(ar1, ar2)

def get3WJaccardOnSets(set1, set2):
    len1 = len(set1)
    len2 = len(set2)
    nMatch = 0
    for s in set1:
        if s in set2:
            nMatch += 1
    return 3.0 * nMatch / (len1 + len2 + nMatch + 0.01)