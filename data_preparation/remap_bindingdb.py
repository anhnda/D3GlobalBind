import glob

from commons.filebuffer import TextFileBuffer
from data_preparation.extract_nonbindingdb.stats_bindingdb import __skipUntil
from commons.func import loadMapFromFile, get_dict
PDB_LIGAND_SDF_DIR = "/home/gpux1/Data/PDB/PDBinding/LigandDataSDF"
BINDINGDB_DIR = "/home/gpux1/Data/PDB/BindingDB/BindingDB_All_2D_2022m5.sdf"
BINDINGDB_EXTRACT_FILE = "%s/BindingDB_All_2D.sdf.extract.txt" % BINDINGDB_DIR
BINDINGDB_NAME_PROTEIN_UNIPROT = "%s/../BindingDB_UniProt.txt" % BINDINGDB_DIR
LIGAND_INCHIKEY_2_LIGAND_PDB = "%s/LigandMap.txt" % BINDINGDB_DIR

def loadBindingProteinNameMap():
    d = {}
    fin = open(BINDINGDB_NAME_PROTEIN_UNIPROT)
    fin.readline()
    while True:
        line = fin.readline()
        if line == "":
            break
        parts = line.strip().split("\t")
        uniprotname = parts[1]
        bindingName= parts[2]
        d[bindingName] = uniprotname
    fin.close()
    return d

def getInchiKeyMapLigand(path):
    f = TextFileBuffer(path, bufferSize=50, stripLine=True)
    lines, _ = f.getNextLines(nLine=-1, endingMarker="$$$$")
    # print(lines)
    cLineId = 0
    pdbLigandID = lines[0]
    cLineId = __skipUntil(lines, cLineId,'> <')
    cLineId = __skipUntil(lines, cLineId, '> <OPENEYE_INCHIKEY>')
    ligandInchiKey = lines[cLineId]
    f.close()
    return pdbLigandID, ligandInchiKey


def exportLIGAND_InchikeyMap():
    paths = glob.glob("%s/*.sdf" % PDB_LIGAND_SDF_DIR)
    dLigandInchikey2PDBLigandId = {}
    for i,path in enumerate(paths):
        if i % 10 == 0:
            print("\r%s" % i, end="")
        pdbLigandID, ligandInchikey = getInchiKeyMapLigand(path)
        dLigandInchikey2PDBLigandId[ligandInchikey] = pdbLigandID
    f = open(LIGAND_INCHIKEY_2_LIGAND_PDB, "w")
    print("\n Done")
    for k,v in dLigandInchikey2PDBLigandId.items():
        f.write("%s\t%s\n" % (k,v))
    f.close()

def loadLigandMap():
    return loadMapFromFile(LIGAND_INCHIKEY_2_LIGAND_PDB)

def pad(v, ex="???"):

    if v == "":
        v = ex
    return v
def convertBindingDB():
    dLigand = loadLigandMap()
    dProtein = loadBindingProteinNameMap()
    # print(dProtein)
    # fin = TextFileBuffer(BINDINGDB_EXTRACT_FILE, bufferSize=1000, stripLine=True)
    fin = open(BINDINGDB_EXTRACT_FILE)
    fout = open("%s_convert" % BINDINGDB_EXTRACT_FILE, "w")
    iLine = 0
    while True:
        # line, exitCode = fin.getNextLines(1)
        # if exitCode == -1:
        #     break
        # line = line[0]
        line = fin.readline()
        if line == "":
            break
        line = line.strip()
        # print(line)
        iLine += 1
        if iLine % 100 == 0:
            print("\r%s"%iLine, end="")
        parts = line.split("$$$")
        ligandInchiKey = parts[1]
        pdbName = parts[2]
        pdbComplex = parts[4]
        if pdbComplex != "":
            pdbComplex = pdbComplex.split(",")[0]
        else:
            pdbComplex = "###"
        ligandPDBID = get_dict(dLigand, ligandInchiKey, "")
        proteinUniprot = get_dict(dProtein, pdbName, "")
        if ligandPDBID != "" and proteinUniprot != "":
            ki, ic, kd = pad(parts[-5]), pad(parts[-3]), pad(parts[-1])

            l = [ligandPDBID, proteinUniprot, pdbComplex, ki, ic, kd]
            fout.write("%s\n" % "\t".join(l))
    fout.close()
    fin.close()
    print("\nDone at ", iLine)

if __name__=="__main__":
    # exportLIGAND_InchikeyMap()
    convertBindingDB()


