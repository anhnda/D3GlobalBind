from selenium import webdriver
import time
from selenium.webdriver.common.by import By
from commons.func import load_list_from_file, ensure_dir, load_obj, save_obj
# import requests
import urllib.request
PDBBinding_DIR = "/home/gpux1/Data/PDB/PDBinding"
Ligand_DATA_DIR = "%s/LigandData" % PDBBinding_DIR
ensure_dir(Ligand_DATA_DIR)

LigandPDBBatchMap_Path = "%s/ligandbatch.dict" % Ligand_DATA_DIR

LIGAND_LIST = "/home/gpux1/Data/PDB/PDBinding/PDBbind_v2020_plain_text_index/index/ligand_ids.txt"


BATCH_SIZE = 50


def getLigandList():
    ls = load_list_from_file(LIGAND_LIST)
    return ls

# def downloadFromURL(url,targetPath):
#     response = requests.get(url)
#     fout = open(targetPath, "w")
#     fout.write(response.text)
#     fout.close()


def download_url(url, save_path):
    with urllib.request.urlopen(url) as dl_file:
        with open(save_path, 'wb') as out_file:
            out_file.write(dl_file.read())

def downloadBatch():

    browser = webdriver.Chrome()

    dLigandBatchRe = dict()
    try:
        dLigandBatchRe = load_obj(LigandPDBBatchMap_Path)
    except:
        dLigandBatchRe = dict()


    ligandList = getLigandList()
    nBatch = len(ligandList) // BATCH_SIZE + 1
    browser.get("https://www.rcsb.org/downloads/ligands")
    time.sleep(2)
    inputLigand = browser.find_element(By.ID, "ligandIdList")
    submitButtion = browser.find_element(By.ID, "submitBtn")

    for i in range(nBatch):
        startId = i * BATCH_SIZE
        endId = min((i+1) * BATCH_SIZE, len(ligandList))
        ligandBatchIds = ligandList[startId:endId]
        ligandBatchIdsString = ",".join(ligandBatchIds)
        batchName = "%s-%s" % (ligandBatchIds[0], ligandBatchIds[-1])
        batchName = batchName.replace("/", "__")
        if batchName in dLigandBatchRe:
            continue
        inputLigand.clear()
        inputLigand.send_keys(ligandBatchIdsString)
        time.sleep(2)
        submitButtion.click()
        time.sleep(2)
        url = browser.find_element(By.XPATH, "//a[starts-with(@href,'https://download.rcsb.org/batch/ccd/')]")
        url = url.get_attribute('href')
        targetPath = "%s/%s.zip" % (Ligand_DATA_DIR, batchName)
        # print(url)
        download_url(url, targetPath)
        dLigandBatchRe[batchName] = targetPath
        print(i, targetPath)
        save_obj(dLigandBatchRe, LigandPDBBatchMap_Path)

    save_obj(dLigandBatchRe, LigandPDBBatchMap_Path)


if __name__ == "__main__":
    downloadBatch()