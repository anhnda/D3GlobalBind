import time

from commons.filebuffer import TextFileBuffer
from data_preparation.extract_nonbindingdb import sconf as conf


class BindingDBTextFileBuffer:
    def __init__(self, bufferSize=100000, endingMarker="$$$$"):
        self.fileBuffer = TextFileBuffer(conf.BINDING_DB_PATH_IN, bufferSize=bufferSize, stripLine=True)
        self.endingMarker = endingMarker

    def getNextData(self):
        lines, returnCode = self.fileBuffer.getNextLines(nLine=-1, endingMarker=self.endingMarker)
        return lines, returnCode


def __skipUntil(lines, currentLineId, marker, nextId=True):
    maxLineId = len(lines)
    for i in range(currentLineId, maxLineId):
        l = lines[i]
        if l.startswith(marker):
            currentLineId = i
            break
    if nextId:
        currentLineId += 1
    return currentLineId


def extractInfo(lines):
    # print("In extracting...")
    currentLineId = 0
    currentLineId = __skipUntil(lines, currentLineId, "> ")

    currentLineId = __skipUntil(lines, currentLineId, "> <BindingDB Reactant_set_id>")

    bindingDBId = lines[currentLineId]

    currentLineId = __skipUntil(lines, currentLineId, "> <Ligand InChI Key>")
    ligandInchi = lines[currentLineId]
    currentLineId = __skipUntil(lines, currentLineId, "> <Target Name ")
    targetName = lines[currentLineId]

    currentLineId = __skipUntil(lines, currentLineId, "> <Ki", nextId=False)

    kiUnit = lines[currentLineId][3:-1]
    kiValue = lines[currentLineId + 1]

    currentLineId = __skipUntil(lines, currentLineId + 1, "> <IC50", nextId=False)
    ic50Unit = lines[currentLineId][3:-1]
    ic50Value = lines[currentLineId + 1]

    currentLineId = __skipUntil(lines, currentLineId + 1, "> <Kd", nextId=False)
    kdUnit = lines[currentLineId][3:-1]
    kdValue = lines[currentLineId + 1]

    currentLineId = __skipUntil(lines, currentLineId + 1, "> <Ligand HET ID in PDB>", nextId=True)
    ligandHETID = lines[currentLineId]

    currentLineId = __skipUntil(lines, currentLineId + 1, "> <PDB ID(s) for Ligand-Target Complex>", nextId=True)
    pdbComplexID = lines[currentLineId]

    return bindingDBId, ligandInchi, targetName, ligandHETID, pdbComplexID, kiUnit, kiValue, ic50Unit, ic50Value, kdUnit, kdValue


# def t():
#     textFileBuffer =
#     for i in range(100):
#         lines, _ = textFileBuffer.getNextLines()
#         for l in lines:
#             print(l)

def produceData(fileBufferStream, queue, maxSize):
    while True:
        while queue.qsize() >= maxSize / 2:
            time.sleep(0.001)
            continue

        data, returnCode = fileBufferStream.getNextData()
        queue.put(data)
        if returnCode == -1:
            queue.put(None)
            print("Put None")
            break


def processData(queue, queuout):
    isEnd = False
    while True:
        dat = queue.get()
        if dat is None:
            isEnd = True
            # print("Reput ", None)
            queue.put(None)
        else:
            if len(dat) < 10:
                continue
            try:
                re = extractInfo(dat)
                re = "$$$".join(re) + "\n"
                queuout.put(re)
            except:
                continue

        if isEnd:
            # print("Out None Q")
            queuout.put(None)
            break


def consume(queueOut, nC, nMax, fout):
    cc = 0
    while True:
        d = queueOut.get()
        cc += 1
        if cc % 100 == 0:
            print("\r%s" % cc, end="")
        if d is None:
            nC.value += 1
            if nC.value == nMax:
                print("\nDone at: ", cc)

                break
        fout.write("%s" %d)






def t():
    from multiprocessing import Pool, Queue, Process, Value
    queue = Queue()
    queueOut = Queue()
    filetx = BindingDBTextFileBuffer(bufferSize=100000)
    fout = open(conf.BINDING_DB_PATH_OUT, "w")
    maxQSize = 1000
    nC = Value('i', 0)
    dataProducer = Process(target=produceData, args=(filetx, queue, maxQSize))
    dataProcessors = []
    nProcess = 20
    for i in range(nProcess):
        dataProcess = Process(target=processData, args=(queue, queueOut))
        dataProcessors.append(dataProcess)
    consumer = Process(target=consume, args=(queueOut, nC, nProcess, fout))

    dataProducer.start()
    for dataProcess in dataProcessors:
        dataProcess.start()
    consumer.start()

    dataProducer.join()
    for dataProcess in dataProcessors:
        dataProcess.join()
    consumer.join()

    fout.close()


if __name__ == "__main__":
    t()
