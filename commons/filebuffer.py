class TextFileBuffer:
    def __init__(self, filePath=None, bufferSize=10000, stripLine=False):
        self.buffer = []
        self.currentLineId = -1
        self.currentBufferId = -1
        self.isEOF = False
        self.bufferSize = bufferSize
        self.filePath = filePath
        self.f = open(filePath)
        self.stripLine = stripLine
        self.currentLineIdBuffer = 0
        self.isClosed = False

    def __getNextBuffer(self):
        if len(self.buffer) != 0:
            self.buffer.clear()
        self.currentBufferId += 1
        self.currentLineIdBuffer = 0
        c = 0
        while True:
            line = self.f.readline()
            if line == "":
                self.isEOF = True
                break
            if self.stripLine:
                line = line.strip()
            self.buffer.append(line)
            c += 1
            if c == self.bufferSize:
                break
        return len(self.buffer)

    def __getNumNextLineBuffer(self, pullNext=True):
        nLineBuffer = len(self.buffer) - self.currentLineIdBuffer
        assert nLineBuffer >= 0
        if nLineBuffer == 0 and pullNext:
            nLineBuffer = self.__getNextBuffer()
        return nLineBuffer

    def getNextLines(self, nLine=1, endingMarker=None):
        assert not self.isClosed
        if nLine == -1:
            assert endingMarker is not None
            nLine = 9999999
        lines = []
        nLineBuffer = self.__getNumNextLineBuffer(pullNext=True)
        if nLineBuffer == 0:
            return lines, -1

        returnCode = 0
        hitEndingMarker = False
        while nLine > 0:
            nextNumLine = min(nLine, nLineBuffer)
            # print("Next N Line", nextNumLine)
            nextLinesBuffer = self.buffer[self.currentLineIdBuffer:self.currentLineIdBuffer + nextNumLine]
            if endingMarker is not None:
                nextLines = []
                for l in nextLinesBuffer:
                    nextLines.append(l)
                    if l.startswith(endingMarker):
                        hitEndingMarker = True
                        break
            else:
                nextLines = nextLinesBuffer
            nextNumLine = len(nextLines)
            self.currentLineIdBuffer += nextNumLine
            if len(lines) == 0:
                lines = nextLines
            else:
                lines = lines + nextLines
            nLine -= nextNumLine
            nLineBuffer = self.__getNumNextLineBuffer(pullNext=True)
            if nLineBuffer == 0:
                returnCode = -1
                break
            if hitEndingMarker:
                returnCode = 1
                break

        self.currentLineId += len(lines)
        return lines, returnCode

    def close(self):
        self.isClosed = True
        self.buffer.clear()
        self.f.close()
