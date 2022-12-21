from __future__ import print_function

import os.path
import time
import psutil
from datetime import timedelta

import multiprocessing

from commons.xlogger.logger2 import MyLogger

def get_time_elapse(start_time, ):
    return str(timedelta(seconds=time.time() - start_time))


def __get_memory(process, subtract_share=True, dv=1e9):
    memory_info = process.memory_info()
    memory = round(memory_info.rss / dv, 2)
    if subtract_share:
        memory -= round(memory_info.shared / dv, 2)
    return memory


def get_process_memory(pid=None, subtract_share=True, with_children=False):
    if pid is None:
        pid = os.getpid()
    process = psutil.Process(pid)
    memory = __get_memory(process, subtract_share=subtract_share)
    if with_children:
        for child in process.children(recursive=True):
            memory += __get_memory(child, subtract_share=subtract_share)

    return memory



class MemWatchDog:
    def __init__(self, logPath=None, starttime=None, interval=1):

        self._stop = multiprocessing.Value('i', 0)
        # self.logPath = logPath
        self.logger = MyLogger(logPath)
        self.pid = -1
        self.start_time = starttime
        self.interval = interval

    def start_monitor(self, pid=None, start_time=None):
        if pid is not None:
            self.pid = pid
        if start_time is not None:
            self.start_time = start_time
        process = multiprocessing.Process(target=self.__loop)
        process.daemon = True
        process.start()

    def stop_monitor(self):
        self._stop.value = 1

    def __loop(self):
        while self._stop.value == 0:
            try:
                mem1 = get_process_memory(self.pid, with_children=True, subtract_share=False)
                mem2 = get_process_memory(self.pid, with_children=True, subtract_share=True)
                elapsed_time = get_time_elapse(self.start_time)
                self.logger.infoFile((elapsed_time, "RSS: %.2f" % mem1, "NoShared: %.2f" % mem2))
                time.sleep(self.interval)
            except:
                self.logger.infoFile("Exception. Stop. ")
                break


def memPlot(path, outpath=""):
    print("Mem plot")
    print(path)
    print(outpath)
    fin = open(path)
    from matplotlib import pyplot as plt
    mems = []
    while True:
        line = fin.readline()
        if line == "":
            break
        line = line.strip()[1:-1]
        parts = line.split(",")
        if len(parts) != 3:
            print("Unknown line", line)
            break
        v = parts[2][1:-1].split(":")[1].strip()
        mems.append(float(v))
    x = [i for i in range(len(mems))]
    plt.plot(x, mems)
    plt.title("Memory usage")
    plt.ylabel("GB")
    plt.xlabel("Time x5s")
    if outpath == "":
        outpath = path[:-4]
    plt.savefig("%s.png" % outpath)


if __name__ == "__main__":
    # print("Start")
    # mem_watch_dog = MemWatchDog(logPath="../test_wd_log.txt")
    # mem_watch_dog.start_monitor(pid=23257)
    # time.sleep(10000)
    # mem_watch_dog.stop_monitor()
    # print("Done")
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-l", "--log", dest="log", type='str', help="Memory usage log path")
    parser.add_option("-o", "--out", dest="out", type='str', default='', help="Output image path")

    (options, args) = parser.parse_args()
    memPlot(options.log, options.out)
    """
        python -m utility.mem_watch_dog -l "LOG_DIR/mem_watch_dog_log.txt"
    """
    pass