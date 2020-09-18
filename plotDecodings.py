import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, isdir, join, splitext
from pathlib import Path

if __name__ == "__main__":
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure()
    ax = plt.axes()

    aggressiveDecodingsDirNames = [f for f in listdir('Decodings/Aggressive') if isdir(join('Decodings/Aggressive', f))]
    for _, logName in enumerate(aggressiveDecodingsDirNames):
        logName = splitext(logName)[0]
        aggressiveDecodingsFileNames = [f for f in listdir('Decodings/Aggressive/{0}'.format(logName)) if isfile(join('Decodings/Aggressive/{0}'.format(logName), f))]
        for _, fileName in enumerate(aggressiveDecodingsFileNames):
            data = pd.read_csv("Decodings/Aggressive/{0}/{1}".format(logName, fileName))
            readings = data.values[:,1]
            timestamps = data.values[:,2]
            plt.title(fileName + " (Aggressive)")
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.plot(timestamps, readings)
            Path('Plots/Aggressive/{0}'.format(logName)).mkdir(parents=True, exist_ok=True)
            plt.savefig('Plots/Aggressive/{0}/{1}.png'.format(logName,fileName), bbox_inches='tight')
            plt.clf()

    passiveDecodingsDirNames = [f for f in listdir('Decodings/Passive') if isdir(join('Decodings/Passive', f))]
    for _, logName in enumerate(passiveDecodingsDirNames):
        logName = splitext(logName)[0]
        passiveDecodingsFileNames = [f for f in listdir('Decodings/Passive/{0}'.format(logName)) if isfile(join('Decodings/Passive/{0}'.format(logName), f))]
        for _, fileName in enumerate(passiveDecodingsFileNames):
            data = pd.read_csv("Decodings/Passive/{0}/{1}".format(logName, fileName))
            readings = data.values[:,1]
            timestamps = data.values[:,2]
            plt.title(fileName + ' (Passive)')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.plot(timestamps, readings)
            Path('Plots/Passive/{0}'.format(logName)).mkdir(parents=True, exist_ok=True)
            plt.savefig('Plots/Passive/{0}/{1}.png'.format(logName,fileName), bbox_inches='tight')
            plt.clf()
