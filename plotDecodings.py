import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cantools
from os import listdir
from os.path import isfile, isdir, join, splitext
from pathlib import Path

class Frame:
    def __init__(self, data):
        self.frame = data
        self.id_dec = int(self.frame[0])
        self.len = int(self.frame[1])
        self.hex_data = self.frame[2:(self.len+2)]
        self.time_stamp = self.frame[-1]

def write_decoded_mesage_to_file(db, frame, drivingStyle, logName):
    hexPayload = ''
    for byte in frame.hex_data:
        hexPayload += byte
    hexPayloadBytes = bytearray.fromhex(hexPayload)
    try:
        decodedFrame = db.decode_message(frame.id_dec, hexPayloadBytes)
        for key, value in decodedFrame.items():
            stringToWrite = key + "," + str(value) + "," + str(frame.time_stamp) + "\n"
            m = open("Decodings/{0}/{1}/{2}.csv".format(drivingStyle, logName, key), "a")
            m.write(stringToWrite)
            m.close
    except:
        pass

if __name__ == "__main__":
    # Load DBC database file
    db = cantools.database.load_file('Dbc/hyundai_i30_2014.dbc')
    plt.style.use('seaborn-whitegrid')

    # Decode aggressive driving dataset
    aggressiveDatasetLogNames = [f for f in listdir('Dataset/Aggressive') if isfile(join('Dataset/Aggressive', f))]
    for _, logName in enumerate(aggressiveDatasetLogNames):
        logName = splitext(logName)[0]
        Path('Decodings/Aggressive/{0}'.format(logName)).mkdir(parents=True, exist_ok=True)
        dataStream = pd.read_csv("Dataset/Aggressive/{0}.csv".format(logName))
        for _, data in enumerate(dataStream.values):
            newFrame = Frame(data)
            write_decoded_mesage_to_file(db, newFrame, 'Aggressive', logName)

    # Decode passive driving dataset
    passiveDatasetLogNames = [f for f in listdir('Dataset/Passive') if isfile(join('Dataset/Passive', f))]
    for _, logName in enumerate(passiveDatasetLogNames):
        logName = splitext(logName)[0]
        Path('Decodings/Passive/{0}'.format(logName)).mkdir(parents=True, exist_ok=True)
        dataStream = pd.read_csv("Dataset/Passive/{0}.csv".format(logName))
        for _, data in enumerate(dataStream.values):
            newFrame = Frame(data)
            write_decoded_mesage_to_file(db, newFrame, 'Passive', logName)

    # Create plots for all aggressive decodings
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

    # Create plots for all passive decodings
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
