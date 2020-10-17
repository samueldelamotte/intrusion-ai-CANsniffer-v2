import h5py
import matplotlib.pyplot as plt
import pandas as pd
import cantools
import sys
import numpy as np
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


# def write_decoded_mesage_to_file(db, frame, logName):
#     hexPayload = ''
#     for byte in frame.hex_data:
#         hexPayload += byte
#     hexPayloadBytes = bytearray.fromhex(hexPayload)
#     try:
#         decodedFrame = db.decode_message(frame.id_dec, hexPayloadBytes)
#         for key, value in decodedFrame.items():
#             stringToWrite = key + "," + str(value) + "," + str(frame.time_stamp) + "\n"
#             m = open("Decodings/X/{0}/{1}.csv".format(logName, key), "a")
#             m.write(stringToWrite)
#             m.close
#     except:
#         pass

def write_decoded_mesage_to_file(data, logName):
    stringToWrite = 'EngineSpeed_CAN,'
    x = 0
    for i in data:
        if x == 0:
            stringToWrite = stringToWrite + str(i)
            x = 1
        else:
            stringToWrite = stringToWrite + ',' + str(i) + "\n"
            m = open("Decodings/X/{0}/{1}.csv".format(logName, "EngineSpeed_CAN"), "a")
            m.write(stringToWrite)
            x = 0
            m.close


if __name__ == "__main__":
    # # Load the database .dbc file
    # try:
    #     db = cantools.database.load_file("DBC/")
    # except:
    #     print("Could not load .dbc file at '{0}'".format(sys.argv[1]))
    #     quit()

    # Sets the matplotlib plot styles
    plt.style.use('seaborn-whitegrid')

    # # Decode aggressive driving dataset
    # logNames = [f for f in listdir('Other Datasets/X') if isfile(join('Other Datasets/X', f))]
    # for _, logName in enumerate(logNames):
    #     logName = splitext(logName)[0]
    #     Path('Decodings/X/{0}'.format(logName)).mkdir(parents=True, exist_ok=True)
    #     dataStream = pd.read_csv("Other Datasets/X/{0}.csv".format(logName))
    #     for _, data in enumerate(dataStream.values):
    #         write_decoded_mesage_to_file(data, logName)

    # Create plots for all aggressive decodings
    aggressiveDecodingsDirNames = [f for f in listdir('Decodings/Osclab') if isdir(join('Decodings/Osclab', f))]
    for _, logName in enumerate(aggressiveDecodingsDirNames):
        logName = splitext(logName)[0]
        aggressiveDecodingsFileNames = [f for f in listdir('Decodings/Osclab/{0}'.format(logName)) if isfile(join('Decodings/Osclab/{0}'.format(logName), f))]
        for _, fileName in enumerate(aggressiveDecodingsFileNames):
            data = pd.read_csv("Decodings/Osclab/{0}/{1}".format(logName, fileName))
            readings = data.values[:,1]
            timestamps = data.values[:,2]
            plt.title("{} ({})".format(logName,fileName))
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.plot(timestamps, readings)
            Path('Plots/Osclab/{0}'.format(logName)).mkdir(parents=True, exist_ok=True)
            plt.savefig('Plots/Osclab/{0}/{1}.png'.format(logName,fileName), bbox_inches='tight')
            plt.clf()