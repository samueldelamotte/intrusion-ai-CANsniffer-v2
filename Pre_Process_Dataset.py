import pandas as pd
import glob
import os


def process(data, file):
    v = round(len(data) * 0.1)
    p = 10
    c = 0
    # Pre-process concatenated data
    for x, y in enumerate(data.values):
        numOfBytes = y[1]
        if numOfBytes != 8:
            for i, j in enumerate(y):                                   # Remove timestamps leftover
                if '.' in str(j):
                    data.at[x, 'byte{0}'.format(numOfBytes+1)] = ''
        for byte in range(1, numOfBytes+1):                             # Convert hex_data to decimal
            hex_value = data.at[x, 'byte{0}'.format(byte)]
            data.at[x, 'byte{0}'.format(byte)] = int(hex_value, 16)
        if c == v:
            print('{0}% processed for "{1}"'.format(p, file))
            p += 10
            c = 0
        else:
            c += 1
    return data


if __name__ == '__main__':
    AGGRESSIVE_DIR = os.path.join('Dataset', 'Aggressive')
    PASSIVE_DIR = os.path.join('Dataset', 'Passive')
    PASSIVE_SAVE_DIR = os.path.join('Dataset', 'Processed', 'Passive')
    AGGRESSIVE_SAVE_DIR = os.path.join('Dataset', 'Processed', 'Aggressive')
    aggressive_logs = glob.glob(os.path.join(AGGRESSIVE_DIR, '*.csv'))
    passive_logs = glob.glob(os.path.join(PASSIVE_DIR, '*.csv'))

    # Concatenate passive logs
    passiveTmp = []
    for file in passive_logs:
        df = pd.read_csv(file, index_col=None, header=0)
        del df['time_stamp']                                    # Delete time_stamp column as it is not necessary
        df['class_label'] = '0'                                 # Add new column with class_label, 0 => passive
        df_processed = process(df, file)                        # Process data
        passiveTmp.append(df_processed)                         # Append to temporary list
    df1 = pd.concat(passiveTmp, axis=0, ignore_index=True)

    # Concatenate aggressive logs
    aggressiveTmp = []
    for file in aggressive_logs:
        df = pd.read_csv(file, index_col=None, header=0)
        del df['time_stamp']                                    # Delete time_stamp column as it is not necessary
        df['class_label'] = '1'                                 # Add new column with class_label, 1 => aggressive
        df_processed = process(df, file)                        # Process data
        aggressiveTmp.append(df_processed)                      # Append to temporary list
    df2 = pd.concat(aggressiveTmp, axis=0, ignore_index=True)

    # Save pre-processed passive concatenated logs to one file
    if not os.path.exists(PASSIVE_SAVE_DIR):
        os.mkdir(os.path.join(PASSIVE_SAVE_DIR))
    df1.to_csv(os.path.join(PASSIVE_SAVE_DIR, 'passive-dataset.csv'), index=False)

    # Save pre-processed aggressive concatenated logs to one file
    if not os.path.exists(os.path.join(AGGRESSIVE_SAVE_DIR)):
        os.mkdir(os.path.join(AGGRESSIVE_SAVE_DIR))
    df2.to_csv(os.path.join(AGGRESSIVE_SAVE_DIR, 'aggressive-dataset.csv'), index=False)
