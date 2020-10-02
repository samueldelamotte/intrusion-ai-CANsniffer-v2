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
    df = pd.read_csv(r'Dataset\Processed\dataset.csv', index_col=None, header=0)
    x = len(df)
    AGGRESSIVE_DIR = r'Dataset\Aggressive'
    PASSIVE_DIR = r'Dataset\Passive'
    SAVE_DIR = r'Dataset\Processed'

    aggressive_logs = glob.glob(AGGRESSIVE_DIR + '/*.csv')
    passive_logs = glob.glob(PASSIVE_DIR + '/*.csv')

    # Concatenate all logs
    tmp = []
    for file in passive_logs:
        df = pd.read_csv(file, index_col=None, header=0)
        del df['time_stamp']                                    # Delete time_stamp column as it is not necessary
        df['class_label'] = '0'                                 # Add new column with class_label, 0 => passive
        df_processed = process(df, file)                        # Process data
        tmp.append(df_processed)                                # Append to temporary list
    for file in aggressive_logs:
        df = pd.read_csv(file, index_col=None, header=0)
        del df['time_stamp']                                    # Delete time_stamp column as it is not necessary
        df['class_label'] = '1'                                 # Add new column with class_label, 1 => aggressive
        df_processed = process(df, file)                        # Process data
        tmp.append(df_processed)                                # Append to temporary list
    df = pd.concat(tmp, axis=0, ignore_index=True)

    # Save pre-processed concatenated logs to one file
    if not os.path.exists(os.path.join(SAVE_DIR)):
        os.mkdir(os.path.join(SAVE_DIR))
    df.to_csv(SAVE_DIR + r'\dataset.csv', index=False)
