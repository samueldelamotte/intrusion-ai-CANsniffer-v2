import pandas as pd
import re, sys
import time
import struct
import cursor

class Frame:
    def __init__(self, data):
        self.frame = data
        self.id_dec = self.frame[0]
        self.len = self.frame[1]
        self.hex_data = self.frame[2:(self.len+2)]
        self.hex_byte_1 = '  '
        self.hex_byte_2 = '  '
        self.hex_byte_3 = '  '
        self.hex_byte_4 = '  '
        self.hex_byte_5 = '  '
        self.hex_byte_6 = '  '
        self.hex_byte_7 = '  '
        self.hex_byte_8 = '  '
        self.extract_hex_bytes()
        self.xPos = 0
        self.colour = ''
        self.count = 1
        self.count_per_second = 0
        self.count_per_second_storage = [0]
        self.count_per_second_in_last_3_seconds = 0

    def extract_hex_bytes(self):
        if (len(self.hex_data) == 1):
            self.hex_byte_1 = self.hex_data[0]
        elif (len(self.hex_data) == 2):
            self.hex_byte_1 = self.hex_data[0]
            self.hex_byte_2 = self.hex_data[1]
        elif (len(self.hex_data) == 3):
            self.hex_byte_1 = self.hex_data[0]
            self.hex_byte_2 = self.hex_data[1]
            self.hex_byte_3 = self.hex_data[2]
        elif (len(self.hex_data) == 4):
            self.hex_byte_1 = self.hex_data[0]
            self.hex_byte_2 = self.hex_data[1]
            self.hex_byte_3 = self.hex_data[2]
            self.hex_byte_4 = self.hex_data[3]
        elif (len(self.hex_data) == 5):
            self.hex_byte_1 = self.hex_data[0]
            self.hex_byte_2 = self.hex_data[1]
            self.hex_byte_3 = self.hex_data[2]
            self.hex_byte_4 = self.hex_data[3]
            self.hex_byte_5 = self.hex_data[4]
        elif (len(self.hex_data) == 6):
            self.hex_byte_1 = self.hex_data[0]
            self.hex_byte_2 = self.hex_data[1]
            self.hex_byte_3 = self.hex_data[2]
            self.hex_byte_4 = self.hex_data[3]
            self.hex_byte_5 = self.hex_data[4]
            self.hex_byte_6 = self.hex_data[5]
        elif (len(self.hex_data) == 7):
            self.hex_byte_1 = self.hex_data[0]
            self.hex_byte_2 = self.hex_data[1]
            self.hex_byte_3 = self.hex_data[2]
            self.hex_byte_4 = self.hex_data[3]
            self.hex_byte_5 = self.hex_data[4]
            self.hex_byte_6 = self.hex_data[5]
            self.hex_byte_7 = self.hex_data[6]
        elif (len(self.hex_data) == 8):
            self.hex_byte_1 = self.hex_data[0]
            self.hex_byte_2 = self.hex_data[1]
            self.hex_byte_3 = self.hex_data[2]
            self.hex_byte_4 = self.hex_data[3]
            self.hex_byte_5 = self.hex_data[4]
            self.hex_byte_6 = self.hex_data[5]
            self.hex_byte_7 = self.hex_data[6]
            self.hex_byte_8 = self.hex_data[7]

    def increase_count(self):
        self.count += 1

    def calc_counts_per_second(self, oldFrame):
        global START_TIME
        self.count = oldFrame.count
        self.count_per_second = (self.count / (get_current_time() - START_TIME))
        self.count_per_second_storage = oldFrame.count_per_second_storage
        self.count_per_second_storage.append(self.count_per_second)
    
    def calc_count_per_second_in_last_3_seconds(self):
        global TEMP_TIME
        if (round(get_current_time() - TEMP_TIME) == 3):
            self.count_per_second_storage = []
            TEMP_TIME = get_current_time()
        else:
            self.count_per_second_in_last_3_seconds = (sum(self.count_per_second_storage) / len(self.count_per_second_storage))

    def set_position(self, x):
        self.xPos = x

    def print(self, hex_byte_pos=[], colour=''):
        sys.stdout.write(str(self.id_dec)+'\t')
        sys.stdout.write(' '+str(self.len)+'\t')
        sys.stdout.write('| ')
        if (1 in hex_byte_pos):
            set_text_colour(colour)
        sys.stdout.write(self.hex_byte_1+' ')
        set_text_colour('RESET')
        sys.stdout.write('| ')
        if (2 in hex_byte_pos):
            set_text_colour(colour)
        sys.stdout.write(self.hex_byte_2+' ')
        set_text_colour('RESET')
        sys.stdout.write('| ')
        if (3 in hex_byte_pos):
            set_text_colour(colour)
        sys.stdout.write(self.hex_byte_3+' ')
        set_text_colour('RESET')
        sys.stdout.write('| ' )
        if (4 in hex_byte_pos):
            set_text_colour(colour)
        sys.stdout.write(self.hex_byte_4+' ')
        set_text_colour('RESET')
        sys.stdout.write('| ')
        if (5 in hex_byte_pos):
            set_text_colour(colour)
        sys.stdout.write(self.hex_byte_5+' ')
        set_text_colour('RESET')
        sys.stdout.write('| ')
        if (6 in hex_byte_pos):
            set_text_colour(colour)
        sys.stdout.write(self.hex_byte_6+' ')
        set_text_colour('RESET')
        sys.stdout.write('| ')
        if (7 in hex_byte_pos):
            set_text_colour(colour)
        sys.stdout.write(self.hex_byte_7+' ')
        set_text_colour('RESET')
        sys.stdout.write('| ')
        if (8 in hex_byte_pos):
            set_text_colour(colour)
        sys.stdout.write(self.hex_byte_8+' ')
        set_text_colour('RESET')
        sys.stdout.write('| \t')
        sys.stdout.write(str(self.count))
        sys.stdout.write('\t |\t')
        sys.stdout.write(str(round(self.count_per_second,2)))
        sys.stdout.write('\t |')
        sys.stdout.write('      \t')
        sys.stdout.write(str(round(self.count_per_second_in_last_3_seconds,2)))
        sys.stdout.write('\t\t   |')
        sys.stdout.write('\n')

def print_header():
    sys.stdout.write('\033[1m') # Bold text
    print('ID\tLEN\t  B1   B2   B3   B4   B5   B6   B7   B8       COUNT          COUNT/SEC          COUNT/SEC (LAST 3 SECS)')
    sys.stdout.write('\033[0m') # Normal text
    print('----------------------------------------------------------------------------------------------------------------------------')

def get_current_time():
    return time.time()

def clear_terminal_then_print(uniqueFrames):
    sys.stdout.write('\033[2J')     # Clear
    sys.stdout.write('\033[0;0H')   # Set cursor to 0,0
    print_header()
    for frame in uniqueFrames:
        frame.print()

def move_to_line_in_terminal_then_print(newFrame, hex_byte_pos):
    sys.stdout.write('\033[s')      # Save cursor position
    # Move cursor to line that has this frame
    sys.stdout.write('\033[{0};0H'.format(newFrame.xPos+3))
    sys.stdout.write("\033[K")      # Clear terminal line
    newFrame.print(hex_byte_pos, 'CYAN')
    sys.stdout.write('\033[u')      # Re-load saved cursor position

def set_text_colour(colour):
    colours = {
        1: '\033[1;31m',
        2: '\033[1;34m',
        3: '\033[1;36m',
        4: '\033[1;32m',
        5: '\033[0;0m'
    }
    if (colour == 'RED'):
        sys.stdout.write(colours.get(1))
    elif (colour == 'BLUE'):
        sys.stdout.write(colours.get(2))
    elif (colour == 'CYAN'):
        sys.stdout.write(colours.get(3))
    elif (colour == 'GREEN'):
        sys.stdout.write(colours.get(4))
    elif (colour == 'RESET'):
        sys.stdout.write(colours.get(5))
    else:
        raise Exception('Invalid colour option')

def get_index_in_list(uniqueFrames, id_dec):
    for index,frame in enumerate(uniqueFrames):
        if frame.id_dec == id_dec:
            return index
    return -1

def sort_list_by_id(uniqueFrames):
    return sorted(uniqueFrames, key=lambda x: x.id_dec)

def check_for_byte_changes(newFrame, oldFrame):
    differentBytes = []
    for ii in range(len(newFrame.hex_data)):
        if newFrame.hex_data[ii] != oldFrame.hex_data[ii]:
            differentBytes.append(ii+1)
    return differentBytes

def read_csv_file(filePath):
    return pd.read_csv(filePath)

def listen_to_usb_serial(dataStream):
    uniqueFrames = []

    for _, data in enumerate(dataStream.values):
        # Imitate delay
        time.sleep(0.01)

        # Create a new frame object
        newFrame = Frame(data)

        # Check if we have seen this frame before
        index = get_index_in_list(uniqueFrames, newFrame.id_dec)
        if (index == -1): # If no, append to uniqueFrames, sort then print
            uniqueFrames.append(newFrame)
            uniqueFrames = sort_list_by_id(uniqueFrames)
            clear_terminal_then_print(uniqueFrames)
        else: # If yes, overwrite frame in uniqueFrames
            newFrame.set_position(index)
            newFrame.calc_counts_per_second(uniqueFrames[index])
            newFrame.calc_count_per_second_in_last_3_seconds()
            newFrame.increase_count()
            differentBytes = check_for_byte_changes(newFrame, uniqueFrames[index])
            uniqueFrames[index] = newFrame
            move_to_line_in_terminal_then_print(uniqueFrames[index], differentBytes)


if __name__ == "__main__":
    # Start time for collecting data
    START_TIME = get_current_time()
    TEMP_TIME = get_current_time()
    cursor.hide()
    sys.stdout.write("\x1b[8;{rows};{cols}t".format(rows=50, cols=80))
    print_header()
    sys.stdout.write('\033[s')
    dataStream = read_csv_file("data2.csv")
    listen_to_usb_serial(dataStream)