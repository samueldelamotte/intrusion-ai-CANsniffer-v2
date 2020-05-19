# intrusion-ai-CANsniffer

![Demo](https://github.com/samueldelamotte/intrusion-ai-CANsniffer-v2/blob/master/GitHub%20Images/demo.gif)

_**NOTE (1):** As the Sparkfun CAN-BUS Shield does not come with pre-soldered pin headers, you must solder these to the board yourself. Thus, you will need a soldering iron and solder to do this._

_**NOTE (2):** The Arduino library used for this setup should work with any Arduino Uno compatible CAN bus shield that uses the MCP2515 CAN controller chip. Thus, you could try a different CAN bus shield but you may need to tinker with the code i.e. the chip select pin for Sparkfun's module is pin 10, but this may be differnet on other boards. Additionally, Sparkfun's module comes with a micro SD card slot so we can log the data as it is captured and also a joystick which I use to start/stop data logging. This data could potentially be done by the python script instead but has not been implemented by myself yet._

## Hardware requirements

- **Sparkfun's CAN-BUS Shield:** <https://core-electronics.com.au/can-bus-shield-32856.html>
- **Arduino Uno R3:** <https://core-electronics.com.au/uno-r3.html>
- **DB9 to OBD2(RS232) Cable:** <https://core-electronics.com.au/db9-serial-rs232-obd2-cable.html>
- **(32x) 2.54cm Breakable Pin Headers:** <https://core-electronics.com.au/10-pcs-40-pin-headers-straight.html>
- **Any brand 4+ GB micro SD card:** <https://www.officeworks.com.au/shop/officeworks/p/sandisk-ultra-16gb-micro-sdhc-memory-card-sdsq16gb>
- **Any brand micro SD card reader:** <https://www.officeworks.com.au/shop/officeworks/p/verbatim-multi-card-reader-with-usb-2-0-hub-vmcr65676>
- **USB 2.0 Type A to B Cable:** <https://store.arduino.cc/usa/usb-2-0-cable-type-a-b>
  - *(Most vendors will provide one of these cables with your Arduino Uno purchase)*

## Software requirements

- **Arduino IDE:** <https://www.arduino.cc/en/main/software>
- **(Seeedstudio's) CAN Bus Shield Arduino Library:** <https://github.com/Seeed-Studio/CAN_BUS_Shield>
- **Anaconda/Miniconda:** <https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html>
  - *(I prefer to use Miniconda)*

## Setup instructions

### Part 1 - Arduino CAN Bus Sniffer

#### 1.1 - Hardware

1. Solder pin headers to your Sparkfun CAN-BUS shield.

![Solder](https://github.com/samueldelamotte/intrusion-ai-CANsniffer-v2/blob/master/GitHub%20Images/3.jpg)

2. The shield should now slot in perfectly to your Arduino Uno's pins.

![Fit On Arduino](https://github.com/samueldelamotte/intrusion-ai-CANsniffer-v2/blob/master/GitHub%20Images/4.jpg)

3. Slot in your micro SD card on the shield. You should hear a springy click when it is correctly in. For your reference, the joystick next to the SD card slot is used in the Arduino code to begin data logging.

![SD](https://github.com/samueldelamotte/intrusion-ai-CANsniffer-v2/blob/master/GitHub%20Images/5.jpg)

4. Locate your car's OBD2 port, it should be somewhere near the driver or the passenger dash (**Depends on the manufacturer**). Using the DB9 -> OBD2 cable, connect your Arduino setup to the OBD2 port. After following the instructions in the "1.2 - Software" section just below, you are ready to collect CAN bus data.

#### 1.2 - Software

1. Download and install the Arduino IDE from the link above.
2. Install Seeedstudio's Arduino library given in the link above. Instructions on how to install additional Arduino libraries can be found here (<https://www.arduino.cc/en/guide/libraries>). See the section "Importing a .zip library". *Note: If you run into any issues in the next few steps, here is a helpful link (<https://www.arduino.cc/en/guide/troubleshooting>)*
3. Found in the "CAN-Logger" directory of this GitHub is the script that we will now flash to our Arduino Uno. Download the .zip of this entire GitHub and extract the files to your preffered location. Now open the "CAN-Logger.ino" script in the Arduino IDE.
4. At the top of the script is a "#define" macro for the CAN_BITRATE. You will need to research what bitrate your car's CAN bus uses. I believe in most cases it should be 500 Kbps. *Possible values you can set this to are; 500, 250 or 125. Also note that our USB serial Baud Rate is set to 115200, you won't need to change this.*
5. Connect your Arduino Uno to your computer's USB port using the Type A -> Type B USB cable, the Arduino IDE should recognise your device upon doing this. To check, go to Tools -> Port -> Select Port. Ensure that it also states "Board: Arduino Uno".

![Arduino COM Port](https://github.com/samueldelamotte/intrusion-ai-CANsniffer-v2/blob/master/GitHub%20Images/1.PNG)

6. Finally, in the top left of the Arduino IDE, click "upload" and the code will be flashed to your Arduino Uno.
7. To check if the flash worked correctly open the Serial monitor in the Arduino IDE (found in Tools) and ensure your Serial monitor's Baud Rate is set to 115200. The output to your serial monitor should look the same as the image below. Sometimes pressing the reset switch on the Arduino Uno's board or re-opening the tool can fix serial monitor output issues.

![Serial Monitor](https://github.com/samueldelamotte/intrusion-ai-CANsniffer-v2/blob/master/GitHub%20Images/2.PNG)

8. To bein logging data, simply click the joystick on the Sparkfun CAN-BUS Shield. The Arduino script will begin reading and storing all CAN bus traffic after a 3 second countdown. To stop logging data, simply click the joystick again. *Note that the data will be stored on the micro SD card as a simple text file named "data.txt". Every new CAN signal is appended to the file, if you want to keep data from a specific driving period, then make sure you copy the data.txt file to your computer, so trip data doesn't get mixed.*.
9. When the Arduino is connected to your car's OBD2 port via the DB9 -> OBD2 cable, I recommend also connecting the Arduino Uno to your laptop to verify that the CAN_BITRATE was set correctly and everything initialised correctly, and that the CAN bus data can be read. The Arduino IDE serial monitor will output, line by line, each CAN bus signal read.

### Part 2 - Python CAN Monitor Tool

#### 2.1 - Acknowledegment

Thankfully a really handy python script has already been made. I do intend to make some improvements to this tool / potentially re-write the tool to aid in analysing CAN bus data, in the near future.

This tool can be found here <https://github.com/alexandreblin/python-can-monitor>.

#### 2.2 - Installation

1. Download either Anaconda or Miniconda from the link above. Install by following the instructions applicable.
2. If on Windows, open the Anaconda Command Prompt. If on Linux / MacOS, simply open a new terminal window. Now create a new Conda environment by entering the following code;

  ```terminal
    conda create -n <yourEnvName> python=3.7
  ```

Type 'y' to install Conda packages for the environment. After this, we can activate our environment with the following terminal command;

  ```terminal
    conda activate <your env name>
  ```

3. Skip this step if using Linux / MacOS. To run this tool on Windows you will need to install the unofficial Windows "curses" module binary. This can be found here <https://www.lfd.uci.edu/~gohlke/pythonlibs/#curses>. Download the latest .whl binary for your system. Once downloaded, in the Anaconda Command Prompt (with your new Conda environment activated), 'cd' to where you just downloaded this binary. Then use the following code in terminal to install the binary in your environment;

  ```terminal
    pip install <.whl package file name>
  ```

4. Download the "python-can-monitor" GitHub .zip file from the link above and extract the contents to your preferred folder location. Depending on your OS, using Anaconda Command Prompt or terminal, 'cd' into the folder. You can follow the instructions on their GitHub to run the python script. To launch their python script your code will look like something along the lines of;
Handy tip, you can figure out which USB port your Arduino Uno is on by lookking at the Tool dropdown in the Arduino IDE.
If on Windows:

  ```terminal
    canmonitor COM8 115200
  ```

If on Linux / MacOS:
You can type 'ls /dev/tty*' into a terminal window and you'll be able to see all the open serial port connections.

  ```terminal
    canmonitor /dev/tty.usbmodem1234 115200     or maybe,
    canmonitor /dev/tty.usbbluetooth1234 115200
  ```

5. Hopefully the tool will load up and it will be ready to read and display CAN bus data that is read in by the Arduino setup, where it is then sent to your computers USB serial port and then displayed in a way that helps you analyse the data easier.

### Part 3 - Combining the Arduino Setup with the Python CAN Monitor Tool

After completetion of part 1 and part 2 above, you can now read, analyse and collect CAN bus data from your vehicles OBD2 port.

1. Connect Arduino setup to your laptop with Type A -> Type B USB cable.
2. Connect Arduino setup to your car's OBD2 port with the DB9 -> OBD2 cable.
3. Startup the Arduino IDE serial monitor tool, press reset button on Arduino if there are any issues. This should say "Press the joystick to begin".
4. If on Windows, open Anaconda Command Prompt. If on Linux / MacOS, open terminal.
5. Make sure that you have activated you Conda environment that we created before, now we can launch the "Python CAN monitor tool" python script.
6. When you're ready to analyse / collect data, simply press the joystick and wait 3 seconds for the countdown to finish.
7. When you're ready to stop collecting data, simply press the joystick again.
8. Remove micro SD card contents, then rinse and repeat.

### Part 4 - Analysing the CAN Bus Data
If you're lucky enough then there will be a .dbc file corresponding to your car's make and model <https://github.com/commaai/opendbc>.

Go to here <https://www.vector.com/int/en/products/products-a-z/software/candb/#c104686>, then click download CANdb++ Editor. You can use this free software to parse .dbc CAN datafiles, and have them presented to you as CAN Messages and Signals.

Currently the only way to figure out what CAN Message ID corresponds to, for example, the accelerator pedal position... is to physically push in the pedal and then cross reference the .dbc Message ID with the Message IDs being read in via our Arduino + Python CAN monitor tool. If your car doesn't have a corresponding .dbc file available... then you could still figure out what does what. But, reverse engineering something like this is quite difficult.

Modifications to the tool are in the works... to make it easier to spot what CAN Message is altered when a specific action is performed.
