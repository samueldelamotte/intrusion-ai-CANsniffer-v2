# intrusion-ai-CANsniffer-v2

![Demo](https://github.com/samueldelamotte/intrusion-ai-CANsniffer-v2/blob/master/GitHub%20Images/demo.gif)

## Hardware Requirements

_**NOTE (1):** As the Sparkfun CAN-BUS Shield does not come with pre-soldered pin headers, you must solder these to the board yourself. Thus, you will need a soldering iron and solder to do this._

_**NOTE (2):** The Arduino library used for this setup should work with any Arduino Uno compatible CAN bus shield that uses the MCP2515 CAN controller chip. Thus, you could try a different CAN bus shield but you may need to tinker with the code._

- **(1x) Laptop (MUST running Linux or MacOS!)**
- **(2x) Sparkfun's CAN-BUS Shield:** <https://core-electronics.com.au/can-bus-shield-32856.html>
- **(2x) Arduino Uno R3:** <https://core-electronics.com.au/uno-r3.html>
- **(1x) DB9 to OBD2(RS232) Cable:** <https://core-electronics.com.au/db9-serial-rs232-obd2-cable.html>
- **(32x) 2.54cm Breakable Pin Headers:** <https://core-electronics.com.au/10-pcs-40-pin-headers-straight.html>
- **(2x) USB 2.0 Type A to B Cable:** <https://store.arduino.cc/usa/usb-2-0-cable-type-a-b>
  - *(Most vendors will provide one of these cables with your Arduino Uno purchase)*
- **(4x) Female-to-Female Solderless Breadboard Jumper Cables:** <https://core-electronics.com.au/solderless-breadboard-jumper-cable-wires-female-female-40-pieces.html>
- **(1x) 1000 Ohm Resistor**: <https://core-electronics.com.au/resistor-1k-ohm-1-4-watt-pth-20-pack-thick-leads.html>

## Software Requirements

- **Arduino IDE:** <https://www.arduino.cc/en/main/software>
- **(Seeedstudio's) CAN Bus Shield Arduino Library:** <https://github.com/Seeed-Studio/CAN_BUS_Shield>
- **Anaconda/Miniconda:** <https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html>
  - *(I prefer to use Miniconda)*

## Setup and Usage Instructions

### Part 1 - Arduino CAN Bus Sniffer

#### 1.1 - Hardware

1. Solder pin headers onto both of your Sparkfun CAN-BUS shields.

![Solder](https://github.com/samueldelamotte/intrusion-ai-CANsniffer-v2/blob/master/GitHub%20Images/3.jpg)

2. On **ONE** of the Sparkfun CAN-BUS shields, solder a single 1000 Ohm resistor (on the underneath) to its CAN-H and CAN-L pins.

![Resistor1](https://github.com/samueldelamotte/intrusion-ai-CANsniffer-v2/blob/master/GitHub%20Images/resistor1.jpg)
![Resistor2](https://github.com/samueldelamotte/intrusion-ai-CANsniffer-v2/blob/master/GitHub%20Images/resistor2.jpg)

3. Attach both Sparkfun CAN-BUS shields to an Arduino Uno. The shield will slot in perfectly with the Arduino Uno's pins.

![Fit On Arduino](https://github.com/samueldelamotte/intrusion-ai-CANsniffer-v2/blob/master/GitHub%20Images/4.jpg)

4. Put (2x) Arduino Uno + Sparkfun CAN-BUS shield into the 3D printed plastic housing.

![Housing](https://github.com/samueldelamotte/intrusion-ai-CANsniffer-v2/blob/master/GitHub%20Images/housing.jpg)

5. Attach jumper cables between both of the Arduino setups using the following instructions;

  - 5V pin to 5V pin
  - GND pin to GND pin
  - CAN-H pin to CAN-H pin
  - CAN-L pin to CAN-L pin

![Jumpers](https://github.com/samueldelamotte/intrusion-ai-CANsniffer-v2/blob/master/GitHub%20Images/jumpers.jpg)

6. Locate your car's OBD2 port, it should be somewhere near the driver or the passenger dash (**Depends on the manufacturer**). Using the DB9 -> OBD2 cable, connect your Arduino setup that includes the 1000 Ohm resistor to the OBD2 port this will be your **RECEIVER** node. The Arduino setup that does not have the 1000 Ohm resistor soldered to it will be your **SENDER** node, thus this will be used to inject intrusion packets onto the CAN bus. After following the instructions in the "1.2 - Software" section just below, you are ready to collect CAN bus data with or without intrusion attacks.

#### 1.2 - Software

1. Download and install the Arduino IDE from the link above.
2. Install Seeedstudio's Arduino library given in the link above. Instructions on how to install additional Arduino libraries can be found here (<https://www.arduino.cc/en/guide/libraries>). See the section "Importing a .zip library". *Note: If you run into any issues in the next few steps, here is a helpful link (<https://www.arduino.cc/en/guide/troubleshooting>)*
3. Found in the "CAN-Reader" directory of this GitHub is the script that we will now flash to our Arduino **RECEIVER** node. Download the .zip of this entire GitHub and extract the files to your preffered location. Now open the "CAN-Reader.ino" script in the Arduino IDE.
4. At the top of the script is a "#define" macro for the CAN_BITRATE. You will need to research what bitrate your car's CAN bus uses. I believe in most cases it should be 500 Kbps. *Possible values you can set this to are; 500, 250 or 125. Also note that our USB serial Baud Rate is set to 115200, you won't need to change this.*
5. Connect your Arduino Uno to your computer's USB port using the Type A -> Type B USB cable, the Arduino IDE should recognise your device upon doing this. To check, go to Tools -> Port -> Select Port. Ensure that it also states "Board: Arduino Uno".

![Arduino COM Port](https://github.com/samueldelamotte/intrusion-ai-CANsniffer-v2/blob/master/GitHub%20Images/1.PNG)

6. Finally, in the top left of the Arduino IDE, click "upload" and the CAN-Reader code will be flashed to your Arduino Receiver node.
7. Repeat the above steps to install any of the included intrusion attacks (located in the "Intrusions" directory of this GitHub) for the Arduino **SENDER** node. Please refer to the in-code comments for the chosen intrusion attack to setup the attack before flashing it to your sender node.

![Serial Monitor](https://github.com/samueldelamotte/intrusion-ai-CANsniffer-v2/blob/master/GitHub%20Images/2.PNG)

8. When the Arduino Receiver node is connected to your car's OBD2 port via the DB9 -> OBD2 cable, connect the Arduino to your laptop to verify that the CAN_BITRATE was set correctly and everything initialised correctly, and that the CAN bus data can be read. The Arduino IDE serial monitor will output, line by line, each CAN bus signal read.
9. To begin sniffing data, simply click the joystick on the Sparkfun CAN-BUS Shield attached to the Arduino Receiver node. The Arduino script will begin reading CAN bus traffic after a 3 second countdown. To stop reading data, simply click the joystick again. At this point you will then need to follow the next section to in this GitHub to integrate my "CAN-Monitor.py" python script so we can log and analyse CAN bus data that is sniffed via the Arduino setup.
10. To start performing the chosen intrusion attack using the Arduino Sender node, click the joystick on its Sparkfun CAN-BUS Shield (refer to the in-code comments for specific instructions regarding that attack.)

### Part 2 - Python CAN-Monitor / Sniffer Tool (CAN-Monitor&#46;py)

This tool is an improvement on the previous python tool used in (https://github.com/samueldelamotte/intrusion-ai-CANsniffer).

#### 2.1 - Important Information

- This software is designed to be used with vehicle CAN bus data that uses max payload lengths of 8.

#### 2.2 - Installation and Usage

1. Download either Anaconda or Miniconda from the link above. Install by following the instructions applicable.
2. Using terminal, 'cd' into the root directory of this GitHub.

  ```terminal
    cd ~/intrusion-ai-CANsniffer-v2
  ```

3. On your Linux / MacOS laptop, simply open a new terminal window. Now create a new Conda environment with the included "environment.yaml" file in this GitHub directory by entering the following code;

  ```terminal
    conda env create --file environment.yaml
  ```

4. The environment and all the necessary packages will be installed by conda under the environment name "CAN-Monitor". After this process is complete, we can activate our environment with the following terminal command;

  ```terminal
    conda activate CAN-Monitor
  ```

5. To launch the python script you will need to know the name of the USB port connection and your code will look like something along the lines of;

  ```terminal
    python CAN-Monitor.py /dev/tty.usbmodem1234
  ```
  - or maybe this,
  ```terminal
    python CAN-Monitor.py /dev/tty.usbbluetooth1234
  ```

  - **NOTE:** You can type 'ls /dev/tty*' into a terminal window and you'll be able to see all the open serial port connections.
  - **NOTE:** Additionally, the Arduino IDE will be able to also tell you the USB port name.

6. Hopefully the tool will load up and it will be ready to read and display CAN bus data that is read in by the Arduino setup, where it is then sent to your computers USB serial port and then displayed in a way that helps you analyse the data easier. As long as the tool is reading in the CAN bus data, it will also save each message in a .csv file within the same directory.

### Part 3 - Combining the Arduino Setup with the Python CAN-Monitor / Sniffer Tool

After completetion of part 1 and part 2 above, you can now read, analyse and collect CAN bus data from your vehicles OBD2 port.

1. Connect the Arduino **RECEIVER** node to your laptop with Type A -> Type B USB cable.
2. Connect Arduino **RECEIVER** node to your car's OBD2 port with the DB9 -> OBD2 cable.
3. Startup the Arduino IDE serial monitor tool, press reset button on the Arduino if there are any issues. This should say "Press the joystick to begin". Sometimes disconnecting and reconnecting cables may resolves certain issues.
4. On your Linux / MacOS Laptop, open terminal.
5. Make sure that you have activated you Conda environment that we created before, now we can launch the "CAN-Monitor" python script (refer to the section above if needed).
6. When you're ready to analyse / collect data, simply press the joystick on the Arduino **RECEIVER** node and wait 3 seconds for the countdown to finish.
7. When you're ready to inject your chosen intrusion attack packets onto the vehicles CAN bus, press the joystick on the Arduino **SENDER** node.
8. When you're ready to stop the intrusion attack, click the joystick again on the Arduino **SENDER** node.
9. When you're ready to stop collecting data, simply press the joystick again on the Arduino **RECEIVER** node.

### Part 4 - Analysing the Collected CAN Bus Data (CAN-Decoder&#46;py)

#### 4.1 - Important Information

- If you're lucky enough then there will be a .dbc file corresponding to your car's make and model (<https://github.com/commaai/opendbc>).
- Go to here (<https://www.vector.com/int/en/products/products-a-z/software/candb/#c104686>), then click download CANdb++ Editor. You can use this free software to parse .dbc CAN datafiles, and have them presented to you as CAN Messages and Signals. Additionally you are able to export the .dbc file as .csv which may be of use to help understand the messages of your vehicle (assuming you have a .dbc file).
- If you can't find a .dbc file for your car then the only option for you to decode the data is to reverse engineer each packet payload / message / signal, this is a very lengthy and intricate process because the CAN packets are unique to vehicle manufacturers and even differ between vehicles produced by the manufacturer. Reverse engineering would involve, for example, pushing the accelerator pedal... and doing real-time analysis using our Arduino + Python CAN monitor tool to spot bit/byte changes (displayed as blue text). If your car doesn't have a corresponding .dbc file available... then you could still figure out what does what. But, ideally you need to utilise a car that has an open source .dbc file available.

#### 4.2 - Usage

1. Download your car's appropriate .dbc file (you NEED this to decode the CAN bus data without having to reverse engineer the payloads).
2. Organise your Dataset directory so that it is structured properly i.e. passive logs are kept in the "~/intrusion-ai-CANsniffer-v2/Dataset/Passive" sub-directory and aggressive logs are kept in the "~/intrusion-ai-CANsniffer-v2/Dataset/Aggressive".
3. Using terminal (ensure your conda "CAN-Monitor" environment is active),

  ```terminal
    python CAN-Decoder.py "Dbc/hyundai_i30_2014.dbc"
  ```

- **NOTE:** The first argument to the python script is the location of the .dbc for your car data.
- **NOTE:** This python script will decoded the entirety of your dataset, thus if you require decoding specific log files one-at-a-time, then you will need to alter the code.

4. The decoded messages will be saved to the "~/intrusion-ai-CANsniffer-v2/Decodings/{Aggresive or Passive}/{log file name}.csv".
5. Decoded message Vs. time line plots will be constructed for each message and will be placed in the "~/intrusion-ai-CANsniffer-v2/Plots" directory.
