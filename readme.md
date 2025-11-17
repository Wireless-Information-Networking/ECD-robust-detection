# ECD-robust-detection

## Overview

This project aims to reduce, clean, and prevent errors from a commercial RFID reader, specifically the FX7500, by leveraging a Confinement Field Device (CFD). The CFD is essentially an antenna with losses on the order of 1 dB/cm, ensuring that there are no signal leaks or interference with nearby devices. This results in a highly confined electromagnetic field, allowing effective reading of RFID tags only within a few centimeters of the CFD.

## Motivation

Commercial RFID readers can produce erroneous readings due to environmental factors, signal reflections, and fluctuations in antenna gain and directivity. By using a CFD, we minimize these issues, but occasional errors or sporadic tag readings may still occur.

## Objectives

- Develop software to process the output from the FX7500 reader.
- Use the reader's RSSI (Received Signal Strength Indicator), phase information, and the temporal relationship between measurements.
- Filter out erroneous readings and sporadically detected tags caused by fluctuations in the CFD's gain and directivity.

## Features

- Real-time filtering of RFID tag readings.
- Error detection based on signal strength, phase, and timing.
- Improved reliability for applications requiring precise tag detection in confined spaces.

## Usage

1. Connect the FX7500 reader to the CFD.
2. Run the software to process and filter tag readings.
3. Analyze the cleaned data for your application.

## License

MIT License

## Contact

For questions or contributions, please open an issue or submit a pull request.
