# Data File Naming Convention

This document explains the naming convention used for data files in this project. Each part of the filename encodes specific information about the data collection parameters.

## Filename Structure

A typical filename is structured as follows:

```
<DATE>_<TIME>_<TAG_MODEL>_<FREQUENCY>_<DISTANCE>_<OBSTACLE>_<POWER>_<POSITION>
```

### Components

- **DATE**: The initial number represents the date in `YYYYMMDD` format.  
    Example: `20250616` means June 16, 2025.

- **TIME**: The next number is the time of reading in `HHMMSS` format.  
    Example: `130023` means 13:00:23 (1:00:23 PM).

- **TAG_MODEL**: The model of the RFID tag used.  
    Example: `Belt0001`.

- **FREQUENCY**: The reading frequency in MHz.  
    Example: `865.7MHz`.

- **DISTANCE**: The distance between the antenna and the center of the tag, in meters.  
    Example: `0.025m` means 2.5 cm.

- **OBSTACLE**: Indicates the presence of an obstacle.  
    Example: `None` means no obstacle is present, but other values may indicate specific obstacles.

- **POWER**: The emission power of the reader, in dBm.  
    Example: `29.2dBm` is the maximum reading power used.

- **POSITION**: Describes the spatial arrangement and orientation of the tag relative to the antenna.

## Reference System

The antenna serves as the reference for the coordinate system, which is Cartesian:

- **y**: Perpendicular to the plane of the planar antenna.
- **z**: Longitudinal component of the antenna (along its long dimension).
- **x**: Perpendicular to z and y, representing the short dimension.

### Position Codes

- **pxz**: The tag is in the xz plane, i.e., parallel to the antenna.
- **dx**: The tag is displaced along the x-axis.
- **sz**: The longest side of the tag is oriented along the z-axis.

These codes can be combined to describe the tag's exact position and orientation during data collection.
