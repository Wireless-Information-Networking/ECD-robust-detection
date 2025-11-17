#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: compact_daily_csv.py
Author: Javier del Río
Date: 2025-11-17
Description: 
    Compacts multiple CSV files from the same day into a single CSV file per directory.
    Preserves specified columns and adds custom constant columns for each source file.
    
    Preserved columns: eventNum, peakRssi, phase, channel, idHex, time_reader, power
    Custom columns: offsetDirection, offset, moveType (configurable per file)

License: MIT License
"""

import pandas as pd
import os
from typing import Dict, List, Optional
from csv_data_loader import get_csv_files_from_directory

def format_offset_for_filename(value: float) -> str:
    """
    Format float with one decimal only if needed (e.g., 2.5 -> '2.5', 10.0 -> '10').
    """
    return f"{value:.1f}".rstrip('0').rstrip('.')

def compact_csv_files_by_directory(
    base_directory: str,
    output_directory: str,
    custom_values: Optional[Dict[str, Dict[str, any]]] = None,
    columns_to_preserve: Optional[List[str]] = None
):
    """
    Compacts all CSV files in each subdirectory into a single CSV per directory.
    
    :param base_directory: Base directory containing subdirectories with CSV files
    :param output_directory: Directory to save the compacted CSV files
    :param custom_values: Dictionary mapping CSV filenames to their custom column values
                         Example: {'file1.csv': {'offsetDirection': 0, 'offset': 10, 'moveType': 'active'}}
    :param columns_to_preserve: List of column names to preserve from source CSVs
    """
    
    # Default columns to preserve if not specified
    if columns_to_preserve is None:
        columns_to_preserve = [
            'eventNum', 'peakRssi', 'phase', 'channel', 
            'idHex', 'time_reader', 'power'
        ]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Get all subdirectories (each represents a day)
    subdirectories = [
        d for d in os.listdir(base_directory) 
        if os.path.isdir(os.path.join(base_directory, d))
    ]
    
    print(f"\n{'='*60}")
    print("CSV COMPACTION BY DIRECTORY")
    print("="*60)
    print(f"Base directory: {base_directory}")
    print(f"Output directory: {output_directory}")
    print(f"Subdirectories found: {len(subdirectories)}")
    print(f"Columns to preserve: {columns_to_preserve}")
    print("="*60)
    
    # Process each subdirectory (day)
    for subdir in sorted(subdirectories):
        subdir_path = os.path.join(base_directory, subdir)
        
        print(f"\nProcessing directory: {subdir}")
        
        # Get all CSV files in this subdirectory
        csv_files = get_csv_files_from_directory(subdir_path, recursive=False)
        
        if not csv_files:
            print(f"  No CSV files found in {subdir}")
            continue
        
        print(f"  Found {len(csv_files)} CSV files")
        
        # List to store DataFrames from each CSV
        dataframes = []
        
        # Process each CSV file
        for csv_file in csv_files:
            try:
                # Read CSV file
                df = pd.read_csv(csv_file)
                
                # Clean column names (remove whitespace)
                df.columns = df.columns.str.strip()
                
                # Clean idHex from special characters
                if 'idHex' in df.columns:
                    df['idHex'] = df['idHex'].astype(str).str.replace(r'\s+', '', regex=True)
                
                # Filter only the columns we want to preserve
                available_columns = [col for col in columns_to_preserve if col in df.columns]
                df_filtered = df[available_columns].copy()
                                
                # Add custom constant columns if specified for this file
                if custom_values and os.path.basename(csv_file) in custom_values:
                    custom_cols = custom_values[os.path.basename(csv_file)]
                    for col_name, col_value in custom_cols.items():
                        df_filtered[col_name] = col_value
                else:
                    # Do nothing
                    pass
                
                dataframes.append(df_filtered)
                
                print(f"    ✓ Loaded: {os.path.basename(csv_file)} ({len(df_filtered)} rows)")
                
            except Exception as e:
                print(f"    ✗ Error loading {os.path.basename(csv_file)}: {e}")
                continue
        
        # Combine all DataFrames from this directory
        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            
            # Generate output filename
            output_filename = f"{subdir}_compact.csv"
            output_path = os.path.join(output_directory, output_filename)
            
            # Save compacted CSV
            combined_df.to_csv(output_path, index=False)
            
            print(f"\n  ✅ Compacted file saved: {output_filename}")
            print(f"     Total rows: {len(combined_df)}")
            print(f"     Columns: {list(combined_df.columns)}")
        else:
            print(f"  ⚠ No data to compact for {subdir}")
    
    print(f"\n{'='*60}")
    print("COMPACTION COMPLETED")
    print("="*60)


def create_custom_values_config(csv_files: List[str]) -> Dict[str, Dict[str, any]]:
    """
    Interactive function to create custom column values configuration.
    
    :param csv_files: List of CSV file paths
    :return: Dictionary mapping filenames to their custom column values
    """
    custom_values = {}
    
    print("\n" + "="*60)
    print("CUSTOM VALUES CONFIGURATION")
    print("="*60)
    print("Configure custom column values for each CSV file.")
    print("Press Enter to skip a file or use default values.\n")
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        print(f"\nFile: {filename}")
        
        # Get custom values for this file
        try:
            direction_offset = input(f"  offsetDirection (default=0): ").strip()
            direction_offset = float(direction_offset) if direction_offset else 0.0
            
            offset = input(f"  offset (default=0): ").strip()
            offset = float(offset) if offset else 0.0
            
            moveType = input(f"  moveType (default='unknown'): ").strip()
            moveType = moveType if moveType else 'unknown'
            
            custom_values[filename] = {
                'offsetDirection': direction_offset,
                'offset': offset,
                'moveType': moveType
            }
            
            print(f"  ✓ Configured: {custom_values[filename]}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}. Using defaults.")
            custom_values[filename] = {
                'offsetDirection': 0.0,
                'offset': 0.0,
                'moveType': 'unknown'
            }
    
    return custom_values


def compact_with_predefined_config(
    base_directory: str,
    output_directory: str,
    config: Dict[str, Dict[str, any]]
):
    """
    Compact CSV files using a predefined configuration dictionary.
    
    :param base_directory: Base directory with subdirectories containing CSV files
    :param output_directory: Directory to save compacted files
    :param config: Predefined configuration with custom values per file
    
    Example config:
    {
        'dynamic.csv': {'offsetDirection': 0, 'offset': 0, 'moveType': 'dynamic'},
        'static.csv': {'offsetDirection': 0, 'offset': 0, 'moveType': 'static'}
    }
    """
    compact_csv_files_by_directory(
        base_directory=base_directory,
        output_directory=output_directory,
        custom_values=config
    )


def main_interactive():
    """
    Interactive main function for CSV compaction.
    """
    print("\n" + "="*60)
    print("DAILY CSV COMPACTION TOOL")
    print("="*60)
    
    # Get base directory
    base_dir = input("\nEnter base directory path (or press Enter for 'data/'): ").strip()
    base_dir = base_dir if base_dir else 'data/'
    
    if not os.path.exists(base_dir):
        print(f"❌ Error: Directory '{base_dir}' does not exist.")
        return
    
    # Get output directory
    output_dir = input("Enter output directory path (or press Enter for 'output_data/compacted/'): ").strip()
    output_dir = output_dir if output_dir else 'output_data/compacted/'
    
    # Ask if user wants to configure custom values
    configure_custom = input("\nConfigure custom values for each file? (y/n, default=n): ").strip().lower()
    
    custom_values = None
    if configure_custom == 'y':
        # Get all CSV files from all subdirectories
        all_csv_files = []
        for subdir in os.listdir(base_dir):
            subdir_path = os.path.join(base_dir, subdir)
            if os.path.isdir(subdir_path):
                all_csv_files.extend(get_csv_files_from_directory(subdir_path, recursive=False))
        
        if all_csv_files:
            custom_values = create_custom_values_config(all_csv_files)
    
    # Execute compaction
    compact_csv_files_by_directory(
        base_directory=base_dir,
        output_directory=output_dir,
        custom_values=custom_values
    )
    
    print(f"\n✅ Process completed. Check output files in: {output_dir}")


def main_example():
    """
    Example main function with predefined configuration.
    """
    # Define base configuration
    base_directory = 'data/'
    output_directory = 'output_data/compacted/'
    
    # Predefined custom values for specific files
    # Adjust these values according to your dataset
    custom_config = {
        'dynamic.csv': {
            'offsetDirection': 'vertical',
            'offset': 2.0,
            'moveType': 'dynamic'
        },
        'static.csv': {
            'offsetDirection': 'vertical',
            'offset': 2.0,
            'moveType': 'static'
        },
        'magic_mike.csv': {
            'offsetDirection': 'vertical',
            'offset': 2.0,
            'moveType': 'dynamic'
        }
    }
    # Load All possible form offset left and right
    for i in range(1, 11):
        for left_right in ['left', 'right']:
            text = "der" if left_right == 'right' else 'izq'
            for offset in [0, 2.5, 5.0, 7.5, 10.0]:
                filename = f'{format_offset_for_filename(offset)}cm-{text}-{i}.csv'
                custom_config[filename] = {
                    'offsetDirection': left_right,
                    'offset': offset,
                    'moveType': 'dynamic'
                }

    # Load All possible form offset horizontal and vertical
    for i in range(1, 11):
        for horiz_vert in ['horizontal', 'vertical']:
            text = "horiz" if horiz_vert == 'horizontal' else 'vert'
            for offset in [0, 2.5, 5.0, 7.5, 10.0]:
                # format offset with one decimal only if needed
                filename = f'{format_offset_for_filename(offset)}cm{text}-{i}.csv'
                custom_config[filename] = {
                    'offsetDirection': horiz_vert,
                    'offset': offset,
                    'moveType': 'dynamic'
                }

    # Load all static files
    for i in range(1, 11):
        filename = f'static-espurio-{i}.csv'
        custom_config[filename] = {
            'offsetDirection': 'horizontal',
            'offset': 2.5,
            'moveType': 'static'
        }
    
    print("\n" + "="*60)
    print("DAILY CSV COMPACTION - PREDEFINED CONFIG")
    print("="*60)
    print(f"\nBase directory: {base_directory}")
    print(f"Output directory: {output_directory}")
    print(f"\nCustom configuration:")
    for filename, values in custom_config.items():
        print(f"  {filename}: {values}")
    
    # Execute compaction
    compact_with_predefined_config(
        base_directory=base_directory,
        output_directory=output_directory,
        config=custom_config
    )
    
    print(f"\n✅ Process completed. Check output files in: {output_directory}")


if __name__ == "__main__":
    # Choose execution mode
    print("\n" + "="*60)
    print("CSV COMPACTION SCRIPT")
    print("="*60)
    print("\nExecution modes:")
    print("  1. Interactive mode (configure via prompts)")
    print("  2. Example mode (use predefined configuration)")
    
    mode = input("\nSelect mode (1 or 2, default=2): ").strip()
    
    if mode == '1':
        main_interactive()
    else:
        main_example()