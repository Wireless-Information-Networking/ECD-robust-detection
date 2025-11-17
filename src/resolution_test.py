#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: resolution_test.py
Author: Javier del Río
Date: 2025-10-21
Description: 
    Device resolution analysis for RFID tag measurement systems.
    Calculates minimum differences between consecutive measurements for RSSI, phase, and time
    across all available data files to determine the measurement resolution of the device.

License: MIT License
Dependencies: numpy, csv_data_loader (local)
"""

import numpy as np
import os
from typing import Dict, List, Tuple
from csv_data_loader import extract_tag_data, get_csv_files_from_directory

class ResolutionAnalyzer:
    """
    Analyzes measurement resolution by finding minimum differences between consecutive samples.
    """
    
    def __init__(self):
        """Initialize the analyzer with infinite minimum values."""
        self.min_rssi_diff = float('inf')
        self.min_phase_diff = float('inf')
        self.min_time_diff = float('inf')
        
        # Track where minimums were found
        self.min_rssi_source = ""
        self.min_phase_source = ""
        self.min_time_source = ""
        
        # RSSI range tracking
        self.min_rssi_value = float('inf')
        self.max_rssi_value = float('-inf')
        self.min_rssi_value_source = ""
        self.max_rssi_value_source = ""
        
        # Statistics
        self.total_files_processed = 0
        self.total_tags_processed = 0
        self.total_samples_processed = 0
        self.files_with_errors = []
        
    def analyze_tag_data(self, tag_data: Dict[str, np.ndarray], source_info: str) -> Dict[str, float]:
        """
        Analyze a single tag's data for minimum differences and RSSI range.
        
        :param tag_data: Dictionary with 'timestamp', 'rssi', 'phase' arrays
        :param source_info: Information about the data source (file + tag)
        :return: Dictionary with minimum differences found in this tag
        """
        results = {
            'min_rssi_diff': float('inf'),
            'min_phase_diff': float('inf'),
            'min_time_diff': float('inf'),
            'min_rssi_value': float('inf'),
            'max_rssi_value': float('-inf')
        }
        
        # Ensure data is in numpy arrays and has sufficient samples
        for key in ['timestamp', 'rssi', 'phase']:
            if key not in tag_data:
                print(f"  Warning: Missing '{key}' data in {source_info}")
                return results
            
            tag_data[key] = np.array(tag_data[key])
            
            if len(tag_data[key]) < 2:
                print(f"  Warning: Insufficient samples ({len(tag_data[key])}) in {source_info}")
                return results
        
        # Calculate consecutive differences for each measurement type
        try:
            # RSSI differences (absolute values, excluding zeros)
            rssi_diffs = np.abs(np.diff(tag_data['rssi']))
            rssi_diffs_nonzero = rssi_diffs[rssi_diffs > 0]
            
            if len(rssi_diffs_nonzero) > 0:
                min_rssi = np.min(rssi_diffs_nonzero)
                results['min_rssi_diff'] = min_rssi
                
                if min_rssi < self.min_rssi_diff:
                    self.min_rssi_diff = min_rssi
                    self.min_rssi_source = source_info
            
            # RSSI range analysis
            tag_min_rssi = np.min(tag_data['rssi'])
            tag_max_rssi = np.max(tag_data['rssi'])
            
            results['min_rssi_value'] = tag_min_rssi
            results['max_rssi_value'] = tag_max_rssi
            
            # Update global RSSI minimum
            if tag_min_rssi < self.min_rssi_value:
                self.min_rssi_value = tag_min_rssi
                self.min_rssi_value_source = source_info
            
            # Update global RSSI maximum
            if tag_max_rssi > self.max_rssi_value:
                self.max_rssi_value = tag_max_rssi
                self.max_rssi_value_source = source_info
            
            # Phase differences (absolute values, excluding zeros)
            # Handle phase wraparound (e.g., from 359° to 1°)
            phase_diffs = np.abs(np.diff(tag_data['phase']))
            
            # Check for phase wraparound and correct it
            wraparound_mask = phase_diffs > 180
            phase_diffs[wraparound_mask] = 360 - phase_diffs[wraparound_mask]
            
            phase_diffs_nonzero = phase_diffs[phase_diffs > 0]
            
            if len(phase_diffs_nonzero) > 0:
                min_phase = np.min(phase_diffs_nonzero)
                results['min_phase_diff'] = min_phase
                
                if min_phase < self.min_phase_diff:
                    self.min_phase_diff = min_phase
                    self.min_phase_source = source_info
            
            # Time differences (absolute values, excluding zeros)
            time_diffs = np.abs(np.diff(tag_data['timestamp']))
            time_diffs_nonzero = time_diffs[time_diffs > 0]
            
            if len(time_diffs_nonzero) > 0:
                min_time = np.min(time_diffs_nonzero)
                results['min_time_diff'] = min_time
                
                if min_time < self.min_time_diff:
                    self.min_time_diff = min_time
                    self.min_time_source = source_info
            
            # Update sample count
            self.total_samples_processed += len(tag_data['timestamp'])
            
        except Exception as e:
            print(f"  Error analyzing {source_info}: {e}")
            
        return results
    
    def process_file(self, csv_file: str) -> bool:
        """
        Process a single CSV file and analyze all tags within it.
        
        :param csv_file: Path to the CSV file
        :return: True if file was processed successfully, False otherwise
        """
        file_name = os.path.basename(csv_file)
        print(f"\nProcessing file: {file_name}")
        
        try:
            # Extract tag data using csv_data_loader
            tag_data_dict = extract_tag_data(csv_file)
            
            if not tag_data_dict:
                print(f"  No tag data found in {file_name}")
                return False
            
            print(f"  Found {len(tag_data_dict)} tags")
            
            # Process each tag in the file
            tags_processed_in_file = 0
            for tag_id, tag_data in tag_data_dict.items():
                source_info = f"{file_name}::{tag_id}"
                print(f"    Analyzing tag {tag_id} ({len(tag_data.get('timestamp', []))} samples)")
                
                # Analyze this tag's data
                tag_results = self.analyze_tag_data(tag_data, source_info)
                
                # Print results for this tag
                if tag_results['min_rssi_diff'] != float('inf'):
                    print(f"      RSSI min diff: {tag_results['min_rssi_diff']:.6f} dBm")
                if tag_results['min_phase_diff'] != float('inf'):
                    print(f"      Phase min diff: {tag_results['min_phase_diff']:.6f}°")
                if tag_results['min_time_diff'] != float('inf'):
                    print(f"      Time min diff: {tag_results['min_time_diff']:.6f} s")
                
                # Print RSSI range for this tag
                if tag_results['min_rssi_value'] != float('inf') and tag_results['max_rssi_value'] != float('-inf'):
                    rssi_range = tag_results['max_rssi_value'] - tag_results['min_rssi_value']
                    print(f"      RSSI range: {tag_results['min_rssi_value']:.3f} to {tag_results['max_rssi_value']:.3f} dBm (span: {rssi_range:.3f} dBm)")
                
                tags_processed_in_file += 1
                self.total_tags_processed += 1
            
            print(f"  ✓ Successfully processed {tags_processed_in_file} tags from {file_name}")
            return True
            
        except Exception as e:
            print(f"  ✗ Error processing {file_name}: {e}")
            self.files_with_errors.append(file_name)
            return False
    
    def process_directory(self, directory: str) -> None:
        """
        Process all CSV files in a directory.
        
        :param directory: Directory path to search for CSV files
        """
        print(f"\n{'='*60}")
        print(f"PROCESSING DIRECTORY: {directory}")
        print(f"{'='*60}")
        
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist!")
            return
        
        # Get all CSV files from directory
        csv_files = get_csv_files_from_directory(directory)
        
        if not csv_files:
            print(f"No CSV files found in {directory}")
            return
        
        print(f"Found {len(csv_files)} CSV files to process")
        
        # Process each file
        successful_files = 0
        for csv_file in csv_files:
            if self.process_file(csv_file):
                successful_files += 1
            self.total_files_processed += 1
        
        print(f"\nDirectory summary: {successful_files}/{len(csv_files)} files processed successfully")
    
    def process_multiple_directories(self, directories: List[str]) -> None:
        """
        Process multiple directories.
        
        :param directories: List of directory paths
        """
        print(f"{'='*80}")
        print(f"RFID DEVICE RESOLUTION ANALYSIS")
        print(f"{'='*80}")
        print(f"Analyzing {len(directories)} directories for measurement resolution...")
        
        for directory in directories:
            self.process_directory(directory)
    
    def print_final_results(self) -> None:
        """
        Print the final resolution analysis results.
        """
        print(f"\n{'='*80}")
        print(f"FINAL RESOLUTION ANALYSIS RESULTS")
        print(f"{'='*80}")
        
        # Processing summary
        print(f"Processing Summary:")
        print(f"  Total files processed: {self.total_files_processed}")
        print(f"  Total tags analyzed: {self.total_tags_processed}")
        print(f"  Total samples analyzed: {self.total_samples_processed:,}")
        
        if self.files_with_errors:
            print(f"  Files with errors: {len(self.files_with_errors)}")
            for error_file in self.files_with_errors[:5]:  # Show first 5
                print(f"    - {error_file}")
            if len(self.files_with_errors) > 5:
                print(f"    ... and {len(self.files_with_errors) - 5} more")
        
        print(f"\n{'='*50}")
        print(f"DEVICE MEASUREMENT RESOLUTION")
        print(f"{'='*50}")
        
        # RSSI Resolution
        if self.min_rssi_diff != float('inf'):
            print(f"🎯 RSSI Resolution: {self.min_rssi_diff:.6f} dBm")
            print(f"   Found in: {self.min_rssi_source}")
        else:
            print(f"❌ RSSI Resolution: No valid differences found")
        
        # Phase Resolution
        if self.min_phase_diff != float('inf'):
            print(f"🎯 Phase Resolution: {self.min_phase_diff:.6f}°")
            print(f"   Found in: {self.min_phase_source}")
        else:
            print(f"❌ Phase Resolution: No valid differences found")
        
        # Time Resolution
        if self.min_time_diff != float('inf'):
            print(f"🎯 Time Resolution: {self.min_time_diff:.6f} s")
            print(f"   Found in: {self.min_time_source}")
            
            # Convert to more readable units
            if self.min_time_diff < 1e-3:
                print(f"   Time Resolution: {self.min_time_diff * 1e6:.3f} μs")
            elif self.min_time_diff < 1:
                print(f"   Time Resolution: {self.min_time_diff * 1e3:.3f} ms")
        else:
            print(f"❌ Time Resolution: No valid differences found")
        
        # RSSI Range
        print(f"\n{'='*50}")
        print(f"RSSI MEASUREMENT RANGE")
        print(f"{'='*50}")
        
        if self.min_rssi_value != float('inf') and self.max_rssi_value != float('-inf'):
            rssi_total_range = self.max_rssi_value - self.min_rssi_value
            print(f"📊 RSSI Minimum Value: {self.min_rssi_value:.6f} dBm")
            print(f"   Found in: {self.min_rssi_value_source}")
            print(f"📊 RSSI Maximum Value: {self.max_rssi_value:.6f} dBm")
            print(f"   Found in: {self.max_rssi_value_source}")
            print(f"📊 RSSI Total Range: {rssi_total_range:.6f} dBm")
            
            # Calculate theoretical dynamic range in bits
            if self.min_rssi_diff != float('inf'):
                theoretical_levels = rssi_total_range / self.min_rssi_diff
                theoretical_bits = np.log2(theoretical_levels)
                print(f"📊 Theoretical RSSI levels: {theoretical_levels:.0f}")
                print(f"📊 Theoretical RSSI bits: {theoretical_bits:.1f} bits")
        else:
            print(f"❌ RSSI Range: No valid RSSI values found")
        
        print(f"\n{'='*50}")
        print(f"INTERPRETATION")
        print(f"{'='*50}")
        
        if self.min_rssi_diff != float('inf'):
            rssi_bits = np.log2(100 / self.min_rssi_diff)  # Assuming ~100 dBm range
            print(f"RSSI effective resolution: ~{rssi_bits:.1f} bits")
        
        if self.min_phase_diff != float('inf'):
            phase_bits = np.log2(360 / self.min_phase_diff)  # 360° range
            print(f"Phase effective resolution: ~{phase_bits:.1f} bits")
        
        if self.min_time_diff != float('inf'):
            frequency = 1 / self.min_time_diff
            print(f"Maximum sampling frequency: ~{frequency:.0f} Hz")
        
        # Additional RSSI analysis
        if (self.min_rssi_value != float('inf') and self.max_rssi_value != float('-inf') 
            and self.min_rssi_diff != float('inf')):
            actual_range = self.max_rssi_value - self.min_rssi_value
            actual_bits = np.log2(actual_range / self.min_rssi_diff)
            print(f"RSSI actual dynamic range: {actual_range:.3f} dBm")
            print(f"RSSI actual resolution: ~{actual_bits:.1f} bits")
    
    def get_results_dict(self) -> Dict[str, any]:
        """
        Get results as a dictionary for programmatic use.
        
        :return: Dictionary with resolution results
        """
        return {
            'rssi_resolution': float(self.min_rssi_diff) if self.min_rssi_diff != float('inf') else None,
            'phase_resolution': float(self.min_phase_diff) if self.min_phase_diff != float('inf') else None,
            'time_resolution': float(self.min_time_diff) if self.min_time_diff != float('inf') else None,
            'rssi_min_value': float(self.min_rssi_value) if self.min_rssi_value != float('inf') else None,
            'rssi_max_value': float(self.max_rssi_value) if self.max_rssi_value != float('-inf') else None,
            'rssi_range': float(self.max_rssi_value - self.min_rssi_value) if (self.min_rssi_value != float('inf') and self.max_rssi_value != float('-inf')) else None,
            'rssi_source': self.min_rssi_source,
            'phase_source': self.min_phase_source,
            'time_source': self.min_time_source,
            'rssi_min_value_source': self.min_rssi_value_source,
            'rssi_max_value_source': self.max_rssi_value_source,
            'processing_stats': {
                'total_files': int(self.total_files_processed),
                'total_tags': int(self.total_tags_processed),
                'total_samples': int(self.total_samples_processed),
                'error_files': self.files_with_errors
            }
        }

def analyze_device_resolution(directories: List[str] = None) -> Dict[str, any]:
    """
    Main function to analyze device resolution across multiple directories.
    
    :param directories: List of directories to analyze. If None, uses default directories.
    :return: Dictionary with resolution results
    """
    if directories is None:
        # Default directories based on the project structure
        directories = [
            'data/test/',
            'data/2025-07-15/',
            'data/2025-07-22/',
            'data/2025-07-25/',
            'data/2025-07-28/',
            'data/2025-07-31/',
            'data/2025-10-07/'
        ]
    
    # Filter directories that actually exist
    existing_directories = [d for d in directories if os.path.exists(d)]
    
    if not existing_directories:
        print("No valid directories found!")
        print("Checked directories:")
        for d in directories:
            print(f"  - {d} ({'EXISTS' if os.path.exists(d) else 'NOT FOUND'})")
        return {}
    
    print(f"Will analyze {len(existing_directories)} directories:")
    for d in existing_directories:
        print(f"  ✓ {d}")
    
    # Create analyzer and process directories
    analyzer = ResolutionAnalyzer()
    analyzer.process_multiple_directories(existing_directories)
    
    # Print results
    analyzer.print_final_results()
    
    return analyzer.get_results_dict()

if __name__ == "__main__":
    import sys
    
    print("=== RFID DEVICE RESOLUTION ANALYZER ===\n")
    
    # Check if custom directories were provided as command line arguments
    if len(sys.argv) > 1:
        directories = sys.argv[1:]
        print(f"Using custom directories: {directories}")
    else:
        directories = None
        print("Using default directories from project structure")
    
    # Run the analysis
    try:
        results = analyze_device_resolution(directories)
        
        if results:
            print(f"\n{'='*50}")
            print("ANALYSIS COMPLETED SUCCESSFULLY")
            print("="*50)
            
            # Save results to file with custom JSON encoder
            import json
            from datetime import datetime
            
            class NumpyEncoder(json.JSONEncoder):
                """Custom JSON encoder that converts numpy types to Python native types."""
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super(NumpyEncoder, self).default(obj)
            
            output_data = {
                'analysis_date': datetime.now().isoformat(),
                'device_resolution': results,
                'analysis_type': 'minimum_consecutive_differences'
            }
            
            os.makedirs('output_data', exist_ok=True)
            output_file = 'output_data/device_resolution_analysis.json'
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2, cls=NumpyEncoder)
            
            print(f"Results saved to: {output_file}")
        else:
            print("Analysis failed - no valid results obtained.")
            
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()