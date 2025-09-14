#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple NeuroSky EEG data reader that sends data via socket
No chart rendering - just data collection and transmission
"""

from NeuroPy import NeuroPy
import time
import sys
import os
from collections import namedtuple

# Add backend directory to path to import client
backend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend')
sys.path.insert(0, backend_path)
import client_py27_sender

# Define EEG data structure
EEGReading = namedtuple('EEGReading', [
    'timestamp', 'attention', 'meditation', 'delta', 'theta', 
    'lowAlpha', 'highAlpha', 'lowBeta', 'highBeta', 'lowGamma', 'midGamma'
])

# Global variables
start_time = None
current_reading = {
    'attention': None, 'meditation': None, 'delta': None, 'theta': None,
    'lowAlpha': None, 'highAlpha': None, 'lowBeta': None, 'highBeta': None,
    'lowGamma': None, 'midGamma': None
}

def find_neurosky_port():
    """Find NeuroSky device port"""
    import serial.tools.list_ports
    import serial
    
    ports = serial.tools.list_ports.comports()
    print("Available ports:")
    for port in ports:
        print("  {}: {}".format(port.device, port.description))
    
    # Try common NeuroSky ports first
    test_ports = ['COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8']
    
    for port_name in test_ports:
        try:
            print("Testing port: {}".format(port_name))
            ser = serial.Serial(port_name, 57600, timeout=0.5)
            ser.close()
            print("Found accessible port: {}".format(port_name))
            return port_name
        except Exception as e:
            print("Port {} failed: {}".format(port_name, str(e)))
            continue
    
    return None

def send_eeg_data(reading):
    """Send EEG data via socket"""
    data = {
        "timestamp": reading.timestamp,
        "attention": reading.attention,
        "meditation": reading.meditation,
        "delta": reading.delta,
        "theta": reading.theta,
        "lowAlpha": reading.lowAlpha,
        "highAlpha": reading.highAlpha,
        "lowBeta": reading.lowBeta,
        "highBeta": reading.highBeta,
        "lowGamma": reading.lowGamma,
        "midGamma": reading.midGamma
    }
    client_py27_sender.send_eeg_data(data)
    print("Sent: Att={:.1f} Med={:.1f} Delta={:.0f} Theta={:.0f} LowGamma={:.0f} MidGamma={:.0f}".format(
        reading.attention, reading.meditation, reading.delta, reading.theta, reading.lowGamma, reading.midGamma))

def update_reading(key, value):
    """Update current reading and send when complete"""
    global start_time, current_reading
    
    if start_time is None:
        start_time = time.time()
    
    current_reading[key] = value
    
    # Send data when we have attention and meditation (core values)
    if (current_reading['attention'] is not None and 
        current_reading['meditation'] is not None):
        
        timestamp = time.time() - start_time
        reading = EEGReading(
            timestamp=timestamp,
            attention=current_reading['attention'] or 0,
            meditation=current_reading['meditation'] or 0,
            delta=current_reading['delta'] or 0,
            theta=current_reading['theta'] or 0,
            lowAlpha=current_reading['lowAlpha'] or 0,
            highAlpha=current_reading['highAlpha'] or 0,
            lowBeta=current_reading['lowBeta'] or 0,
            highBeta=current_reading['highBeta'] or 0,
            lowGamma=current_reading['lowGamma'] or 0,
            midGamma=current_reading['midGamma'] or 0
        )
        
        send_eeg_data(reading)
        
        # Reset current reading
        for k in current_reading:
            current_reading[k] = None

# Callback functions
def handle_attention(level):
    update_reading('attention', level)

def handle_meditation(level):
    update_reading('meditation', level)

def handle_delta(level):
    update_reading('delta', level)

def handle_theta(level):
    update_reading('theta', level)

def handle_low_alpha(level):
    update_reading('lowAlpha', level)

def handle_high_alpha(level):
    update_reading('highAlpha', level)

def handle_low_beta(level):
    update_reading('lowBeta', level)

def handle_high_beta(level):
    update_reading('highBeta', level)

def handle_low_gamma(level):
    update_reading('lowGamma', level)

def handle_mid_gamma(level):
    update_reading('midGamma', level)

def main():
    """Main function"""
    print("Finding NeuroSky device...")
    port = find_neurosky_port()
    
    if not port:
        print("No NeuroSky device found")
        return
    
    print("Using NeuroSky device on port: " + port)
    
    # Connect to socket server
    print("Connecting to socket server...")
    if not client_py27_sender.connect_socket():
        print("Failed to connect to socket server")
        return
    
    # Initialize NeuroSky
    neuropy = NeuroPy(port, 57600)
    
    # Set callbacks
    neuropy.setCallBack("attention", handle_attention)
    neuropy.setCallBack("meditation", handle_meditation)
    neuropy.setCallBack("delta", handle_delta)
    neuropy.setCallBack("theta", handle_theta)
    neuropy.setCallBack("lowAlpha", handle_low_alpha)
    neuropy.setCallBack("highAlpha", handle_high_alpha)
    neuropy.setCallBack("lowBeta", handle_low_beta)
    neuropy.setCallBack("highBeta", handle_high_beta)
    neuropy.setCallBack("lowGamma", handle_low_gamma)
    neuropy.setCallBack("midGamma", handle_mid_gamma)
    
    print("Starting NeuroSky data collection...")
    neuropy.start()
    
    try:
        print("EEG data streaming started. Press Ctrl+C to stop.")
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
        neuropy.stop()
        client_py27_sender.close_connection()
        print("Stopped.")

if __name__ == "__main__":
    main()
