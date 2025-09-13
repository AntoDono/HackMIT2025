#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NeuroSky EEG data reader with real-time plotting and socket transmission
Combines data visualization with socket communication to backend
"""

from NeuroPy import NeuroPy
import time
import sys
import os
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import statistics

# Add backend directory to path to import client
backend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend')
sys.path.insert(0, backend_path)
import client_py27_sender

# Define EEG data structure
EEGReading = namedtuple('EEGReading', [
    'timestamp', 'attention', 'meditation', 'delta', 'theta', 
    'lowAlpha', 'highAlpha', 'lowBeta', 'highBeta'
])

# Global variables
MAX_SCORE_CACHE = 500
eeg_data = deque(maxlen=MAX_SCORE_CACHE)
start_time = None
current_reading = {
    'attention': None, 'meditation': None, 'delta': None, 'theta': None,
    'lowAlpha': None, 'highAlpha': None, 'lowBeta': None, 'highBeta': None
}

# Real-time plotting globals
fig = None
axes = None  # Will hold multiple subplots
lines = {}   # Dictionary to store all plot lines
plot_initialized = False
plot_thread = None
plot_running = False

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
        "highBeta": reading.highBeta
    }
    client_py27_sender.send_eeg_data(data)
    print("Sent: Att={:.1f} Med={:.1f} Delta={:.0f} Theta={:.0f}".format(
        reading.attention, reading.meditation, reading.delta, reading.theta))

def init_plot():
    """Initialize the real-time plot with separate charts for each brainwave type"""
    global fig, axes, lines, plot_initialized
    
    if plot_initialized:
        return
    
    # Create a figure with multiple subplots (3 rows, 2 columns)
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('NeuroSky Real-time Brainwave Monitor (with Socket)', fontsize=16, fontweight='bold')
    
    # Define the chart configurations with appropriate ranges
    chart_configs = [
        # Row 1
        {'ax': axes[0,0], 'title': 'Attention & Meditation', 'ylabel': 'Level (0-100)', 'ylim': (0, 100), 
         'signals': [('attention', 'red'), ('meditation', 'blue')]},
        {'ax': axes[0,1], 'title': 'Delta Waves (0.5-4 Hz)', 'ylabel': 'Power', 'ylim': (0, 100000), 
         'signals': [('delta', 'purple')]},
        
        # Row 2  
        {'ax': axes[1,0], 'title': 'Theta Waves (4-8 Hz)', 'ylabel': 'Power', 'ylim': (0, 50000), 
         'signals': [('theta', 'orange')]},
        {'ax': axes[1,1], 'title': 'Alpha Waves (8-13 Hz)', 'ylabel': 'Power', 'ylim': (0, 30000), 
         'signals': [('lowAlpha', 'green'), ('highAlpha', 'darkgreen')]},
        
        # Row 3
        {'ax': axes[2,0], 'title': 'Beta Waves (13-30 Hz)', 'ylabel': 'Power', 'ylim': (0, 20000), 
         'signals': [('lowBeta', 'cyan'), ('highBeta', 'darkcyan')]},
        {'ax': axes[2,1], 'title': 'Status', 'ylabel': 'Connection Status', 'ylim': (0, 1), 
         'signals': []}
    ]
    
    # Initialize each chart
    for config in chart_configs:
        ax = config['ax']
        ax.set_title(config['title'], fontsize=12, fontweight='bold')
        ax.set_ylabel(config['ylabel'], fontsize=10)
        ax.set_xlabel('Time (seconds)', fontsize=10)
        ax.set_ylim(config['ylim'])
        ax.grid(True, alpha=0.3)
        
        # Create lines for each signal in this chart
        for signal_name, color in config['signals']:
            line, = ax.plot([], [], color=color, linewidth=2, label=signal_name.title(), alpha=0.8)
            lines[signal_name] = line
        
        # Add legend if multiple signals
        if len(config['signals']) > 1:
            ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plot_initialized = True
    
    # Show plot in non-blocking mode
    plt.ion()
    plt.show()

def render_scores():
    """Render real-time scores on all plots - runs in separate thread"""
    global fig, axes, lines, start_time, plot_running
    
    # Skip if no data yet
    if not eeg_data:
        return
    
    try:
        current_time = time.time() - start_time if start_time else 0
        
        # Extract data from our unified structure
        readings = list(eeg_data)
        if not readings:
            return
            
        timestamps = [r.timestamp for r in readings]
        
        # Update all plot lines with their respective data
        signal_data = {
            'attention': [r.attention for r in readings],
            'meditation': [r.meditation for r in readings],
            'delta': [r.delta for r in readings],
            'theta': [r.theta for r in readings],
            'lowAlpha': [r.lowAlpha for r in readings],
            'highAlpha': [r.highAlpha for r in readings],
            'lowBeta': [r.lowBeta for r in readings],
            'highBeta': [r.highBeta for r in readings]
        }
        
        # Update each line with its data
        for signal_name, line in lines.items():
            if signal_name in signal_data:
                line.set_data(timestamps, signal_data[signal_name])
        
        # Auto-scale x-axis for all subplots (last 30 seconds)
        if current_time > 0:
            x_min = max(0, current_time - 30)
            x_max = current_time + 2
            for ax_row in axes:
                for ax in ax_row:
                    ax.set_xlim(x_min, x_max)
        
        # Auto-scale y-axis for brainwave power charts (not attention/meditation)
        brainwave_signals = ['delta', 'theta', 'lowAlpha', 'highAlpha', 'lowBeta', 'highBeta']
        for signal_name in brainwave_signals:
            if signal_name in signal_data and signal_data[signal_name]:
                max_val = max(signal_data[signal_name])
                if max_val > 0:
                    # Find which axis this signal belongs to and update its ylim
                    line = lines[signal_name]
                    ax = line.axes
                    current_ylim = ax.get_ylim()[1]
                    if max_val > current_ylim * 0.8:  # Update if near or exceeding current limit
                        ax.set_ylim(0, max_val * 1.2)  # Add 20% padding
        
        # Add current values as text on the status subplot
        if readings:
            latest = readings[-1]
            info_text = "Time: {:.1f}s | Readings: {}\n".format(current_time, len(readings))
            info_text += "Att: {:.1f} | Med: {:.1f}\n".format(latest.attention, latest.meditation)
            info_text += "Delta: {:.0f} | Theta: {:.0f}\n".format(latest.delta, latest.theta)
            info_text += "Socket: Active"
            
            # Remove any existing text annotations
            for txt in axes[2,1].texts:
                if hasattr(txt, '_info_text'):
                    txt.remove()
            
            # Add new text annotation to status subplot
            text_obj = axes[2,1].text(0.05, 0.95, info_text, transform=axes[2,1].transAxes, 
                                    fontsize=10, verticalalignment='top',
                                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
            text_obj._info_text = True
        
        # Refresh the plot
        plt.draw()
        plt.pause(0.01)  # Small pause to allow GUI updates
        
    except Exception as e:
        print("Plot update error: " + str(e))

def plot_worker():
    """Worker function that runs in separate thread for plotting"""
    global plot_running, plot_initialized
    
    # Initialize plot in this thread
    init_plot()
    plot_running = True
    
    print("Plot thread started...")
    
    try:
        while plot_running:
            if eeg_data:  # Only update if we have data
                render_scores()
            time.sleep(0.1)  # Update plot every 100ms
    except Exception as e:
        print("Plot thread error: " + str(e))
    finally:
        plot_running = False
        print("Plot thread stopped.")

def update_reading(key, value):
    """Update current reading, store data, and send when complete"""
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
            highBeta=current_reading['highBeta'] or 0
        )
        
        # Store data for plotting
        eeg_data.append(reading)
        
        # Send data via socket
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
    
    # Start the plotting thread
    print("Starting plot thread...")
    global plot_thread, plot_running
    plot_thread = threading.Thread(target=plot_worker)
    plot_thread.daemon = True  # Dies when main thread dies
    plot_thread.start()
    
    # Give plot thread time to initialize
    time.sleep(1)
    
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
    
    print("Starting NeuroSky data collection...")
    neuropy.start()
    
    try:
        print("EEG data streaming with real-time plotting started. Press Ctrl+C to stop.")
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
        plot_running = False  # Signal plot thread to stop
        neuropy.stop()
        client_py27_sender.close_connection()
        
        # Wait for plot thread to finish
        if plot_thread.is_alive():
            plot_thread.join(timeout=2)
        
        plt.close('all')
        print("Stopped.")

if __name__ == "__main__":
    main()
