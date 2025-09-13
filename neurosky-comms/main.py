from NeuroPy import NeuroPy
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
from collections import deque, namedtuple
import statistics

# Define a comprehensive data structure for EEG readings
EEGReading = namedtuple('EEGReading', [
    'timestamp', 'attention', 'meditation', 'delta', 'theta', 
    'lowAlpha', 'highAlpha', 'lowBeta', 'highBeta', 
    'lowGamma', 'midGamma', 'highGamma'
])

# Unified data storage
MAX_SCORE_CACHE = 500
eeg_data = deque(maxlen=MAX_SCORE_CACHE)
start_time = None

# Real-time plotting globals
fig = None
axes = None  # Will hold multiple subplots
lines = {}   # Dictionary to store all plot lines
plot_initialized = False
plot_thread = None
plot_running = False

# Temporary storage for incomplete readings
current_reading = {
    'attention': None, 'meditation': None, 'delta': None, 'theta': None,
    'lowAlpha': None, 'highAlpha': None, 'lowBeta': None, 'highBeta': None,
    'lowGamma': None, 'midGamma': None, 'highGamma': None
}

def find_port():
    import serial.tools.list_ports
    import serial
    ports = serial.tools.list_ports.comports()
    
    print("Available ports:")
    for port in ports:
        print("  {}: {} (VID:PID = {}:{})".format(port.device, port.description, port.vid, port.pid))
    
    # Look for NeuroSky-specific identifiers first
    neurosky_ports = []
    for port in ports:
        port_desc = str(port.description).lower()
        # NeuroSky devices often appear as:
        # - "USB-SERIAL CH340" or similar USB-to-serial adapters
        # - Bluetooth SPP devices
        # - Devices with specific VID/PID combinations
        vid_pid_str = ""
        if port.vid and port.pid:
            vid_pid_str = "{:04x}:{:04x}".format(port.vid, port.pid)
        
        if (any(keyword in port_desc for keyword in ['ch340', 'cp210', 'ftdi', 'prolific', 'bluetooth']) or
            vid_pid_str in ['0403:6001', '10c4:ea60', '1a86:7523']):
            neurosky_ports.append(port.device)
            print("Potential NeuroSky port found: {} ({})".format(port.device, port.description))
    
    # Test NeuroSky-likely ports first, then all others
    test_ports = neurosky_ports + [p.device for p in ports if p.device not in neurosky_ports]
    
    for port_name in test_ports:
        print("Testing port: {}".format(port_name))
        try:
            # Test basic serial connectivity first
            ser = serial.Serial(port_name, 57600, timeout=0.5)
            ser.close()
            
            # Now test with NeuroPy - but with timeout protection
            import threading
            import time
            
            neuropy = NeuroPy(port_name, 57600)
            test_result = {'found': False, 'error': None}
            
            def test_neuropy():
                try:
                    data_count = [0]
                    def count_callback(value):
                        data_count[0] += 1
                    
                    neuropy.setCallBack("rawValue", count_callback)
                    neuropy.start()
                    
                    # Wait for data with timeout
                    start_time = time.time()
                    while time.time() - start_time < 1:  # 1 second timeout
                        if data_count[0] > 0:  # Just need any data
                            test_result['found'] = True
                            break
                        time.sleep(0.05)
                    
                    neuropy.stop()
                except Exception as e:
                    test_result['error'] = str(e)
            
            # Run test in separate thread with timeout
            test_thread = threading.Thread(target=test_neuropy)
            test_thread.daemon = True
            test_thread.start()
            test_thread.join(timeout=1.5)  # 1.5 second total timeout
            
            if test_result['found']:
                print("NeuroSky device confirmed on port: {}".format(port_name))
                return port_name
            elif test_result['error']:
                print("Port {} error: {}".format(port_name, test_result['error']))
            else:
                print("Port {}: No data".format(port_name))
                
        except Exception as e:
            print("Port {} failed: {}: {}".format(port_name, type(e).__name__, str(e)))
            continue
    
    return None

def init_plot():
    """Initialize the real-time plot with separate charts for each brainwave type"""
    global fig, axes, lines, plot_initialized
    
    if plot_initialized:
        return
    
    # Create a figure with multiple subplots (3 rows, 2 columns)
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('NeuroSky Real-time Brainwave Monitor', fontsize=16, fontweight='bold')
    
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
        {'ax': axes[2,1], 'title': 'Gamma Waves (30+ Hz)', 'ylabel': 'Power', 'ylim': (0, 15000), 
         'signals': [('lowGamma', 'magenta'), ('midGamma', 'darkmagenta')]}
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

def update_reading_data(key, value):
    """Helper function to update current reading and create complete reading when all data is available"""
    global start_time, current_reading
    
    if start_time is None:
        start_time = time.time()
    
    current_reading[key] = value
    
    # Check if we have all the essential data (attention and meditation are always present)
    # We'll create readings even if some brainwave data is missing
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
            midGamma=current_reading['midGamma'] or 0,
            highGamma=current_reading['highGamma'] or 0
        )
        eeg_data.append(reading)
        
        # Reset current reading
        for k in current_reading:
            current_reading[k] = None

def handle_attention_callback(level):
    update_reading_data('attention', level)

def handle_meditation_callback(level):
    update_reading_data('meditation', level)
        
def handle_delta_callback(level):
    update_reading_data('delta', level)

def handle_theta_callback(level):
    update_reading_data('theta', level)

def handle_low_alpha_callback(level):
    update_reading_data('lowAlpha', level)

def handle_high_alpha_callback(level):
    update_reading_data('highAlpha', level)

def handle_low_beta_callback(level):
    update_reading_data('lowBeta', level)

def handle_high_beta_callback(level):
    update_reading_data('highBeta', level)

def handle_low_gamma_callback(level):
    update_reading_data('lowGamma', level)

def handle_mid_gamma_callback(level):
    update_reading_data('midGamma', level)

def handle_high_gamma_callback(level):
    update_reading_data('highGamma', level)
    
def handle_raw_value(level):
    # Raw value callback - no longer triggers plotting directly
    pass

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
            'highBeta': [r.highBeta for r in readings],
            'lowGamma': [r.lowGamma for r in readings],
            'midGamma': [r.midGamma for r in readings],
            'highGamma': [r.highGamma for r in readings]
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
        brainwave_signals = ['delta', 'theta', 'lowAlpha', 'highAlpha', 'lowBeta', 'highBeta', 'lowGamma', 'midGamma']
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
        
        # Add current values as text on the first subplot
        if readings:
            latest = readings[-1]
            info_text = "Time: {:.1f}s | Readings: {}\n".format(current_time, len(readings))
            info_text += "Att: {:.1f} | Med: {:.1f}\n".format(latest.attention, latest.meditation)
            info_text += "Delta: {:.0f} | Theta: {:.0f}".format(latest.delta, latest.theta)
            
            # Remove any existing text annotations
            for txt in axes[0,0].texts:
                if hasattr(txt, '_info_text'):
                    txt.remove()
            
            # Add new text annotation
            text_obj = axes[0,0].text(0.02, 0.98, info_text, transform=axes[0,0].transAxes, 
                                    fontsize=9, verticalalignment='top',
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))
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

# Main execution
print("Finding NeuroSky device...")
port = find_port()

# If auto-detection fails, try common NeuroSky ports
if not port:
    print("Auto-detection failed. Trying common NeuroSky ports...")
    common_ports = ['COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8']
    for test_port in common_ports:
        try:
            import serial
            with serial.Serial(test_port, 57600, timeout=1) as ser:
                time.sleep(0.1)
            print("Found accessible port: {}".format(test_port))
            port = test_port
            break
        except:
            continue

if port:
    print("Using NeuroSky device on port: " + port)
    
    # Start the plotting thread
    print("Starting plot thread...")
    plot_thread = threading.Thread(target=plot_worker)
    plot_thread.daemon = True  # Dies when main thread dies
    plot_thread.start()
    
    # Give plot thread time to initialize
    time.sleep(1)
    
    neuropy = NeuroPy(port, 57600)
    
    # Set callbacks for attention and meditation
    neuropy.setCallBack("attention", handle_attention_callback)
    neuropy.setCallBack("meditation", handle_meditation_callback)
    
    # Set callbacks for all brainwave frequencies
    neuropy.setCallBack("delta", handle_delta_callback)
    neuropy.setCallBack("theta", handle_theta_callback)
    neuropy.setCallBack("lowAlpha", handle_low_alpha_callback)
    neuropy.setCallBack("highAlpha", handle_high_alpha_callback)
    neuropy.setCallBack("lowBeta", handle_low_beta_callback)
    neuropy.setCallBack("highBeta", handle_high_beta_callback)
    neuropy.setCallBack("lowGamma", handle_low_gamma_callback)
    neuropy.setCallBack("midGamma", handle_mid_gamma_callback)
    # Note: highGamma might not be available in all NeuroPy versions
    
    # Raw value callback (optional)
    neuropy.setCallBack("rawValue", handle_raw_value)
    
    print("Starting NeuroSky data collection...")
    neuropy.start()
   
    try:
        print("Real-time brain activity monitoring started. Press Ctrl+C to stop.")
        while True:
            time.sleep(0.1)  # Small sleep to prevent excessive CPU usage
    except KeyboardInterrupt:
        print("\nStopping monitoring...")
        plot_running = False  # Signal plot thread to stop
        neuropy.stop()
        
        # Wait for plot thread to finish
        if plot_thread.is_alive():
            plot_thread.join(timeout=2)
        
        plt.close('all')
        print("Monitoring stopped.")
else:
    print("No NeuroSky device found")
