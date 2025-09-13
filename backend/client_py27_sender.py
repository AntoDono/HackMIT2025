#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python 2.7 compatible WebSocket client for sending EEG data
Requires: pip install websocket-client==0.48.0
"""

import json
import time
import random
import websocket

def send_eeg_data():
    """Send random EEG data to the WebSocket endpoint."""
    
    # WebSocket URL
    ws_url = "ws://127.0.0.1:8000/ws/brainwaves"
    
    print("Connecting to: {}".format(ws_url))
    
    def on_message(ws, message):
        """Handle incoming messages."""
        try:
            data = json.loads(message)
            if data.get('ok'):
                print("✓ ACK: count={}".format(data.get('count', 'unknown')))
            else:
                print("✗ ERROR: {}".format(data.get('error', 'unknown')))
        except Exception as e:
            print("✗ Parse error: {}".format(e))
    
    def on_error(ws, error):
        """Handle WebSocket errors."""
        print("✗ WebSocket error: {}".format(error))
    
    def on_close(ws):
        """Handle WebSocket close."""
        print("✗ WebSocket connection closed")
    
    def on_open(ws):
        """Handle WebSocket open."""
        print("✓ Connected to EEG WebSocket endpoint")
        print("Sending ~120 random EEG samples...")
        
        # Send random EEG data
        for i in range(120):
            # Generate random but realistic EEG values
            sample = {
                "attention": random.uniform(20, 80),
                "meditation": random.uniform(15, 85),
                "delta": random.uniform(1000, 50000),
                "theta": random.uniform(500, 25000),
                "lowAlpha": random.uniform(200, 15000),
                "highAlpha": random.uniform(100, 10000),
                "lowBeta": random.uniform(50, 8000),
                "highBeta": random.uniform(25, 5000)
            }
            
            # Send as JSON
            message = json.dumps(sample)
            ws.send(message)
            
            # Print progress every 10 samples
            if (i + 1) % 10 == 0:
                print("Sent {} samples...".format(i + 1))
            
            # Small delay between samples
            time.sleep(0.05)
        
        print("✓ Finished sending all samples")
        ws.close()
    
    # Create WebSocket connection
    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    
    # Run the WebSocket client
    ws.run_forever()

if __name__ == "__main__":
    send_eeg_data()