#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python 2.7 compatible socket client for sending EEG data
Simple TCP socket communication
"""

import json
import socket
import time
import threading
import Queue

# Global socket connection
socket_connection = None
server_host = "10.189.119.65"
server_port = 8001  # Different port for plain socket

# Threading globals
send_queue = Queue.Queue()
sender_thread = None
sender_running = False

def connect_socket():
    """Initialize global socket connection"""
    global socket_connection
    
    if socket_connection is not None:
        return True
    
    try:
        print("Connecting to: {}:{}".format(server_host, server_port))
        
        # Create simple TCP socket connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((server_host, server_port))
        
        socket_connection = sock
        print("Connected to EEG socket endpoint")
        return True
    except Exception as e:
        print("Failed to connect to socket: {}".format(e))
        socket_connection = None
        return False

def send_data(data):
    """Send data through socket - simple JSON with newline delimiter"""
    message = json.dumps(data) + '\n'
    return message.encode('utf-8')

def sender_worker():
    """Worker thread that handles socket sending"""
    global socket_connection, sender_running
    
    while sender_running:
        try:
            # Wait for data to send (with timeout to check sender_running)
            try:
                sample_data = send_queue.get(timeout=0.1)
            except Queue.Empty:
                continue
            
            # Ensure connection exists
            if socket_connection is None:
                if not connect_socket():
                    continue
            
            # Send the data
            try:
                message = send_data(sample_data)
                socket_connection.send(message)
            except Exception as e:
                print("Socket send error: {}".format(e))
                # Try to reconnect on next iteration
                socket_connection = None
            
            # Mark task as done
            send_queue.task_done()
            
        except Exception as e:
            print("Sender thread error: {}".format(e))

def start_sender_thread():
    """Start the background sender thread"""
    global sender_thread, sender_running
    
    if sender_thread is not None and sender_thread.is_alive():
        return True
    
    sender_running = True
    sender_thread = threading.Thread(target=sender_worker)
    sender_thread.daemon = True
    sender_thread.start()
    print("Socket sender thread started")
    return True

def send_eeg_data(sample_data):
    """Queue EEG data for non-blocking socket sending"""
    global send_queue, sender_thread
    
    # Start sender thread if not running
    if sender_thread is None or not sender_thread.is_alive():
        start_sender_thread()
    
    try:
        # Add data to queue (non-blocking)
        send_queue.put_nowait(sample_data)
        return True
    except Queue.Full:
        print("Socket send queue full, dropping data")
        return False
    except Exception as e:
        print("Queue error: {}".format(e))
        return False

def close_connection():
    """Close the global socket connection and stop sender thread"""
    global socket_connection, sender_running, sender_thread
    
    # Stop sender thread
    sender_running = False
    if sender_thread and sender_thread.is_alive():
        sender_thread.join(timeout=1)
    
    # Close socket connection
    if socket_connection:
        try:
            socket_connection.close()
        except:
            pass
        socket_connection = None
    
    print("Socket connection and sender thread closed")

if __name__ == "__main__":
    # Test connection
    if connect_socket():
        print("Socket client ready for EEG data...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            close_connection()