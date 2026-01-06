#!/usr/bin/env python3
"""
Real-time data receiver for ESP32 gas sensor
Receives streaming data over serial and can save to CSV or plot in real-time
"""

import serial
import csv
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import argparse
import sys

class RealTimeReceiver:
    def __init__(self, port, baudrate=230400, max_points=1000):
        self.port = port
        self.baudrate = baudrate
        self.max_points = max_points
        
        # Data storage
        self.timestamps = deque(maxlen=max_points)
        self.settings = deque(maxlen=max_points)
        self.channel_0 = deque(maxlen=max_points)
        
        # Serial connection
        self.ser = None
        self.streaming = False
        
        # CSV writer
        self.csv_writer = None
        self.csv_file = None
        
    def connect(self):
        """Connect to ESP32"""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"Connected to {self.port} at {self.baudrate} baud")
            return True
        except serial.SerialException as e:
            print(f"Failed to connect: {e}")
            return False
    
    def start_csv_logging(self, filename):
        """Start logging to CSV file"""
        self.csv_file = open(filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Setting', 'Timestamp', 'Channel_0'])
        print(f"CSV logging started: {filename}")
    
    def stop_csv_logging(self):
        """Stop CSV logging"""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
            print("CSV logging stopped")
    
    def send_command(self, command):
        """Send command to ESP32"""
        if self.ser:
            self.ser.write(f"{command}\n".encode())
            print(f"Sent command: {command}")
    
    def read_data(self):
        """Read and parse incoming data"""
        if not self.ser:
            return None
            
        try:
            line = self.ser.readline().decode().strip()
            
            if line.startswith("DATA,"):
                # Parse: DATA,setting,timestamp,channel_0
                parts = line.split(',')
                if len(parts) == 4:
                    setting = int(parts[1])
                    timestamp = int(parts[2])
                    channel_0 = int(parts[3])
                    
                    # Store data
                    self.timestamps.append(timestamp)
                    self.settings.append(setting)
                    self.channel_0.append(channel_0)
                    
                    # Log to CSV if enabled
                    if self.csv_writer:
                        self.csv_writer.writerow([setting, timestamp, channel_0])
                        self.csv_file.flush()  # Ensure data is written
                    
                    # Debug print for first few data points
                    if len(self.timestamps) <= 5:
                        print(f"DEBUG: Received data point {len(self.timestamps)}: {setting}, {timestamp}, {channel_0}")
                    
                    return (setting, timestamp, channel_0)
            
            elif line == "STREAM_START":
                self.streaming = True
                print("✅ Stream started")
            
            elif line == "STREAM_STOP":
                self.streaming = False
                print("❌ Stream stopped")
            
            elif line and not line.startswith("DATA,"):
                # Print other messages from ESP32
                print(f"ESP32: {line}")
                
        except Exception as e:
            print(f"Error reading data: {e}")
        
        return None
    
    def plot_realtime(self):
        """Create real-time plot"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        def animate(frame):
            # Always read data to keep the connection alive
            self.read_data()
            
            if len(self.timestamps) > 0:
                times = list(self.timestamps)
                settings = list(self.settings)
                values = list(self.channel_0)
                
                # Convert timestamps to relative time (seconds)
                if times:
                    start_time = times[0]
                    rel_times = [(t - start_time) / 1000.0 for t in times]
                    
                    ax1.clear()
                    ax2.clear()
                    
                    # Plot sensor values
                    ax1.plot(rel_times, values, 'b-', alpha=0.7, marker='o', markersize=2)
                    ax1.set_ylabel('ADC Value')
                    ax1.set_title('Real-time Gas Sensor Data')
                    ax1.grid(True)
                    
                    # Plot heater settings
                    ax2.plot(rel_times, settings, 'r-', alpha=0.7, marker='s', markersize=3)
                    ax2.set_ylabel('Heater Setting')
                    ax2.set_xlabel('Time (seconds)')
                    ax2.grid(True)
                    
                    # Show streaming status and data count
                    status = "STREAMING" if self.streaming else "NOT STREAMING"
                    fig.suptitle(f'ESP32 Gas Sensor - {status} ({len(values)} points)')
            else:
                # Show waiting message when no data
                ax1.clear()
                ax2.clear()
                ax1.text(0.5, 0.5, 'Waiting for data...', ha='center', va='center', transform=ax1.transAxes)
                ax2.text(0.5, 0.5, 'Send "stream" command to ESP32', ha='center', va='center', transform=ax2.transAxes)
                status = "STREAMING" if self.streaming else "NOT STREAMING"
                fig.suptitle(f'ESP32 Gas Sensor - {status} (0 points)')
        
        ani = animation.FuncAnimation(fig, animate, interval=100, blit=False, cache_frame_data=False)
        plt.tight_layout()
        plt.show()
    
    def run_console(self):
        """Run in console mode"""
        print("Console mode - press Ctrl+C to exit")
        print("Available commands:")
        print("  s - toggle streaming")
        print("  c <filename> - start CSV logging")
        print("  x - stop CSV logging")
        print("  start - start acquisition")
        print("  stop - stop acquisition")
        print("  status - show ESP32 status")
        print("  q - quit")
        print("")
        
        import threading
        import queue
        
        # Queue for user input
        input_queue = queue.Queue()
        
        def input_thread():
            while True:
                try:
                    cmd = input()
                    input_queue.put(cmd)
                except EOFError:
                    break
        
        # Start input thread
        thread = threading.Thread(target=input_thread, daemon=True)
        thread.start()
        
        try:
            while True:
                # Check for user input (non-blocking)
                try:
                    user_input = input_queue.get_nowait().strip()
                    
                    if user_input == 's':
                        self.send_command('stream')
                    elif user_input.startswith('c '):
                        filename = user_input[2:]
                        self.start_csv_logging(filename)
                    elif user_input == 'x':
                        self.stop_csv_logging()
                    elif user_input == 'q':
                        break
                    elif user_input in ['start', 'stop', 'status', 'format', 'read', 'files']:
                        self.send_command(user_input)
                    elif user_input:
                        print(f"Unknown command: {user_input}")
                        
                except queue.Empty:
                    pass
                
                # Read data
                data = self.read_data()
                if data:
                    setting, timestamp, channel_0 = data
                    print(f"Setting: {setting:3d}, Time: {timestamp:8d}, ADC: {channel_0:5d}")
                
                time.sleep(0.01)  # Small delay
                
        except KeyboardInterrupt:
            print("\nExiting...")
        
        self.stop_csv_logging()

def main():
    parser = argparse.ArgumentParser(description='ESP32 Gas Sensor Real-time Receiver')
    parser.add_argument('port', help='Serial port (e.g., COM3 or /dev/ttyUSB0)')
    parser.add_argument('--baudrate', type=int, default=230400, help='Baud rate')
    parser.add_argument('--plot', action='store_true', help='Show real-time plot')
    parser.add_argument('--csv', help='Save data to CSV file')
    parser.add_argument('--max-points', type=int, default=1000, help='Maximum points to store')
    parser.add_argument('--auto-stream', action='store_true', help='Automatically start streaming')
    
    args = parser.parse_args()
    
    receiver = RealTimeReceiver(args.port, args.baudrate, args.max_points)
    
    if not receiver.connect():
        sys.exit(1)
    
    if args.csv:
        receiver.start_csv_logging(args.csv)
    
    # Auto-start streaming if requested
    if args.auto_stream:
        print("Auto-starting stream...")
        time.sleep(1)  # Give ESP32 time to initialize
        receiver.send_command('stream')
    else:
        print("Connected! Use 's' to start streaming, or 'q' to quit.")
    
    if args.plot:
        # For plotting mode, auto-start streaming
        if not args.auto_stream:
            print("Auto-starting stream for plot mode...")
            time.sleep(1)
            receiver.send_command('stream')
        receiver.plot_realtime()
    else:
        receiver.run_console()

if __name__ == "__main__":
    main() 