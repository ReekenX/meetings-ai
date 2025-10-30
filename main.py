# /// script
# dependencies = [
#   "rumps",
# ]
# ///

import rumps
import subprocess
import signal
import threading
import os

class MeetingRecorderApp(rumps.App):
    def __init__(self):
        super().__init__("Meeting Recorder")
        self.process = None
        self.recording = False
        self.blink_state = False
        self.blink_timer = None

        # Set up icon paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.icon_active = os.path.join(script_dir, "icons", "active.png")
        self.icon_inactive = os.path.join(script_dir, "icons", "inactive.png")

        # Set initial icon
        self.icon = self.icon_inactive

        # Create menu items
        self.record_button = rumps.MenuItem("Start Recording", callback=self.toggle_recording)
        self.menu = [self.record_button]

    def blink(self):
        """Toggle between active and inactive icon"""
        if self.recording:
            self.icon = self.icon_active if self.blink_state else self.icon_inactive
            self.blink_state = not self.blink_state
            self.blink_timer = threading.Timer(1.0, self.blink)
            self.blink_timer.start()

    def toggle_recording(self, sender):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        response = rumps.Window(
            message="Enter meeting title:",
            title="Meeting Recorder",
            default_text="",
            ok="Start",
            cancel="Cancel",
            dimensions=(320, 24)
        ).run()

        if response.clicked:
            meeting_title = response.text
            script_dir = os.path.dirname(os.path.abspath(__file__))
            record_script = os.path.join(script_dir, "record.sh")
            self.process = subprocess.Popen(
                [record_script, meeting_title]
            )
            self.recording = True
            self.blink()  # Start blinking
            self.record_button.title = "Stop Recording"

    def stop_recording(self):
        self.recording = False
        if self.blink_timer:
            self.blink_timer.cancel()

        if self.process:
            try:
                # Try graceful termination first
                self.process.terminate()
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                # If graceful termination fails, force kill
                self.process.kill()
                try:
                    self.process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass  # Process is dead or will die soon
            finally:
                self.process = None

        self.icon = self.icon_inactive
        self.record_button.title = "Start Recording"

if __name__ == "__main__":
    MeetingRecorderApp().run()
