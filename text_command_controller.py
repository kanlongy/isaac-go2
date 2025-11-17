"""
Text Command Controller for Go2 Robot
Parses natural language commands and converts them to velocity commands.
"""

import re
import math
import torch
import threading
import queue
import time
from typing import List, Tuple, Optional


class Command:
    """Base class for robot commands"""
    def __init__(self, duration: float = 0.0):
        self.duration = duration
        self.elapsed_time = 0.0
        
    def get_velocity(self) -> Tuple[float, float, float]:
        """Returns (linear_x, linear_y, angular_z) velocities"""
        return (0.0, 0.0, 0.0)
    
    def is_complete(self) -> bool:
        """Check if command execution is complete"""
        return self.elapsed_time >= self.duration
    
    def update(self, dt: float):
        """Update command state"""
        self.elapsed_time += dt


class MoveForwardCommand(Command):
    def __init__(self, distance: float, velocity: float = 1.5):
        duration = abs(distance) / velocity
        super().__init__(duration)
        self.velocity = velocity if distance > 0 else -velocity
        
    def get_velocity(self) -> Tuple[float, float, float]:
        if not self.is_complete():
            return (self.velocity, 0.0, 0.0)
        return (0.0, 0.0, 0.0)


class MoveBackwardCommand(Command):
    def __init__(self, distance: float, velocity: float = 1.5):
        duration = abs(distance) / velocity
        super().__init__(duration)
        self.velocity = -velocity
        
    def get_velocity(self) -> Tuple[float, float, float]:
        if not self.is_complete():
            return (self.velocity, 0.0, 0.0)
        return (0.0, 0.0, 0.0)


class MoveLeftCommand(Command):
    def __init__(self, distance: float, velocity: float = 1.5):
        duration = abs(distance) / velocity
        super().__init__(duration)
        self.velocity = velocity
        
    def get_velocity(self) -> Tuple[float, float, float]:
        if not self.is_complete():
            return (0.0, self.velocity, 0.0)
        return (0.0, 0.0, 0.0)


class MoveRightCommand(Command):
    def __init__(self, distance: float, velocity: float = 1.5):
        duration = abs(distance) / velocity
        super().__init__(duration)
        self.velocity = -velocity
        
    def get_velocity(self) -> Tuple[float, float, float]:
        if not self.is_complete():
            return (0.0, self.velocity, 0.0)
        return (0.0, 0.0, 0.0)


class TurnLeftCommand(Command):
    def __init__(self, angle_deg: float, angular_velocity: float = 1.5):
        angle_rad = math.radians(abs(angle_deg))
        duration = angle_rad / angular_velocity
        super().__init__(duration)
        self.angular_velocity = angular_velocity
        
    def get_velocity(self) -> Tuple[float, float, float]:
        if not self.is_complete():
            return (0.0, 0.0, self.angular_velocity)
        return (0.0, 0.0, 0.0)


class TurnRightCommand(Command):
    def __init__(self, angle_deg: float, angular_velocity: float = 1.5):
        angle_rad = math.radians(abs(angle_deg))
        duration = angle_rad / angular_velocity
        super().__init__(duration)
        self.angular_velocity = -angular_velocity
        
    def get_velocity(self) -> Tuple[float, float, float]:
        if not self.is_complete():
            return (0.0, 0.0, self.angular_velocity)
        return (0.0, 0.0, 0.0)


class StopCommand(Command):
    def __init__(self, duration: float = 0.5):
        super().__init__(duration)
        
    def get_velocity(self) -> Tuple[float, float, float]:
        return (0.0, 0.0, 0.0)


class TextCommandParser:
    """Parses natural language text commands into robot commands"""
    
    @staticmethod
    def parse(text: str) -> List[Command]:
        """
        Parse text command into a list of Command objects.
        
        Examples:
        - "walk forward 1m"
        - "turn left 90 degree"
        - "move right 0.5m, walk backward 2m"
        - "forward 1m, left turn 90 deg, forward 1m"
        """
        commands = []
        
        # Split by comma or semicolon
        segments = re.split(r'[,;]', text.lower())
        
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
                
            cmd = TextCommandParser._parse_segment(segment)
            if cmd:
                commands.append(cmd)
        
        return commands
    
    @staticmethod
    def _parse_segment(segment: str) -> Optional[Command]:
        """Parse a single command segment"""
        
        # Extract numbers (distance or angle)
        number_match = re.search(r'(\d+\.?\d*)', segment)
        value = float(number_match.group(1)) if number_match else 1.0
        
        # Forward movement
        if re.search(r'\b(forward|ahead|front)\b', segment):
            return MoveForwardCommand(value)
        
        # Backward movement
        elif re.search(r'\b(backward|back|reverse)\b', segment):
            return MoveBackwardCommand(value)
        
        # Left movement (strafe)
        elif re.search(r'\b(left)\b', segment) and re.search(r'\b(move|strafe|slide)\b', segment):
            return MoveLeftCommand(value)
        
        # Right movement (strafe)
        elif re.search(r'\b(right)\b', segment) and re.search(r'\b(move|strafe|slide)\b', segment):
            return MoveRightCommand(value)
        
        # Turn left
        elif re.search(r'\b(turn|rotate)\b', segment) and re.search(r'\b(left|ccw)\b', segment):
            return TurnLeftCommand(value)
        
        elif re.search(r'\b(left)\b', segment) and re.search(r'\b(turn|rotate)\b', segment):
            return TurnLeftCommand(value)
        
        # Turn right
        elif re.search(r'\b(turn|rotate)\b', segment) and re.search(r'\b(right|cw)\b', segment):
            return TurnRightCommand(value)
        
        elif re.search(r'\b(right)\b', segment) and re.search(r'\b(turn|rotate)\b', segment):
            return TurnRightCommand(value)
        
        # Stop
        elif re.search(r'\b(stop|halt|wait)\b', segment):
            return StopCommand(value if value > 0 else 0.5)
        
        else:
            print(f"Warning: Could not parse command segment: '{segment}'")
            return None


class TextCommandController:
    """Controller that manages command queue and execution"""
    
    def __init__(self, num_envs: int = 1, env_idx: int = 0, enable_input: bool = True):
        self.num_envs = num_envs
        self.env_idx = env_idx  # Which environment to control
        self.command_queue = queue.Queue()
        self.current_command: Optional[Command] = None
        self.base_vel_cmd_input = torch.zeros((num_envs, 3), dtype=torch.float32)
        self.running = True
        self.parser = TextCommandParser()
        self.enable_input = enable_input
        
        # Start input thread only if enabled
        if self.enable_input:
            self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
            self.input_thread.start()
            
            print("\n" + "="*60)
            print("Text Command Controller Initialized!")
            print("="*60)
            print("Commands you can use:")
            print("  - walk forward 1m")
            print("  - walk backward 2m")
            print("  - turn left 90 degree")
            print("  - turn right 45 degree")
            print("  - move left 0.5m")
            print("  - move right 1m")
            print("  - stop")
            print("\nYou can chain commands with commas:")
            print("  Example: forward 1m, turn left 90, forward 1m")
            print("\nType 'quit' or 'exit' to stop the controller")
            print("="*60 + "\n")
    
    def _input_loop(self):
        """Background thread for reading user input"""
        while self.running:
            try:
                print()  # 换行，避免被状态输出覆盖
                user_input = input("\n> Enter command: ").strip()  # 添加新行和提示符
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Shutting down command controller...")
                    self.running = False
                    break
                
                if user_input:
                    commands = self.parser.parse(user_input)
                    if commands:
                        for cmd in commands:
                            self.command_queue.put(cmd)
                        print(f"✓ Queued {len(commands)} command(s)")
                    else:
                        print("✗ No valid commands parsed")
            except EOFError:
                break
            except Exception as e:
                print(f"Error reading input: {e}")
    
    def update(self, dt: float) -> torch.Tensor:
        """
        Update the controller state and return velocity commands.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            Velocity command tensor [num_envs, 3] where each row is [lin_x, lin_y, ang_z]
        """
        # If no current command, try to get one from queue
        if self.current_command is None or self.current_command.is_complete():
            try:
                self.current_command = self.command_queue.get_nowait()
                cmd_type = type(self.current_command).__name__
                duration = self.current_command.duration
                print(f"▶ Executing: {cmd_type} (duration: {duration:.2f}s)")
            except queue.Empty:
                self.current_command = None
        
        # Update current command
        if self.current_command is not None:
            self.current_command.update(dt)
            vel = self.current_command.get_velocity()
            self.base_vel_cmd_input[self.env_idx] = torch.tensor(vel, dtype=torch.float32)
            
            # Print progress
            if self.current_command.is_complete():
                print(f"✓ Command completed")
        else:
            # No command, stay still
            self.base_vel_cmd_input[self.env_idx].zero_()
        
        return self.base_vel_cmd_input.clone()
    
    def add_command_text(self, text: str):
        """Manually add a command from text (for API use)"""
        commands = self.parser.parse(text)
        for cmd in commands:
            self.command_queue.put(cmd)
        return len(commands)
    
    def clear_queue(self):
        """Clear all pending commands"""
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
            except queue.Empty:
                break
        self.current_command = None
        print("✓ Command queue cleared")
    
    def shutdown(self):
        """Shutdown the controller"""
        self.running = False
        self.clear_queue()

