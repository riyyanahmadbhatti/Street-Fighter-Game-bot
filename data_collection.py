import socket
import json
import sys
import pandas as pd
import time
import os
from game_state import GameState
from command import Command
from buttons import Buttons
from bot import Bot

def connect(port):
    """
    Establish connection with the game emulator
    
    Args:
        port: Port number to connect to
        
    Returns:
        Connected socket object
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("127.0.0.1", port))
    server_socket.listen(5)
    print("Waiting for game to connect...")
    (client_socket, _) = server_socket.accept()
    print("Connected to game!")
    return client_socket

def receive(client_socket):
    """
    Receive game state from the emulator
    
    Args:
        client_socket: Connected socket object
        
    Returns:
        GameState object representing current game state
    """
    try:
        pay_load = client_socket.recv(4096)
        if not pay_load:
            raise ConnectionError("No data received from game")
        input_dict = json.loads(pay_load.decode())
        game_state = GameState(input_dict)
        return game_state
    except json.JSONDecodeError:
        print("Error: Could not decode game state data")
        return None
    except Exception as e:
        print(f"Error receiving data: {e}")
        return None

def send(client_socket, command):
    """
    Send command to the game emulator
    
    Args:
        client_socket: Connected socket object
        command: Command object containing button presses
    """
    try:
        command_dict = command.object_to_dict()
        pay_load = json.dumps(command_dict).encode()
        client_socket.sendall(pay_load)
    except Exception as e:
        print(f"Error sending command: {e}")

def extract_features(game_state, player_num):
    """
    Extract features from the game state for dataset collection
    
    Args:
        game_state: Current game state
        player_num: Player number ('1' or '2')
        
    Returns:
        Dictionary of features
    """
    if player_num == "1":
        my_player = game_state.player1
        opponent = game_state.player2
    else:
        my_player = game_state.player2
        opponent = game_state.player1
    
    distance = abs(my_player.x_coord - opponent.x_coord)
    direction = opponent.x_coord - my_player.x_coord
    
    features = {
        'my_health': my_player.health,
        'opponent_health': opponent.health,
        'health_diff': my_player.health - opponent.health,
        'distance': distance,
        'direction': direction,
        'my_x': my_player.x_coord,
        'my_y': my_player.y_coord,
        'opponent_x': opponent.x_coord,
        'opponent_y': opponent.y_coord,
        'my_jumping': int(my_player.is_jumping),
        'my_crouching': int(my_player.is_crouching),
        'opponent_jumping': int(opponent.is_jumping),
        'opponent_crouching': int(opponent.is_crouching),
        'my_in_move': int(my_player.is_player_in_move),
        'opponent_in_move': int(opponent.is_player_in_move),
        'my_move_id': my_player.move_id,
        'opponent_move_id': opponent.move_id,
        'timer': game_state.timer,
        'player_id': my_player.player_id,  # Character ID
        'opponent_id': opponent.player_id,  # Opponent Character ID
    }
    
    # Add my button presses
    for button, value in my_player.player_buttons.object_to_dict().items():
        features[f'my_{button}'] = int(value)
    
    # Add opponent button presses
    for button, value in opponent.player_buttons.object_to_dict().items():
        features[f'opponent_{button}'] = int(value)
        
    return features

def save_dataset(dataset, filename):
    """
    Append the dataset to a CSV file
    
    Args:
        dataset: List of feature dictionaries
        filename: Name of the file to save to
    """
    if len(dataset) > 0:
        df = pd.DataFrame(dataset)
        
        # Check if file exists to determine if we need to write headers
        file_exists = os.path.isfile(filename)
        
        if file_exists:
            # Append without writing the header
            df.to_csv(filename, mode='a', header=False, index=False)
            print(f"Appended {len(dataset)} samples to {filename}")
        else:
            # Create new file with header
            df.to_csv(filename, index=False)
            print(f"Created new dataset file {filename} with {len(dataset)} samples")

def main():
    """Main function for data collection"""
    if len(sys.argv) < 2:
        print("Usage: python data_collector.py [1|2]")
        return
        
    player_num = sys.argv[1]
    
    if player_num not in ['1', '2']:
        print("Player number must be 1 or 2")
        return
        
    # Connect to game
    port = 9999 if player_num == '1' else 10000
    
    # Initialize bot for actions
    bot = Bot()
    
    # Initialize empty command
    my_command = Command()
    
    # Dataset collection
    dataset = []
    dataset_file = f"sf2_dataset_p{player_num}.csv"
    
    # Game loop
    round_count = 0
    was_round_over = False  # Flag to track round state changes
    
    print(f"Collecting data for player {player_num}")
    print("Press Ctrl+C to stop data collection")
    
    try:
        # Connect to game
        client_socket = connect(port)
        
        while True:
            try:
                # Get current game state
                current_game_state = receive(client_socket)
                
                if current_game_state is None:
                    print("Failed to receive game state, retrying...")
                    time.sleep(0.1)
                    # Keep sending empty commands to prevent connection timeout
                    send(client_socket, my_command)
                    continue
                
                # Check for round state transitions
                if current_game_state.is_round_over and not was_round_over:
                    # Round just ended
                    round_count += 1
                    print(f"Round {round_count} completed. Collected {len(dataset)} samples.")
                    
                    # Save data collected so far
                    save_dataset(dataset, dataset_file)
                    
                    # Set flag to true
                    was_round_over = True
                    
                    # Keep connection alive with empty commands
                    send(client_socket, my_command)
                    
                elif not current_game_state.is_round_over and was_round_over:
                    # Round state changed from over to not over (new round starting)
                    print("New round starting...")
                    was_round_over = False
                    
                    # Send empty command to keep connection alive
                    send(client_socket, my_command)
                    
                # Only collect data if round has started and is not over
                elif current_game_state.has_round_started and not current_game_state.is_round_over:
                    # Get action from bot
                    bot_command = bot.fight(current_game_state, player_num)
                    
                    # Extract features from current state
                    features = extract_features(current_game_state, player_num)
                    
                    # Add action label
                    features['action'] = bot.current_action if hasattr(bot, 'current_action') else 'idle'
                    
                    # Add to dataset
                    dataset.append(features)
                    
                    # Send command to game
                    send(client_socket, bot_command)
                    
                    # Periodically save dataset
                    if len(dataset) % 100 == 0:
                        save_dataset(dataset, dataset_file)
                        # Clear dataset after saving to avoid duplicate appends
                        dataset = []
                else:
                    # If round hasn't started or is over, send empty command to keep connection alive
                    send(client_socket, my_command)
                
                # Brief delay to prevent CPU hogging
                time.sleep(0.01)
                
            except ConnectionError as e:
                print(f"Connection error: {e}")
                print("Attempting to reconnect...")
                try:
                    client_socket.close()
                    client_socket = connect(port)
                except:
                    print("Failed to reconnect. Exiting.")
                    break
            except Exception as e:
                print(f"Error in game loop: {e}")
                # Send empty command to keep connection alive
                send(client_socket, my_command)
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\nStopping data collection")
        
    finally:
        # Save final dataset
        save_dataset(dataset, dataset_file)
        
        # Close socket
        try:
            client_socket.close()
        except:
            pass

if __name__ == "__main__":
    main()