from command import Command
import numpy as np
from buttons import Buttons
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from collections import deque

class Bot:
    """
    AI Bot for Street Fighter II game using machine learning approach
    """
    def __init__(self):
        #Initialize bot variables
        self.my_command = Command()
        self.buttn = Buttons()
        
        #Dataset collection
        self.collect_data = True  #Set to True to collect training data
        self.dataset = []
        self.dataset_file = "sf2_dataset.csv"
        
        #For tracking previous states (for temporal features)
        self.state_history = deque(maxlen=5)
        
        #Load the ML model if it exists
        self.model_file = "sf2_model.joblib"
        self.model = self.load_model() if os.path.exists(self.model_file) else None
        
        #Move execution tracking
        self.exe_code = 0
        self.remaining_code = []
        self.fire_code = []
        
        #Define action sets (combinations of buttons with meaningful moves)
        self.action_sets = {
            'idle': [],
            'move_right': [">", "!>"],
            'move_left': ["<", "!<"],
            'crouch': ["v", "!v"],
            'jump': ["^", "!^"],
            'punch_y': ["Y", "!Y"],
            'punch_x': ["X", "!X"],
            'kick_b': ["B", "!B"],
            'kick_a': ["A", "!A"],
            'special1': ["<", "!<", "v+<", "!v+!<", "v", "!v", "v+>", "!v+!>", ">+Y", "!>+!Y"],  #Hadouken-like
            'special2': ["v+R", "v+R", "v+R", "!v+!R"],  #Defensive special
            'jump_kick': [">+^+B", ">+^+B", "!>+!^+!B"],  #Jump kick
        }
        
        #Current action being executed
        self.current_action = 'idle'

    def load_model(self):
        """Load the trained machine learning model"""
        try:
            return joblib.load(self.model_file)
        except:
            print("Model file not found or error loading model.")
            return None
    
    def extract_features(self, game_state, player_num):
        """
        Extract features from the game state for ML model
        
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
        
        #Calculate distance between players
        distance = abs(my_player.x_coord - opponent.x_coord)
        
        #Direction to opponent (positive means opponent is to the right)
        direction = opponent.x_coord - my_player.x_coord
        
        #Extract features
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
        }
        
        #Add opponent button presses as features
        for button, value in opponent.player_buttons.object_to_dict().items():
            features[f'opponent_{button}'] = int(value)
        
        return features
    
    def save_state_to_dataset(self, game_state, player_num, action):
        """
        Save the current state and action to dataset
        
        Args:
            game_state: Current game state
            player_num: Player number ('1' or '2')
            action: Current action taken
        """
        if not self.collect_data:
            return
            
        features = self.extract_features(game_state, player_num)
        features['action'] = action
        self.dataset.append(features)
        
        #Save periodically to avoid data loss
        if len(self.dataset) % 100 == 0:
            df = pd.DataFrame(self.dataset)
            df.to_csv(self.dataset_file, index=False)
            print(f"Dataset saved with {len(self.dataset)} samples")
    
    def predict_action(self, features):
        """
        Predict the best action using the trained model
        
        Args:
            features: Dictionary of game state features
            
        Returns:
            String representing the action to take
        """
        if self.model is None:
            #Default to rule-based if no model is available
            return self.rule_based_action(features)
        
        #Convert features to format expected by the model
        X = pd.DataFrame([features])
        
        #Remove target column if present
        if 'action' in X.columns:
            X = X.drop('action', axis=1)
            
        #Fill missing values (if any)
        X = X.fillna(0)
        
        #Predict action
        try:
            action = self.model.predict(X)[0]
            return action
        except Exception as e:
            print(f"Error predicting action: {e}")
            return self.rule_based_action(features)
    
    def rule_based_action(self, features):
        """
        Rule-based fallback when model is not available
        
        Args:
            features: Dictionary of game state features
            
        Returns:
            String representing the action to take
        """
        distance = features['distance']
        direction = features['direction']
        
        #Basic strategy based on distance
        if distance > 60:  #Far away
            if direction > 0:  #Opponent is to the right
                return np.random.choice(['move_right', 'special1', 'jump_kick'])
            else:  #Opponent is to the left
                return np.random.choice(['move_left', 'special1', 'jump_kick'])
        elif distance < 30:  #Very close
            if features['opponent_in_move']:  #Opponent is attacking
                return 'special2'  #Defensive move
            else:
                return np.random.choice(['punch_y', 'kick_b', 'crouch'])
        else:  #Medium distance
            #Mix of movement and attacks
            return np.random.choice(['move_right', 'move_left', 'special1', 'jump_kick', 'punch_y'])
    
    def fight(self, current_game_state, player):
        """
        Main fight function that gets called by the controller
        
        Args:
            current_game_state: Current state of the game
            player: Player number ('1' or '2')
            
        Returns:
            Command object with button presses
        """
        #If already executing a multi-step action, continue
        if self.exe_code != 0:
            self.run_command([], current_game_state.player1 if player == "1" else current_game_state.player2)
            #Save which action is being executed
            self.save_state_to_dataset(current_game_state, player, self.current_action)
            
        else:
            #Extract features
            features = self.extract_features(current_game_state, player)
            
            #Add to state history
            self.state_history.append(features)
            
            #Predict action
            action = self.predict_action(features)
            self.current_action = action
            
            #Get button sequence for the action
            if action in self.action_sets:
                button_sequence = self.action_sets[action]
                
                #Execute the sequence
                player_obj = current_game_state.player1 if player == "1" else current_game_state.player2
                self.run_command(button_sequence, player_obj)
                
                #Save state and action to dataset
                self.save_state_to_dataset(current_game_state, player, action)
            else:
                #Default to idle
                self.run_command([], current_game_state.player1 if player == "1" else current_game_state.player2)
                self.save_state_to_dataset(current_game_state, player, 'idle')
        
        #Set the appropriate buttons in the command object
        if player == "1":
            self.my_command.player_buttons = self.buttn
        else:
            self.my_command.player2_buttons = self.buttn
            
        return self.my_command

    def run_command(self, com, player):
        """
        Execute a sequence of button presses
        
        Args:
            com: List of button commands to execute
            player: Player object
        """
        #Reset all buttons first
        self.buttn = Buttons()
        
        if self.exe_code - 1 == len(self.fire_code):
            #Complete the sequence
            self.exe_code = 0
            print("Command sequence completed")
            
        elif len(self.remaining_code) == 0:
            #Start new sequence
            self.fire_code = com
            self.exe_code += 1
            self.remaining_code = self.fire_code[0:]
            
        else:
            #Continue sequence
            self.exe_code += 1
            
            #Process button combinations
            cmd = self.remaining_code[0]
            
            #Handle all possible button combinations
            if "+" in cmd:
                #Combined button press
                parts = cmd.split("+")
                for part in parts:
                    if part.startswith("!"):
                        #Button release
                        btn = part[1:]
                        self.set_button(btn, False)
                    else:
                        #Button press
                        self.set_button(part, True)
            elif cmd.startswith("!"):
                #Button release
                btn = cmd[1:]
                self.set_button(btn, False)
            elif cmd == "-":
                #Delay/pause
                pass
            else:
                #Single button press
                self.set_button(cmd, True)
                
            #Move to next command in sequence
            self.remaining_code = self.remaining_code[1:]
        
        return
        
    def set_button(self, btn, value):
        """
        Set a specific button to a value
        
        Args:
            btn: Button to set
            value: Boolean value (True for press, False for release)
        """
        if btn == "v":
            self.buttn.down = value
        elif btn == "^":
            self.buttn.up = value
        elif btn == "<":
            self.buttn.left = value
        elif btn == ">":
            self.buttn.right = value
        elif btn == "Y":
            self.buttn.Y = value
        elif btn == "X":
            self.buttn.X = value
        elif btn == "B":
            self.buttn.B = value
        elif btn == "A":
            self.buttn.A = value
        elif btn == "L":
            self.buttn.L = value
        elif btn == "R":
            self.buttn.R = value
        elif btn == "select":
            self.buttn.select = value
        elif btn == "start":
            self.buttn.start = value

    def train_model(self):
        """
        Train the ML model using collected dataset
        This would typically be called separately, not during gameplay
        """
        if not os.path.exists(self.dataset_file):
            print("Dataset file not found.")
            return
            
        try:
            #Load dataset
            df = pd.read_csv(self.dataset_file)
            print(f"Loaded dataset with {len(df)} samples")
            
            if len(df) < 100:
                print("Dataset too small to train model effectively.")
                return
                
            #Separate features and target
            X = df.drop('action', axis=1)
            y = df['action']
            
            #Train a Random Forest model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            #Save the model
            joblib.dump(model, self.model_file)
            print(f"Model trained and saved to {self.model_file}")
            
            #Update the model
            self.model = model
            
        except Exception as e:
            print(f"Error training model: {e}")