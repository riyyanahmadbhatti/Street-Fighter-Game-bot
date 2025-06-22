AI Project Street Fighter II Bot

Firstly Download the zip of Street Fighter bot using the link below
https://drive.google.com/file/d/18SN8e_XqJFEPZ0wcWXQ8GnzuZk58cn-2/view?usp=sharing

Now add the files above into the PythonApi folder in the zip file 

This README provides instructions for running the Street Fighter II AI Bot, which uses machine
learning to play Street Fighter II Turbo on the BizHawk emulator.
Setup Instructions
Prerequisites
● Windows 7 or above (64-bit)
● Python 3.6.3 or above
● BizHawk emulator (provided in the project files)
● Street Fighter II Turbo ROM
Installation
1. Install Python dependencies:
pip install numpy pandas scikit-learn joblib matplotlib seaborn
2. Install BizHawk prerequisites from the provided ZIP file
Running the Project: Step-by-Step Guide
Step 1: Emulator Setup
1. Run EmuHawk.exe from the BizHawk folder
2. Go to File → Open ROM (Ctrl+O)
3. Select "Street Fighter II Turbo (U).smc" from the single-player folder
4. Go to Tools → Tool Box (Shift+T) and leave it open
Step 2: Data Collection
Data collection is required to train the machine learning model:
1. Open Command Prompt in the project directory
2. Run:
python data_collection.py 1
3. In the game, select a character and choose normal mode
4. Click the "Gyroscope Bot" icon (second icon in the top row) in the emulator
5. Play several rounds - the script will automatically collect data
6. Press Ctrl+C when done to save the dataset
Optional: Collect data for player 2:
python data_collection.py 2
Step 3: Model Training
After collecting data, train the ML model:
1. Run:
python model_trainer.py
2. The script will:
○ Merge datasets (if both player 1 and 2 data exists)
○ Train a Random Forest model
○ Evaluate model performance
○ Generate feature importance analysis
○ Save the model as sf2_model.joblib
Step 4: Running the AI Bot
To use the trained bot:
1. For player 1 (left side):
python controller.py 1
2. For player 2 (right side):
python controller.py 2
3. In the game, select character(s) and mode
4. Click the "Gyroscope Bot" icon to connect
5. The bot will play automatically using the trained model
Two-Player Mode (Bot vs Bot)
To have two bots fight each other:
1. Open two separate Command Prompt windows
2. In the first window:
python controller.py 1
3. In the second window:
python controller.py 2
4. Select VS Battle mode in the game
5. Click the "Gyroscope Bot" icon to connect both bots
Troubleshooting
Connection Issues
● Make sure the emulator is running before executing the Python script
● Check that the BizHawk prerequisites are installed
● Verify you clicked the "Gyroscope Bot" icon in the emulator
Model Performance Issues
● Collect more training data by playing additional rounds
● Try different characters to ensure the model generalizes well
● Verify that the sf2_model.joblib file exists in the project directory
Game Won't Start
● Ensure you're using the correct ROM file
● Check that the port (9999 for player 1, 10000 for player 2) isn't in use by another
application
File Overview
● bot.py: Main bot implementation using machine learning
● data_collection.py: Script for generating training data
● model_trainer.py: Script for training the ML model
● controller.py: Connects the bot to the game
● Various support files (buttons.py, command.py, etc.)
Notes
● The bot will only be as good as the training data it receives
● For best results, collect at least 5-10 full rounds of gameplay data
● The model automatically falls back to rule-based behavior if no trained model is found
