import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import numpy as np

from agent import Agent 

# The brain of our agent, defines the structure of the neural network of the agent
class DQN_CNN(nn.Module):
    """
    Tracks all layers and parameters (weights)
    Can switch between training and evaluation modes (.train(), .eval())
    """
    def __init__(self, board_size=10, n_frames=4, n_actions=3, version='v17.1'):
        super(DQN_CNN, self).__init__()

        # Loads the networks structure from json file
        with open(f'model_config/{version}.json', 'r') as f:
            config = json.load(f)
            model_config = config['model']
            board_size = config['board_size']
            n_frames = config['frames']

        # Builds the layers given from the json file
        self.layers = nn.ModuleList()
        
        # Define Conv layers based on the JSON file (which has 2 conv layers)
        conv1_params = model_config['Conv2D_1']
        conv2_params = model_config['Conv2D_2']
        
        self.layers.append(nn.Conv2d(in_channels=n_frames, 
                                     out_channels=conv1_params['filters'], 
                                     kernel_size=conv1_params['kernel_size']))
        
        self.layers.append(nn.Conv2d(in_channels=conv1_params['filters'], 
                                     out_channels=conv2_params['filters'], 
                                     kernel_size=conv2_params['kernel_size']))

        # Add the Flatten layer
        self.layers.append(nn.Flatten())

        flattened_size_from_convs = 1024

        # Define Dense layer based on the JSON file
        dense1_params = model_config['Dense_1']
        self.layers.append(nn.Linear(in_features=flattened_size_from_convs, 
                                     out_features=dense1_params['units']))

        # The final output layer maps the features from the last hidden layer to the number of actions the 
        # agent can perform
        in_features = dense1_params['units']
        self.output_layer = nn.Linear(in_features, n_actions)


    def forward(self, x):
            """Defines the forward pass of the network
            
            Input
            x - input tensor
            
            Return
            q_values - output_layer
            """

            # First Conv2D layer, followed by ReLU activation
            x = F.relu(self.layers[0](x))

            #Second Conv2D layer + ReLU
            x = F.relu(self.layers[1](x))

            # Flatten layer
            x = self.layers[2](x)

            # dense Linear layer + RELU
            x = F.relu(self.layers[3](x))

            # Output layer to get Q-values
            q_values = self.output_layer(x)
            
            return q_values

class DeepQLearningAgent(Agent):
    """
    Class containing the logic of the agents behaviour, (learning, acting, saving...)
    """

    # Constructor to initialize the agent
    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True, version='v17.1'):
        
        # Call the constructor of the parent Agent class.
        super().__init__(board_size=board_size, frames=frames, buffer_size=buffer_size,
                         gamma=gamma, n_actions=n_actions, use_target_net=use_target_net,
                         version=version)
        
        # Set device to cuda or cpu if cuda not avaiable
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create a instance of DQN_CNN 
        self._model = DQN_CNN(board_size, frames, n_actions, version).to(self.device)

        # Create a second identical instance of DQN_NN to provide stable targets during our training
        if self._use_target_net:
            self._target_net = DQN_CNN(board_size, frames, n_actions, version).to(self.device)
            self.update_target_net()

        # Setup the RMSprop optimizer
        # Setup the HuberLoss loss function
        self.optimizer = optim.RMSprop(self._model.parameters(), lr=0.0005)
        self.loss_fn = nn.HuberLoss()


    def _prepare_input(self, board):
        """
        Prepares a batch of games states for the network
        Handles normalization and channel permutation
        """

        # Ensures the input has a batch dimestions
        if board.ndim == 3:
            board = np.expand_dims(board, axis=0)

        # Rearanges the numpy dimensions to fit PyTorch's
        board = np.transpose(board, (0, 3, 1, 2))
        
        # Normalize the pixel values
        board = board.astype(np.float32) / 4.0

        # Convert the numpy array to a pyTorch tesnor and move it to GPU
        return torch.from_numpy(board).to(self.device)

    def _get_model_outputs(self, board):
        """
        Performs a froward pass to get raw Q-Values for a state.
        """
        state_tensor = self._prepare_input(board)
        
        self._model.eval()
        with torch.no_grad():
            q_values = self._model(state_tensor)
        
        return q_values.cpu().numpy()


    def move(self, board, legal_moves, value=None):
        """Selects the best action using the main model"""

        state_tensor = self._prepare_input(board)
        
        # Set model to evaluation mode
        self._model.eval()
        #Tells Pytorch to not calculate gradients (for faster interferance)
        with torch.no_grad():
            # Forward pass to get Q-values
            q_values = self._model(state_tensor)
        # Move q-values tensor to cpu to convert it ti a numpy array
        q_values = q_values.cpu().numpy()
        # Finds the best legal move (highest remaning Q-value)
        return np.argmax(np.where(legal_moves==1, q_values, -np.inf), axis=1)

    def update_target_net(self):
        """Copies the weights from the main model to the target model"""

        if self._use_target_net:
            self._target_net.load_state_dict(self._model.state_dict())

    def get_q_values(self, board):
        """Performs a froward pass to get Q-values for a state"""

        state_tensor = self._prepare_input(board)
        self._model.eval()
        with torch.no_grad():
            q_values = self._model(state_tensor)
        return q_values.cpu().numpy()

    def save_model(self, file_path='', iteration=None):
        """Saves the models state dictionary"""

        # Saves the models state_dict (the weights)
        if iteration is None: iteration = 0
        torch.save(self._model.state_dict(), f"{file_path}/model_{iteration:04d}.pth")

    def load_model(self, file_path='', iteration=None):
        """Loads the trained weights from a file to the main model"""
        if iteration is None: iteration = 0
        model_path = f"{file_path}/model_{iteration:04d}.pth"
        try:
            # Loards the saved weights
            self._model.load_state_dict(torch.load(model_path, map_location=self.device))

            # Updates the target networds to match the loaded network.
            if self._use_target_net:
                self.update_target_net()
            print(f"Successfully loaded model:{model_path}")
        except FileNotFoundError:
            print(f"Model not found:{model_path}")

    def train_agent(self, batch_size=32, num_games=1, reward_clip=False):
        """Performs one step of the training"""

        # Get rando, batch of past experiences
        s, a, r, next_s, done, legal_moves = self._buffer.sample(batch_size)
        
        #Clips rewards to -1, 0 or 1. Help stabilize training
        if reward_clip:
            r = np.sign(r)

        #Convert numpy arrays intor  PyTorch tensors and move to GPU/CPU
        states = self._prepare_input(s)
        next_states = self._prepare_input(next_s)
        actions = torch.from_numpy(np.argmax(a, axis=1)).unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(r).to(self.device, dtype=torch.float32)
        dones = torch.from_numpy(done).to(self.device, dtype=torch.float32)
        legal_moves_tensor = torch.from_numpy(legal_moves).to(self.device, dtype=torch.float32)

        # Calculate the Q_values using the target network for stable training
        # Belleman equation used
        model_for_target = self._target_net if self._use_target_net else self._model
        model_for_target.eval()
        with torch.no_grad():
            # Get Q-value for next states from the target network
            #Remove illegal moves
            #Find best Q-value from next state
            next_q_values = model_for_target(next_states)
            next_q_values[legal_moves_tensor == 0] = -float('inf')
            max_next_q = next_q_values.max(dim=1, keepdim=True)[0]
        
        #Belleman equation : Target = Reward + gamma * max_Q(next_state)
        target_q_values = rewards + (self._gamma * max_next_q * (1 - dones))

        #Set main model to training mode
        #Get Q-value prediction from the OG states from the main model
        #Get only the taken action Q-value
        self._model.train()
        q_values = self._model(states)
        predicted_q_values = q_values.gather(1, actions)
        
        # Calc the loss between predicted and target Q_value
        loss = self.loss_fn(predicted_q_values, target_q_values)

        #Clear old gradients from prev step
        #Calc gradients of the loss using the models weights
        #Update weights using our optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Return the scalar value of the loss
        return loss.item()