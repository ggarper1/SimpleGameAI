import torch.nn as nn

# Simple neural network model
class Game_Model(nn.Module):
    def __init__(self):
        super(Game_Model, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(37, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 9)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.shared(x)
        
        policy = self.policy_head(features)
        value = self.value_head(features)
        
        return policy, value
 
