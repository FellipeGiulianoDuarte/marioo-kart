import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Deep Q-Network para controlar o kart.

    Recebe o estado do kart e retorna Q-values para cada ação possível.
    """

    def __init__(self, state_size, action_size, hidden_size=128):
        """
        Args:
            state_size: Dimensão do vetor de estado
            action_size: Número de ações possíveis
            hidden_size: Tamanho das camadas ocultas
        """
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        """
        Forward pass da rede neural.

        Args:
            x: Estado do ambiente (tensor)

        Returns:
            Q-values para cada ação
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
