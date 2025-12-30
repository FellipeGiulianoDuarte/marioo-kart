import math
import random
import torch
import numpy as np
import pygame

from rl.dqn import DQN
from game.elements.boost import Boost
from game.elements.grass import Grass
from game.elements.road import Road
from game.elements.checkpoint import Checkpoint
from game.elements.lava import Lava

BLOCK_SIZE = 50


class RLController:
    """
    Controlador de RL usando Deep Q-Learning para treinar o kart.

    O agente aprende a navegar pela pista através de tentativa e erro,
    recebendo recompensas por passar checkpoints e completar a pista rapidamente.
    """

    def __init__(self, state_size=12, action_size=7, learning_rate=0.001, device=None):
        """
        Args:
            state_size: Tamanho do vetor de estado
            action_size: Número de ações possíveis
            learning_rate: Taxa de aprendizado
            device: Dispositivo (CPU/GPU)
        """
        self.kart = None
        self.action_size = action_size

        # Configurar dispositivo (CPU ou GPU)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Redes neurais (policy e target)
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Otimizador
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Parâmetros de exploração
        self.epsilon = 1.0  # Taxa inicial de exploração
        self.epsilon_min = 0.01  # Taxa mínima de exploração
        self.epsilon_decay = 0.995  # Decaimento da exploração

        # Ações possíveis (combinações de teclas)
        # [UP, DOWN, LEFT, RIGHT]
        self.actions = [
            [True, False, False, False],   # 0: Frente
            [True, False, True, False],    # 1: Frente + Esquerda
            [True, False, False, True],    # 2: Frente + Direita
            [False, False, True, False],   # 3: Apenas Esquerda
            [False, False, False, True],   # 4: Apenas Direita
            [False, True, False, False],   # 5: Ré
            [False, False, False, False]   # 6: Nenhuma ação
        ]

    def get_state(self, string):
        """
        Extrai o estado atual do ambiente.

        Estado inclui:
        - Posição x, y do kart
        - Ângulo do kart
        - Velocidade x, y
        - Distância e ângulo relativo ao próximo checkpoint
        - Tipo de terreno atual
        - Distâncias a obstáculos em 4 direções

        Args:
            string: String representando a pista

        Returns:
            Vetor de estado normalizado
        """
        # Posição e orientação
        pos_x, pos_y = self.kart.position
        angle = self.kart.angle

        # Velocidade
        speed_x, speed_y = self.kart.get_speed()
        speed = math.sqrt(speed_x**2 + speed_y**2)

        # Próximo checkpoint
        checkpoint_chars = ['C', 'D', 'E', 'F']
        next_checkpoint_char = checkpoint_chars[self.kart.next_checkpoint_id]

        # Encontrar posição do checkpoint
        rows = string.split('\n')
        rows = [row.strip() for row in rows if row.strip()]
        checkpoint_x, checkpoint_y = 0, 0

        for row_index in range(len(rows)):
            row = rows[row_index]
            for col_index in range(len(row)):
                if row[col_index] == next_checkpoint_char:
                    checkpoint_x = col_index * BLOCK_SIZE + BLOCK_SIZE // 2
                    checkpoint_y = row_index * BLOCK_SIZE + BLOCK_SIZE // 2
                    break

        # Distância e ângulo ao checkpoint
        dx = checkpoint_x - pos_x
        dy = checkpoint_y - pos_y
        distance_to_checkpoint = math.sqrt(dx**2 + dy**2)
        angle_to_checkpoint = math.atan2(dy, dx)
        relative_angle = (angle_to_checkpoint - angle + math.pi) % (2 * math.pi) - math.pi

        # Tipo de terreno atual
        track_element, _ = self.kart.get_track_element(string, pos_x, pos_y)
        terrain_type = 0
        if track_element == Road:
            terrain_type = 0
        elif track_element == Grass:
            terrain_type = 1
        elif track_element == Boost:
            terrain_type = 2
        elif track_element == Checkpoint:
            terrain_type = 3

        # Sensores de distância (4 direções)
        sensor_distance = 100  # Distância máxima dos sensores
        sensor_angles = [0, math.pi/2, math.pi, -math.pi/2]  # Frente, direita, trás, esquerda
        sensor_readings = []

        for sensor_angle in sensor_angles:
            absolute_angle = angle + sensor_angle
            check_x = pos_x + sensor_distance * math.cos(absolute_angle)
            check_y = pos_y + sensor_distance * math.sin(absolute_angle)
            element, _ = self.kart.get_track_element(string, check_x, check_y)

            # Normalizar: 1 se seguro (Road, Boost, Checkpoint), 0 se perigoso (Grass, Lava)
            if element in (Road, Boost, Checkpoint):
                sensor_readings.append(1.0)
            else:
                sensor_readings.append(0.0)

        # Montar vetor de estado normalizado
        state = np.array([
            pos_x / 1000.0,  # Normalizar posição
            pos_y / 1000.0,
            math.cos(angle),  # Usar cos/sin para representar ângulo
            math.sin(angle),
            speed / 10.0,  # Normalizar velocidade
            distance_to_checkpoint / 1000.0,  # Normalizar distância
            math.cos(relative_angle),  # Ângulo relativo ao checkpoint
            math.sin(relative_angle),
            terrain_type / 3.0,  # Normalizar tipo de terreno
            sensor_readings[0],  # Sensor frente
            sensor_readings[1],  # Sensor direita
            sensor_readings[2],  # Sensor esquerda
        ], dtype=np.float32)

        return state

    def select_action(self, state, training=True):
        """
        Seleciona uma ação usando epsilon-greedy.

        Args:
            state: Estado atual
            training: Se está em modo de treinamento (usa exploração)

        Returns:
            Índice da ação selecionada
        """
        if training and random.random() < self.epsilon:
            # Exploração: ação aleatória
            return random.randrange(self.action_size)
        else:
            # Exploração: melhor ação segundo a rede
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(1).item()

    def move(self, string):
        """
        Decide a próxima ação do kart.

        Args:
            string: String representando a pista

        Returns:
            Dicionário de teclas pressionadas
        """
        state = self.get_state(string)
        action_idx = self.select_action(state, training=False)
        action = self.actions[action_idx]

        key_list = [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]
        keys = {key: action[i] for i, key in enumerate(key_list)}

        return keys

    def decay_epsilon(self):
        """Reduz a taxa de exploração após cada episódio."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        """Atualiza a rede target com os pesos da rede policy."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath):
        """
        Salva o modelo treinado.

        Args:
            filepath: Caminho do arquivo
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"Modelo salvo em {filepath}")

    def load(self, filepath):
        """
        Carrega um modelo treinado.

        Args:
            filepath: Caminho do arquivo
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Modelo carregado de {filepath}")
