import random
from collections import deque, namedtuple


# Estrutura para armazenar transições
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory:
    """
    Memória de replay para armazenar experiências do agente.

    Usado no treinamento DQN para quebrar a correlação temporal
    entre experiências consecutivas.
    """

    def __init__(self, capacity):
        """
        Args:
            capacity: Capacidade máxima da memória
        """
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        """
        Adiciona uma transição à memória.

        Args:
            state: Estado atual
            action: Ação tomada
            next_state: Próximo estado
            reward: Recompensa recebida
            done: Se o episódio terminou
        """
        self.memory.append(Transition(state, action, next_state, reward, done))

    def sample(self, batch_size):
        """
        Amostra um batch aleatório de transições.

        Args:
            batch_size: Tamanho do batch

        Returns:
            Lista de transições
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Retorna o tamanho atual da memória."""
        return len(self.memory)
