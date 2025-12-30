"""
Configuração centralizada do Mario Kart RL.
Todas as constantes e variáveis de ambiente do projeto.
"""

# ============================================================================
# FÍSICA DO KART
# ============================================================================
MAX_ANGLE_VELOCITY = 0.05  # Velocidade máxima de rotação
MAX_ACCELERATION = 0.25    # Aceleração/desaceleração
BOOST_VELOCITY = 25        # Velocidade do boost
KART_RADIUS = 20          # Raio do kart para desenho

# ============================================================================
# DIMENSÕES DA PISTA
# ============================================================================
BLOCK_SIZE = 50  # Tamanho de cada bloco da pista em pixels

# ============================================================================
# PISTAS - Sistema de múltiplas pistas
# ============================================================================

# PISTA 1: Circuito original (271 passos com NN, 440 com A*)
TRACKS = {
    'pista1': {
        'name': 'Circuito Original',
        'track_string': """GGGGGGGGGGGGGGGGGGGGGGGGGG
                GRRRRRRCRRRRRRRRRBRRRRRRRG
                GRRRRRRCRRRRRRRRRBRRRRRRRG
                GRRRRRRCRRRRRRRRRRRRRRRRRG
                GRRRRRRCRRRRRRRRRRRRRRRRRG
                GGGGGGGGGGGGGGGGGGGGGRRRRG
                GGGGGGGGGGGGGGGGGGGGGRRRRG
                GRRRRGGGGGGGGGGGGGGGGRRRRG
                GFFRRGGGGGGGGGGGGGGGGRRRRG
                GLRRRGGGGGGGGGGGGGGGGRRRRG
                GRRRRGGGGGGGGGGGGGGGGDDDDG
                GRRRRRERRRRRRRBRRRRRRRRLLG
                GRRRRRERRRRRRRBRRRRRRRRRRG
                GLRRRRERRRRRGGBRRRRRRRRRRG
                GLLRRRERRRRRGGBRRRRRRRRRRG
                GGGGGGGGGGGGGGGGGGGGGGGGGG""",
        'initial_position': [75, 75],
        'initial_angle': 0,
        'best_model': 'pista1_best.pth',
        'warmstart_model': 'pista1_warmstart.pth',
        'target_steps': 260,
        'astar_benchmark': 440
    },

    # PISTA RETANGULAR: Circuito simples retangular
    'pista_retangular': {
        'name': 'Circuito Retangular',
        'track_string': """GGGGGGGGGGGGGGGGGGGGGGGGGG
                GRRRRRRCRRRRRRRRRBRRRRRRRG
                GRRRRRRCRRRRRRRRRBRRRRRRRG
                GRRRRRRCRRRRRRRRRBRRRRRRRG
                GRRRRRRCRRRRRRRRRBRRRRRRRG
                GRRRRGGGGGGGGGGGGGGGGRRRRG
                GRRRRGGGGGGGGGGGGGGGGRRRRG
                GFFFFGGGGGGGGGGGGGGGGDDDDG
                GFFFFGGGGGGGGGGGGGGGGDDDDG
                GRRRRGGGGGGGGGGGGGGGGRRRRG
                GRRRRGGGGGGGGGGGGGGGGRRRRG
                GRRRRRRERRRRRRRRRBRRRRRRRG
                GRRRRRRERRRRRRRRRBRRRRRRRG
                GRRRRRRERRRRRRRRRBRRRRRRRG
                GRRRRRRERRRRRRRRRBRRRRRRRG
                GGGGGGGGGGGGGGGGGGGGGGGGGG""",
        'initial_position': [75, 75],
        'initial_angle': 0,
        'best_model': None,
        'warmstart_model': None,
        'target_steps': 300,
        'astar_benchmark': 462
    },
    'pista_retangular_lava': {
        'name': 'Circuito Retangular com Lava',
        'track_string': """GGGGGGGGGGGGGGGGGGGGGGGGGG
                GRRRRRRCRRRRLRRRRBRRRRRRRG
                GRRRRRRCRRRRLRRRRBRRRLLRRG
                GRRRRRRCRRRRRRRRRBRRRRLRRG
                GRRRRRRCRRRRRRRRRBRRRRRRRG
                GRRRRGGGGGGGGGGGGGGGGRRRRG
                GRRRRGGGGGGGGGGGGGGGGRRRRG
                GFFFFGGGGGGGGGGGGGGGGDDDDG
                GFFFFGGGGGGGGGGGGGGGGDDDDG
                GRRRRGGGGGGGGGGGGGGGGRRRRG
                GRRRRGGGGGGGGGGGGGGGGRRRRG
                GRRRRRRERRRRLRRRRBRRRRRRRG
                GRLRRRRERRRRLRRRRBRRRRRRRG
                GRLLRRRERRRRRRRRRBRRLRRRRG
                GRRRRRRERRRRRRRRRBRRLRRRRG
                GGGGGGGGGGGGGGGGGGGGGGGGGG""",
        'initial_position': [75, 75],
        'initial_angle': 0,
        'best_model': None,
        'warmstart_model': None,
        'target_steps': 300,
        'astar_benchmark': 907
    }
}

# Pista ativa (pode ser alterada dinamicamente)
ACTIVE_TRACK = 'pista1'

# Funções auxiliares para acessar pista ativa
def get_active_track():
    """Retorna a configuração da pista ativa."""
    return TRACKS[ACTIVE_TRACK]

def set_active_track(track_id):
    """Muda a pista ativa."""
    global ACTIVE_TRACK
    if track_id not in TRACKS:
        raise ValueError(f"Pista '{track_id}' não existe. Pistas disponíveis: {list(TRACKS.keys())}")
    ACTIVE_TRACK = track_id

# Aliases para compatibilidade com código existente
DEFAULT_TRACK = TRACKS[ACTIVE_TRACK]['track_string']
KART_INITIAL_POSITION = TRACKS[ACTIVE_TRACK]['initial_position']
KART_INITIAL_ANGLE = TRACKS[ACTIVE_TRACK]['initial_angle']

# ============================================================================
# TREINAMENTO EVOLUTIVO
# ============================================================================
# População
DEFAULT_POPULATION_SIZE = 50
ELITE_SIZE = 3  # Top N passam direto para próxima geração
HALL_OF_FAME_SIZE = 5  # Melhores de TODAS as gerações

# Mutação
MUTATION_RATE_INITIAL = 0.2  # Taxa inicial (mais conservador)
MUTATION_RATE_MIN = 0.03     # Taxa mínima de mutação

# Episódio
MAX_STEPS_PER_EPISODE = 700  # Mais tempo para aprender

# Benchmark (valores da pista ativa)
ASTAR_BENCHMARK_STEPS = TRACKS[ACTIVE_TRACK]['astar_benchmark']
TARGET_STEPS = TRACKS[ACTIVE_TRACK]['target_steps']

# Fitness - Estratégia FOCADA
# Se NÃO COMPLETA a pista: fitness baseado só em progresso (ruim)
# Se COMPLETA a pista: fitness MASSIVO + bônus por velocidade

COMPLETION_BASE_REWARD = 50000      # Recompensa GIGANTE por completar
SPEED_EXCELLENCE_BONUS = 20000      # Bônus se completar < 400 passos
BEAT_ASTAR_BONUS = 10000            # Bônus se bater A* (< 440 passos)

# Para karts que NÃO completam (muito menor que completar)
CHECKPOINT_PARTIAL_REWARD = 1000    # Por checkpoint (se não completa)
PROGRESS_PARTIAL_REWARD = 300       # Progresso ao próximo CP

# Penalidades SEVERAS
GRASS_PENALTY_PER_FRAME = 10        # MUITO caro ficar na grama
LAVA_PENALTY = 2000                 # Penalidade MASSIVA por lava
STUCK_PENALTY = 5                   # Penalidade por ficar parado
TIMEOUT_PENALTY = 5000              # Penalidade por timeout sem completar

# ============================================================================
# VISUALIZAÇÃO
# ============================================================================
DEFAULT_FPS = 60
FAST_TRAINING_FPS = 240
ULTRA_FAST_FPS = 1000

# ============================================================================
# REINFORCEMENT LEARNING (DQN)
# ============================================================================
STATE_SIZE = 12
ACTION_SIZE = 7

# Rede Neural
HIDDEN_LAYER_1 = 128
HIDDEN_LAYER_2 = 128
HIDDEN_LAYER_3 = 64

# Hiperparâmetros
LEARNING_RATE = 0.001
GAMMA = 0.99              # Fator de desconto
EPSILON_START = 1.0       # Exploração inicial
EPSILON_END = 0.01        # Exploração mínima
EPSILON_DECAY = 0.995     # Taxa de decaimento

BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE = 10        # Atualizar rede target a cada N episódios

# ============================================================================
# PATHS
# ============================================================================
MODEL_DIR = "rl/models"
METRICS_DIR = "metrics"

# Nomes de modelos (dinâmico baseado na pista ativa)
EVOLUTION_BEST_MODEL = TRACKS[ACTIVE_TRACK]['best_model']
WARMSTART_MODEL = TRACKS[ACTIVE_TRACK]['warmstart_model']

# Modelo generalista (treinado em múltiplas pistas)
GENERALIST_MODEL = "generalist_model.pth"

# ============================================================================
# A* PATHFINDING
# ============================================================================
# Custos de terreno para A*
TERRAIN_COSTS = {
    'R': 3,    # Road (estrada)
    'B': 1,    # Boost
    'G': 300,  # Grass (grama - muito custoso)
    'C': 3,    # Checkpoint 0
    'D': 3,    # Checkpoint 1
    'E': 3,    # Checkpoint 2
    'F': 3,    # Checkpoint 3
    'L': 100,  # Lava (evitar)
}
