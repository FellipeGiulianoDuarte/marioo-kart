# Guia de Treinamento - Mario Kart RL

## Scripts Disponíveis

### 1. Treinamento Evolutivo do Zero
```bash
python train_evolution.py
```
- Inicia população aleatória de 15 karts
- Algoritmo genético com seleção, crossover e mutação
- Salva modelos a cada 10 gerações em `rl/models/`

### 2. Continuar Treinamento Existente
```bash
# Continua do melhor modelo
python continue_training.py

# Continua de um modelo específico
python continue_training.py --model evolution_gen50

# Define número de gerações
python continue_training.py --generations 200

# Altera tamanho da população
python continue_training.py --population 20

# Combinando opções
python continue_training.py --model evolution_best --generations 100 --population 15
```

### 3. Testar Modelo Treinado
```bash
python race_visual.py
```
- Mostra 3 karts competindo:
  - Verde: A* (sempre funciona)
  - Azul: Modelo RL treinado
  - Vermelho: RL com exploração


## Modelos Salvos

Os modelos ficam em `rl/models/`:
- `evolution_best.pth` - Melhor modelo de todas as gerações
- `evolution_genX.pth` - Modelo da geração X (salvo a cada 10)

## Sistema de Fitness (Evolutivo)

O fitness é calculado baseado em:

### Recompensas
- **Checkpoints alcançados**: 2000 pontos cada
- **Progresso ao checkpoint**: até 500 pontos (quanto mais perto, melhor)
- **Velocidade**: até 300 pontos por checkpoint (menos passos = mais pontos)
- **Distância percorrida**: até 200 pontos

### Penalidades
- **Lava**: -500 pontos por hit
- **Ficar parado**: -2 pontos por frame sem movimento
- **Não fazer nada**: -1000 pontos

## Parâmetros de Evolução

### Seleção
- **Elite**: Top 3 passam direto para próxima geração
- **Torneio**: 4 candidatos competem, melhores 2 são pais
- Apenas top 50% podem ser pais

### Mutação
- Taxa inicial: 30%
- Decai para ~5% ao longo das gerações
- Adaptativa: menos mutação = convergência

### Crossover
- 50% chance de pegar peso do pai 1 ou pai 2
- Para cada camada da rede neural

## Dicas de Treinamento

### Quanto treinar?
- **0-20 gerações**: Aprendendo física básica
- **20-50 gerações**: Alcançando checkpoint 1
- **50-100 gerações**: Progredindo em múltiplos checkpoints
- **100-200 gerações**: Refinamento e otimização de rota

### População
- **Pequena (5-10)**: Convergência rápida, menos diversidade
- **Média (15-20)**: Balanço ideal
- **Grande (30+)**: Mais diversidade, treino mais lento

### Quando parar?
- Se fitness não melhora por 20+ gerações
- Se já alcançou todos os checkpoints consistentemente
- Se quer apenas performance razoável: 50 gerações são suficientes

## Interpretando Output

```
GERAÇÃO X COMPLETA
Melhor kart: #ID
  Fitness: 2500.5
  Checkpoints: 1/4          <- Progresso
  Passos: 450              <- Menos = melhor
  Dist. min ao CP: 120.5   <- Chegou perto?
  Lava hits: 2             <- Quantas vezes bateu
  Preso (frames): 50       <- Ficou travado?
```

### Fitness Típico
- 0-500: Aprendendo a se mover
- 500-1500: Progredindo mas não chegando em CP
- 1500-2500: Alcançando checkpoint 1
- 2500-4500: 2 checkpoints
- 4500+: 3 ou mais checkpoints

## Comandos Úteis

```bash
# Ver modelos salvos
ls -lth rl/models/

# Ver métricas de treino
ls -lth metrics/

# Limpar modelos antigos
rm rl/models/evolution_gen*.pth

# Backup do melhor modelo
cp rl/models/evolution_best.pth rl/models/backup_$(date +%Y%m%d).pth
```

## Controles Durante Treinamento

- **ESC**: Parar treinamento e salvar modelo atual
- **Fechar janela**: Parar imediatamente

## Troubleshooting

### Karts ficam parados
- Aumente taxa de mutação inicial
- Treine por mais gerações
- Verifique se fitness está recompensando movimento

### Convergência prematura
- Aumente tamanho da população
- Aumente elite_size
- Reduza taxa de mutação

### Performance lenta
- Reduza população
- Reduza FPS (editar clock.tick em train_evolution.py)
- Use modo sem visualização

## Próximos Passos

Depois de treinar:
1. Teste com `race_visual.py`
2. Compare com A* (verde)
3. Continue treinando se necessário
4. Experimente diferentes tracks em `main.py`
