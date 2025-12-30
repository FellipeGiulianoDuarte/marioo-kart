# Formato de Pista - Mario Kart RL

Este documento explica o formato das pistas do jogo, as regras que devem ser seguidas e como criar novas pistas válidas.

## Estrutura Básica

Uma pista é definida como uma **string multi-linha** onde cada caractere representa um elemento de 50x50 pixels no grid.

### Exemplo de Pista
```python
track_string = """GGGGGGGGGGGGGGGGGGGGGGGGGG
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
                  GGGGGGGGGGGGGGGGGGGGGGGGGG"""
```

## Elementos da Pista

Cada caractere representa um tipo de bloco:

| Caractere | Nome | Descrição | Física |
|-----------|------|-----------|--------|
| **G** | Grass (Grama) | Área fora da pista | Fricção alta (0.2), penalidade pesada |
| **R** | Road (Estrada) | Pista normal | Fricção baixa (0.02), sem penalidade |
| **B** | Boost | Pad de aceleração | Velocidade fixa de 25, acelera o kart |
| **C** | Checkpoint 0 | Primeiro checkpoint | Deve ser atravessado primeiro |
| **D** | Checkpoint 1 | Segundo checkpoint | Deve ser atravessado segundo |
| **E** | Checkpoint 2 | Terceiro checkpoint | Deve ser atravessado terceiro |
| **F** | Checkpoint 3 | Quarto checkpoint (final) | Último checkpoint antes de completar |
| **L** | Lava | Área perigosa | Reseta kart ao último checkpoint, penalidade massiva |

## Regras Obrigatórias

### 1. Checkpoints em Ordem
Os checkpoints **DEVEM** ser atravessados na ordem: **C → D → E → F**

- Não é possível pular checkpoints
- Passar por um checkpoint fora de ordem não conta
- O kart termina a corrida ao passar pelo checkpoint F (se já passou por C, D, E)

### 2. Bordas da Pista
- **Todas as pistas devem ser cercadas por Grass (G)**
- Isso evita que o kart saia dos limites
- A primeira e última linha devem ser completamente G
- A primeira e última coluna devem ser completamente G

### 3. Dimensões
- Grid típico: **26 colunas x 16 linhas** (1300x800 pixels)
- Cada bloco: **50x50 pixels**
- Pode variar, mas mantenha proporcional

### 4. Conectividade
- Deve haver um **caminho navegável** do início até todos os checkpoints
- Não criar checkpoints isolados por Grass
- O caminho pode ser estreito, mas deve ser possível

### 5. Posição Inicial
- Deve estar em um bloco **R** (Road) ou **B** (Boost)
- Não pode estar em G (Grass) ou L (Lava)
- O ângulo inicial deve apontar na direção do primeiro checkpoint

## Ordem Lógica de Criação

Ao criar uma nova pista, siga esta ordem:

### 1. Defina o Layout Base
```
1. Desenhe as bordas externas com G
2. Desenhe o caminho principal com R
3. Adicione obstáculos internos com G
```

### 2. Posicione os Checkpoints
```
Regras de posicionamento:
- C: Próximo ao início, primeira curva ou reta
- D: Aproximadamente 25% da pista
- E: Aproximadamente 50-75% da pista
- F: Próximo ao final, antes de voltar ao início

Dica: Distribua uniformemente ao longo do circuito
```

### 3. Adicione Elementos Extras
```
- B (Boost): Em retas longas ou saídas de curva
- L (Lava): Armadilhas estratégicas em curvas ou atalhos arriscados
```

### 4. Teste o Caminho
```
Verifique se é possível navegar:
- Do início até C
- De C até D
- De D até E
- De E até F
- De F de volta ao início (opcional, para circuito fechado)
```

## Exemplos Práticos

### Pista 1 - Circuito Original

**Características**:
- Layout em forma de loop
- Checkpoints bem distribuídos
- Seção com obstáculo interno (bloco G central)
- Boosts em posições estratégicas
- Lava como armadilha

**Ordem dos Checkpoints**:
1. **C** (linha 2-4, coluna 7): Logo após o início
2. **D** (linha 10, coluna 22-25): Canto direito inferior
3. **E** (linha 11-12, coluna 6): Volta pelo lado esquerdo
4. **F** (linha 8, coluna 2-3): Final, próximo ao início

### Pista 2 - Circuito Espelhado

**Características**:
- Espelho horizontal da Pista 1
- Mesma lógica, direção invertida
- Checkpoints seguem a mesma ordem lógica (C→D→E→F)

## Como Criar um Espelho Válido

Para espelhar uma pista horizontalmente:

```python
def mirror_track(track_string):
    lines = track_string.strip().split('\n')
    mirrored = []
    for line in lines:
        # Remove espaços, inverte, reaplica espaços
        clean_line = line.strip()
        mirrored_line = clean_line[::-1]
        mirrored.append(mirrored_line)
    return '\n'.join(mirrored)
```

**IMPORTANTE**: Ao espelhar, os checkpoints **mantêm suas letras**, mas suas posições são invertidas. A ordem lógica C→D→E→F deve continuar fazendo sentido no circuito espelhado.

## Validação de Pista

Uma pista é válida se:

✅ **Estrutura**
- Todas as linhas têm o mesmo comprimento
- Existe pelo menos uma borda de G em todos os lados
- Não há caracteres inválidos

✅ **Checkpoints**
- Existem exatamente 4 checkpoints (C, D, E, F)
- Cada checkpoint aparece pelo menos 1 vez
- Os checkpoints formam uma sequência navegável

✅ **Navegabilidade**
- Existe caminho de R/B do início até C
- Existe caminho de C até D
- Existe caminho de D até E
- Existe caminho de E até F

✅ **Posição Inicial**
- Está dentro dos limites da pista
- Está em um bloco navegável (R, B, C, D, E, ou F)

## Erros Comuns

### ❌ Checkpoints Fora de Ordem Espacial
```
Problema: C está no final, F está no início
Solução: Reposicione para que a ordem física corresponda à lógica
```

### ❌ Checkpoint Isolado
```
Problema: Checkpoint D cercado completamente por G
Solução: Crie um caminho de R até o checkpoint
```

### ❌ Sem Bordas
```
Problema: Pista sem G nas bordas
Solução: Adicione G em toda a primeira/última linha/coluna
```

### ❌ Linhas Desalinhadas
```
Problema: Algumas linhas têm 26 caracteres, outras 25
Solução: Todas as linhas devem ter exatamente o mesmo comprimento
```

## Dicas de Design

### Pista Fácil
- Caminho largo (3-5 blocos de largura)
- Poucas curvas fechadas
- Checkpoints em posições óbvias
- Poucos ou nenhum obstáculo

### Pista Média
- Caminho médio (2-3 blocos de largura)
- Curvas moderadas
- Alguns obstáculos internos
- Boosts em retas

### Pista Difícil
- Caminho estreito (1-2 blocos de largura)
- Curvas fechadas
- Muitos obstáculos
- Lava em posições estratégicas
- Checkpoints em posições desafiadoras

## Física do Kart

Para entender o impacto de cada elemento:

```python
# Fricção
Road (R):    0.02  # Baixa fricção, alta velocidade
Grass (G):   0.20  # Alta fricção, desacelera muito
Boost (B):   Velocidade fixa de 25 pixels/frame

# Aceleração
MAX_ACCELERATION = 0.25  # Por frame
MAX_ANGLE_VELOCITY = 0.05  # Rotação máxima por frame

# Velocidade típica
Em R: 5-8 pixels/frame
Em G: 1-3 pixels/frame
Em B: 25 pixels/frame (fixo)
```

## Configuração no Code

Ao adicionar uma pista em `config.py`:

```python
'pista2': {
    'name': 'Nome Descritivo',
    'track_string': """...""",
    'initial_position': [x, y],  # Posição em pixels
    'initial_angle': 0,          # Radianos (0 = direita, π = esquerda)
    'best_model': 'pista2_best.pth',
    'warmstart_model': 'pista2_warmstart.pth',
    'target_steps': 300,         # Meta de passos
    'astar_benchmark': None      # Definir após rodar A*
}
```

## Ferramentas de Análise

Use estas ferramentas para analisar sua pista:

```bash
# Ver a pista visualmente
python main.py pista2

# Rodar A* para benchmark
python main.py pista2  # Conta os passos no terminal

# Analisar limites teóricos
python analysis/analyze_track.py
```

## Exemplo Completo

Aqui está um exemplo completo de como criar uma pista do zero:

```python
# 1. Layout base (26x16)
base = """GGGGGGGGGGGGGGGGGGGGGGGGGG
          GRRRRRRRRRRRRRRRRRRRRRRRRG
          GRRRRRRRRRRRRRRRRRRRRRRRRG
          ...
          GGGGGGGGGGGGGGGGGGGGGGGGGG"""

# 2. Adicionar checkpoints
# Percorra mentalmente o circuito e coloque C, D, E, F na ordem

# 3. Adicionar elementos
# Coloque B em retas, L em curvas perigosas

# 4. Testar
# Rode o jogo e veja se o A* consegue completar
```

## Resumo

**Uma pista válida precisa**:
1. Bordas de G em todos os lados
2. Checkpoints C, D, E, F nesta ordem lógica
3. Caminho navegável entre todos os checkpoints
4. Posição inicial válida
5. Linhas com mesmo comprimento

**Para criar uma nova pista**:
1. Desenhe o layout base
2. Posicione checkpoints na ordem C→D→E→F
3. Adicione boosts e lava
4. Teste com A*
5. Configure em `config.py`

---

**Dica Final**: Use `python main.py pista2` para ver sua pista em ação antes de treinar modelos!
