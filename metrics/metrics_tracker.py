import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import json


class MetricsTracker:
    """
    Classe para rastrear e visualizar métricas de treinamento.

    Salva métricas periodicamente e gera gráficos de evolução.
    """

    def __init__(self, save_dir="metrics", save_frequency=10):
        """
        Args:
            save_dir: Diretório para salvar métricas e gráficos
            save_frequency: Frequência (em episódios) para salvar gráficos
        """
        self.save_dir = save_dir
        self.save_frequency = save_frequency

        # Criar diretório se não existir
        os.makedirs(save_dir, exist_ok=True)

        # Métricas
        self.episodes = []
        self.rewards = []
        self.steps = []
        self.checkpoints = []
        self.losses = []
        self.epsilons = []
        self.success_rate = []

        # Timestamp de início
        self.start_time = datetime.now()
        self.run_id = self.start_time.strftime("%Y%m%d_%H%M%S")

        print(f"MetricsTracker iniciado - Run ID: {self.run_id}")

    def add_episode(self, episode, reward, steps, checkpoints, loss=None, epsilon=None):
        """
        Adiciona métricas de um episódio.

        Args:
            episode: Número do episódio
            reward: Recompensa total do episódio
            steps: Número de passos no episódio
            checkpoints: Número de checkpoints alcançados
            loss: Loss médio do episódio (opcional)
            epsilon: Valor de epsilon (opcional)
        """
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.steps.append(steps)
        self.checkpoints.append(checkpoints)

        if loss is not None:
            self.losses.append(loss)

        if epsilon is not None:
            self.epsilons.append(epsilon)

        # Calcular taxa de sucesso (últimos 100 episódios)
        recent_completions = [1 if cp >= 4 else 0 for cp in self.checkpoints[-100:]]
        self.success_rate.append(np.mean(recent_completions) * 100)

        # Salvar gráficos periodicamente
        if episode % self.save_frequency == 0:
            self.plot_metrics(episode)
            self.save_metrics()

    def plot_metrics(self, episode):
        """
        Gera gráficos das métricas de treinamento.

        Args:
            episode: Número do episódio atual
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Métricas de Treinamento - Episódio {episode} (Run: {self.run_id})', fontsize=16)

        # 1. Recompensa por episódio
        ax = axes[0, 0]
        ax.plot(self.episodes, self.rewards, alpha=0.3, color='blue', label='Recompensa')
        if len(self.rewards) >= 10:
            moving_avg = self._moving_average(self.rewards, window=10)
            ax.plot(self.episodes, moving_avg, color='red', linewidth=2, label='Média Móvel (10)')
        ax.set_xlabel('Episódio')
        ax.set_ylabel('Recompensa Total')
        ax.set_title('Recompensa por Episódio')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Passos por episódio
        ax = axes[0, 1]
        ax.plot(self.episodes, self.steps, alpha=0.3, color='green', label='Passos')
        if len(self.steps) >= 10:
            moving_avg = self._moving_average(self.steps, window=10)
            ax.plot(self.episodes, moving_avg, color='red', linewidth=2, label='Média Móvel (10)')
        ax.set_xlabel('Episódio')
        ax.set_ylabel('Número de Passos')
        ax.set_title('Passos por Episódio')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Checkpoints alcançados
        ax = axes[0, 2]
        ax.plot(self.episodes, self.checkpoints, alpha=0.3, color='purple', label='Checkpoints')
        if len(self.checkpoints) >= 10:
            moving_avg = self._moving_average(self.checkpoints, window=10)
            ax.plot(self.episodes, moving_avg, color='red', linewidth=2, label='Média Móvel (10)')
        ax.axhline(y=4, color='orange', linestyle='--', label='Meta (4 checkpoints)')
        ax.set_xlabel('Episódio')
        ax.set_ylabel('Checkpoints')
        ax.set_title('Checkpoints Alcançados por Episódio')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 5)

        # 4. Loss (se disponível)
        ax = axes[1, 0]
        if len(self.losses) > 0:
            ax.plot(self.episodes[:len(self.losses)], self.losses, alpha=0.3, color='orange', label='Loss')
            if len(self.losses) >= 10:
                moving_avg = self._moving_average(self.losses, window=10)
                ax.plot(self.episodes[:len(self.losses)], moving_avg, color='red', linewidth=2, label='Média Móvel (10)')
            ax.legend()
        ax.set_xlabel('Episódio')
        ax.set_ylabel('Loss Médio')
        ax.set_title('Loss durante Treinamento')
        ax.grid(True, alpha=0.3)

        # 5. Epsilon (se disponível)
        ax = axes[1, 1]
        if len(self.epsilons) > 0:
            ax.plot(self.episodes[:len(self.epsilons)], self.epsilons, color='brown', label='Epsilon')
            ax.legend()
        ax.set_xlabel('Episódio')
        ax.set_ylabel('Epsilon (Taxa de Exploração)')
        ax.set_title('Decaimento de Epsilon')
        ax.grid(True, alpha=0.3)

        # 6. Taxa de sucesso
        ax = axes[1, 2]
        if len(self.success_rate) > 0:
            ax.plot(self.episodes, self.success_rate, color='teal', label='Taxa de Sucesso (%)')
            ax.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Meta 80%')
            ax.legend()
        ax.set_xlabel('Episódio')
        ax.set_ylabel('Taxa de Sucesso (%)')
        ax.set_title('Taxa de Sucesso (Últimos 100 Eps)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

        plt.tight_layout()

        # Salvar gráfico
        filename = os.path.join(self.save_dir, f'metrics_ep{episode}_{self.run_id}.png')
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"  Gráficos salvos em: {filename}")

    def save_metrics(self):
        """Salva métricas em formato JSON."""
        metrics_data = {
            'run_id': self.run_id,
            'start_time': self.start_time.isoformat(),
            'episodes': self.episodes,
            'rewards': self.rewards,
            'steps': self.steps,
            'checkpoints': self.checkpoints,
            'losses': self.losses,
            'epsilons': self.epsilons,
            'success_rate': self.success_rate
        }

        filename = os.path.join(self.save_dir, f'metrics_{self.run_id}.json')
        with open(filename, 'w') as f:
            json.dump(metrics_data, f, indent=2)

    def print_summary(self):
        """Imprime um resumo das métricas."""
        print("\n" + "=" * 60)
        print("RESUMO DE TREINAMENTO")
        print("=" * 60)

        if len(self.rewards) > 0:
            print(f"Total de episódios: {len(self.episodes)}")
            print(f"\nRecompensas:")
            print(f"  Média: {np.mean(self.rewards):.2f}")
            print(f"  Melhor: {np.max(self.rewards):.2f} (Ep {self.episodes[np.argmax(self.rewards)]})")
            print(f"  Últimos 100: {np.mean(self.rewards[-100:]):.2f}")

            print(f"\nPassos:")
            print(f"  Média: {np.mean(self.steps):.2f}")
            print(f"  Menor: {np.min(self.steps)} (Ep {self.episodes[np.argmin(self.steps)]})")
            print(f"  Últimos 100: {np.mean(self.steps[-100:]):.2f}")

            print(f"\nCheckpoints:")
            print(f"  Média: {np.mean(self.checkpoints):.2f}")
            print(f"  Melhor: {np.max(self.checkpoints)} (Ep {self.episodes[np.argmax(self.checkpoints)]})")
            print(f"  Últimos 100: {np.mean(self.checkpoints[-100:]):.2f}")

            completions = sum(1 for cp in self.checkpoints if cp >= 4)
            print(f"\nCompletou a pista: {completions}/{len(self.checkpoints)} vezes ({completions/len(self.checkpoints)*100:.1f}%)")

            if len(self.success_rate) > 0:
                print(f"Taxa de sucesso atual (últimos 100): {self.success_rate[-1]:.1f}%")

        print("=" * 60)

    def _moving_average(self, data, window=10):
        """
        Calcula média móvel dos dados.

        Args:
            data: Lista de valores
            window: Tamanho da janela

        Returns:
            Lista com médias móveis
        """
        if len(data) < window:
            return data

        moving_avg = []
        for i in range(len(data)):
            if i < window - 1:
                moving_avg.append(np.mean(data[:i+1]))
            else:
                moving_avg.append(np.mean(data[i-window+1:i+1]))

        return moving_avg

    def load_metrics(self, json_file):
        """
        Carrega métricas de um arquivo JSON.

        Args:
            json_file: Caminho para o arquivo JSON
        """
        with open(json_file, 'r') as f:
            data = json.load(f)

        self.run_id = data['run_id']
        self.episodes = data['episodes']
        self.rewards = data['rewards']
        self.steps = data['steps']
        self.checkpoints = data['checkpoints']
        self.losses = data.get('losses', [])
        self.epsilons = data.get('epsilons', [])
        self.success_rate = data.get('success_rate', [])

        print(f"Métricas carregadas de: {json_file}")
