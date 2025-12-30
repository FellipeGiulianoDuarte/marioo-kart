import sys
import config
from game.entities.track import Track
from game.controllers.ai import AI
from game.controllers.human import Human
from game.entities.kart import Kart

BLOCK_SIZE = 50

def main():
    # Suporte para escolher pista via argumento
    track_id = 'pista1'  # Default
    if len(sys.argv) > 1:
        track_id = sys.argv[1]

    # Configurar pista ativa
    if track_id not in config.TRACKS:
        print(f"❌ Pista '{track_id}' não existe.")
        print(f"   Pistas disponíveis: {list(config.TRACKS.keys())}")
        return

    config.set_active_track(track_id)
    track_config = config.get_active_track()

    if track_config['track_string'] is None:
        print(f"❌ Pista '{track_id}' não está definida ainda.")
        return

    print(f"\n{'='*60}")
    print(f"MARIO KART - {track_config['name']}")
    print(f"{'='*60}\n")

    track = track_config['track_string']
    kart_initial_position = track_config['initial_position']
    kart_initial_angle = track_config['initial_angle']

    controller = AI()  # ou Human()
    """
    ==================== ATTENTION =====================
    In the original requirement, the lines of code below
    were indicated not to be changed.
    ====================================================
    """
    kart = Kart(controller)
    track = Track(track, kart_initial_position, kart_initial_angle)
    track.add_kart(kart)
    track.play()

if __name__ == '__main__':
    main()