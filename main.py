from track import Track
from ai import AI
from human import Human
from kart import Kart

BLOCK_SIZE = 50

def main():

    track =  """GGGGGGGGGGGGGGGGGGGGGGGGGG
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

    kart_initial_position = [75, 75]
    kart_initial_angle = 0

    controller = AI()  # ou AI()
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