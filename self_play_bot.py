from constant_strings import MAX_MOVE_COUNT_WITH_INITIAL_TEMPERATURE_CONTROL, TEMPERATURE_CONTROL_FOR_MIN_RANDOMNESS, \
    TEMPERATURE_CONTROL_FOR_MAX_RANDOMNESS
from tictactoe_variant import Tictactoe

class SelfPlayBot:
    def __init__(self):
        # trying to keep records of game runs for each board size
        self.game_info = {
            2 : 0,
            3 : 0,
            4 : 0,
            5 : 0,
            6 : 0,
            7 : 0
        }

    def tweak_temp_control_based_on_move_count(self, tictactoe):
         move_count = tictactoe.get_total_move_count()
         size = tictactoe.get_board_size()
         # We flip the temperature control after a certain number of games for optimized move selection
         if (size * 2)- move_count < MAX_MOVE_COUNT_WITH_INITIAL_TEMPERATURE_CONTROL:
             tictactoe.set_temperature_control(TEMPERATURE_CONTROL_FOR_MIN_RANDOMNESS)

    def setup_tictactoe_instance_for_simulations(self, size):
        tictactoe = Tictactoe(size=size, vs_human=False)
        tictactoe.set_temperature_control(TEMPERATURE_CONTROL_FOR_MAX_RANDOMNESS)
        # simulation mode so AI starts with the first move
        tictactoe.ai_player_code = 0
        tictactoe.set_to_simulation_mode()
        return tictactoe

    def run_simulations(self, size):
        count = 0
        tictactoe = self.setup_tictactoe_instance_for_simulations(size)
        while count < 5:
            tictactoe.run_game()
            count += 1











