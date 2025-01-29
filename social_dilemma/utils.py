import numpy as np
from pycolab.rendering import ObservationToArray


def build_map(map_sketch, num_pad_pixels, agent_chars, starting_side='top'):
    numAgents = len(agent_chars)
    gameMap = np.array(map_sketch)

    def pad_with(vector, pad_width, iaxis, kwargs):
        del iaxis
        padValue = kwargs.get('padder', ' ')
        vector[:pad_width[0]] = padValue
        vector[-pad_width[1]:] = padValue
        return vector

    # Define starting coordinates
    if starting_side == 'top':
        starting_x = 0
        starting_y = gameMap.shape[1] // 2
        starting_coords = [(starting_x, starting_y + i) for i in range(numAgents)]
    elif starting_side == 'bottom':
        starting_x = gameMap.shape[0] - 1
        starting_y = gameMap.shape[1] // 2
        starting_coords = [(starting_x, starting_y + i) for i in range(numAgents)]
    elif starting_side == 'left':
        starting_x = gameMap.shape[0] // 2
        starting_y = 0
        starting_coords = [(starting_x + i, starting_y) for i in range(numAgents)]
    elif starting_side == 'right':
        starting_x = gameMap.shape[0] // 2
        starting_y = gameMap.shape[1] - 1
        starting_coords = [(starting_x + i, starting_y) for i in range(numAgents)]
    else:
        raise ValueError("Invalid starting side. Please choose 'top', 'bottom', 'left', or 'right'.")

    # Put agents
    for idx, coord in enumerate(starting_coords):
        if 0 <= coord[0] < gameMap.shape[0] and 0 <= coord[1] < gameMap.shape[1]:
            gameMap[coord[0], coord[1]] = agent_chars[idx]
        else:
            raise ValueError(f"Agent {idx} starting position "
                             f"{coord} is out of bounds for game map of size {gameMap.shape}")

    # Put walls
    gameMap = np.pad(gameMap, num_pad_pixels + 1, pad_with, padder='=')

    gameMap = [''.join(row.tolist()) for row in gameMap]
    return gameMap


class ObservationToArrayWithRGB(object):
    def __init__(self, colour_mapping):
        self._colour_mapping = colour_mapping
        # Rendering functions for the `board` representation and `RGB` values.
        self._renderers = {
            'RGB': ObservationToArray(value_mapping=colour_mapping)
        }

    def __call__(self, observation):
        # Perform observation rendering for agent and for video recording.
        result = {}
        for key, renderer in self._renderers.items():
            result[key] = renderer(observation)
        # Convert to [0, 255] RGB values.
        result['RGB'] = (result['RGB'] * (999.0 / 255.0)).astype(np.uint8)
        return result
