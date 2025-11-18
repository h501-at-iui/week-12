import numpy as np
from IPython.display import clear_output
import time
import seaborn as sns
import matplotlib.pyplot as plt


def update_board(board):
    """
    Executes one step of Conway's Game of Life.
    
    Args:
        board (np.array): A 2D NumPy array representing the game board
                          (1 for live, 0 for dead).
                          
    Returns:
        np.array: The game board after one generation.
    """
    
    # Create a new board to store the next state.
    # We can't modify the original board in-place, as changes
    # would affect the neighbor counts for subsequent cells.
    new_board = np.zeros_like(board)
    
    rows, cols = board.shape
    
    # Iterate through every cell on the board
    for i in range(rows):
        for j in range(cols):
            
            # --- Count live neighbors ---
            # This implementation uses a "toroidal" or "wrapping"
            # boundary, where the top/bottom and left/right
            # edges are connected.
            
            live_neighbors = 0
            # Iterate over the 3x3 grid centered at (i, j)
            for x in range(i - 1, i + 2):
                for y in range(j - 1, j + 2):
                    # Don't count the cell itself
                    if x == i and y == j:
                        continue
                    
                    # Use the modulo operator (%) to wrap around the edges
                    neighbor_row = x % rows
                    neighbor_col = y % cols
                    
                    live_neighbors += board[neighbor_row, neighbor_col]

            # --- Apply Conway's Rules ---
            current_cell_state = board[i, j]
            
            # Rule 1 & 3: A live cell dies (underpopulation or overpopulation)
            if current_cell_state == 1 and (live_neighbors < 2 or live_neighbors > 3):
                new_board[i, j] = 0
                
            # Rule 2: A live cell with 2 or 3 neighbors survives
            elif current_cell_state == 1 and (live_neighbors == 2 or live_neighbors == 3):
                new_board[i, j] = 1
                
            # Rule 4: A dead cell with exactly 3 neighbors becomes live (reproduction)
            elif current_cell_state == 0 and live_neighbors == 3:
                new_board[i, j] = 1
                
            # All other dead cells stay dead (which is the default 
            # value of new_board, so no 'else' is needed)

    return new_board


def show_game(game_board, n_steps=10, pause=0.5):
    """
    Show `n_steps` of Conway's Game of Life, given the `update_board` function.

    Parameters
    ----------
    game_board : numpy.ndarray
        A binary array representing the initial starting conditions for Conway's Game of Life. In this array, ` represents a "living" cell and 0 represents a "dead" cell.
    n_steps : int, optional
        Number of game steps to run through, by default 10
    pause : float, optional
        Number of seconds to wait between steps, by default 0.5
    """
    for step in range(n_steps):
        clear_output(wait=True)

        # update board
        game_board = update_board(game_board)

        # show board
        sns.heatmap(game_board, cmap='plasma', cbar=False, square=True)
        plt.title(f'Board State at Step {step + 1}')
        plt.show()

        # wait for the next step
        if step + 1 < n_steps:
            time.sleep(pause)

def _run_to_stable(board):
    """
    A recursive helper function that steps the game until it stabilizes.
    
    Base Case: The new board is identical to the old board.
    Recursive Step: Call this function with the new board.
    """
    new_board = update_board(board)
    
    # Base Case: If the board has stopped changing (stabilized).
    if np.array_equal(board, new_board):
        return new_board
    
    # Recursive Step: Run the game for another step.
    return _run_to_stable(new_board)

def play_recursive_game():
    """
    Plays Conway's Game of Life recursively.
    
    This function initializes a random 10x10 board and
    calls a recursive helper to run the game until it
    reaches a stable state.
    
    Returns:
        np.array: The final, stable board configuration.
    """
    initial_board = np.random.randint(2, size=(10, 10))
    
    # Start the recursive process
    final_board = _run_to_stable(initial_board)
    
    return final_board