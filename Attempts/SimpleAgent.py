from utils import State, load_data, Action
import numpy as np
import time
import random
from typing import Optional, Tuple

class SimplestAgent:
    def __init__(self, searchDepth: int = 6):
        """Instantiates your agent.
        """
        self.searchDepth = searchDepth
        self.timeLimit = 2.95
        self.startTime = None
        self.transpositionTable = {}

        self.weights = np.array([
            [
                6.5757811e-02, -6.8906374e-02,  6.9117315e-02, -6.1823167e-02,
                2.5279328e-01, -2.4969099e-01, -9.8872995e-03,  6.0097682e-03,
                1.4365486e-04, -4.7026817e-03,  5.5049933e-02, -5.4368228e-02,
                6.2644958e-02, -6.0675163e-02
            ]
        ], dtype=np.float32)

        self.bias = np.array([0.00096601], dtype=np.float32)



    def negamax(self, state: 'State', depth: int, alpha: float = 0.0, beta: float = 1.0) -> Tuple[float, Optional['Action']]:
        """
        Negamax-like search in [0..1] range.
        Returns (value, bestAction) from the perspective of the current player.
        """
        # print(depth)
        ## Timeout check
        if time.time() - self.startTime >= self.timeLimit:
            raise TimeoutError("Time limit exceeded in negamax")

        ## Check cached values first
        stateKey = self.getStateHash(state)
        if stateKey in self.transpositionTable:
            cachedValue, cachedDepth = self.transpositionTable[stateKey]
            ## If we have a cached value at >= depth, we can use it
            if cachedDepth >= depth:
                return (cachedValue, None)

        ## Check Terminal / depth limit
        if state.is_terminal():
            ## Return the 0..1 terminal utility from the CURRENT player's perspective
            terminalValue = state.terminal_utility()  # must be in [0..1]
            self.transpositionTable[stateKey] = (terminalValue, depth)
            return (terminalValue, None)

        if depth == 0:
            ## Just evaluate and return
            val = self.evaluateState(state)
            self.transpositionTable[stateKey] = (val, depth)
            return (val, None)

        bestValue = -float('inf')
        bestAction = None
        actions = state.get_all_valid_actions()

        np.random.shuffle(actions)

        ## This evaluates the remaining moves
        for action in actions:
            val = self._evaluate_single_move(state, action, depth, alpha, beta)

            if val > bestValue:
                bestValue = val
                bestAction = action

            alpha = max(alpha, bestValue)

            if alpha >= beta:
                break

        self.transpositionTable[stateKey] = (bestValue, depth)

        return (bestValue, bestAction)

    def _evaluate_single_move(self, state: 'State', action: 'Action', depth: int, alpha: float, beta: float) -> float:
        """Helper to evaluate a single move in negamax."""
        childState = state.change_state(action)
        childValue, _ = self.negamax(childState.invert(), depth - 1, 1.0 - beta, 1.0 - alpha)
        val = 1.0 - childValue
        return val

    def getStateHash(self, state: State) -> tuple:
        board_flat = tuple(state.board.flatten())
        meta_flat = tuple(state.local_board_status.flatten())
        return (board_flat, meta_flat, state.fill_num)

    def evaluateState(self, state: State) -> float:
        """
        Evaluates the state using the formula
        """
        features = extract_features5(state)
        evaluatedValue = np.dot(self.weights, features) + self.bias
        ## Sigmoid Function
        value = 1 / (1 + np.exp(-evaluatedValue))
        value = np.clip(value, 0.001, 0.999)
        return value.item()

    def choose_action(self, state: State) -> Action:
        """Returns a valid action to be played on the board.
        Assuming that you are filling in the board with number 1.

        Parameters
        ---------------
        state: The board to make a move on.
        """
        self.startTime = time.time()

        if np.count_nonzero(state.board) == 0 and state.fill_num == 1:
            return (1, 1, 1, 1)

        bestAction = state.get_random_valid_action()

        for d in range(1, self.searchDepth + 1):
            # self.transpositionTable.clear()
            try:
                val, action = self.negamax(state, d, alpha=0.0, beta=1.0)
                if action is not None:
                    bestAction = action
            except TimeoutError:
                break

        return bestAction





def extract_features5(state: State) -> np.ndarray:
    local_status = state.local_board_status
    board = state.board
    features = []

    features.append(np.sum(local_status == 1))
    features.append(np.sum(local_status == 2))
    features.append(np.sum(local_status == 0))
    features.append(1 if local_status[1][1] == 1 else 0)
    features.append(1 if local_status[1][1] == 2 else 0)
    features.append(countTwoInARow(local_status, 1))
    features.append(countTwoInARow(local_status, 2))
    features.append(metaLinePotential(local_status, player=1))
    features.append(metaLinePotential(local_status, player=2))

    features.append(countCenterCells(state.board, state.local_board_status, 1))
    features.append(countCenterCells(state.board, state.local_board_status, 2))
    features.append(hasFreedomMove(state.local_board_status, state.prev_local_action))
    features.append(countTotalLocalThreats(state.board, state.local_board_status, 1))
    features.append(countTotalLocalThreats(state.board, state.local_board_status, 2))

    return np.array(features, dtype=np.float32)

##### Helper functions #####

def countLine(line: np.ndarray, player: int) -> int:
    """
    Counts the number of player tokens in a line (row, column, or diagonal).
    """
    return int(np.count_nonzero(line == player) == 2 and np.count_nonzero(line == 0) == 1)

def countTwoInARow(board: np.ndarray, player: int) -> int:
    """
    Checks for how many two in a row pattern each player has on still-active local boards
    Account for diagonals too
    """
    counter = 0
    for i in range(3):
        counter += countLine(board[i, :], player)
        counter += countLine(board[:, i], player)

    firstDiagonal = np.zeros(3, dtype=int)
    secDiagonal = np.zeros(3, dtype=int)
    for i in range(3):
        firstDiagonal[i] = board[i, i]
        secDiagonal[i] = board[i, 2 - i]

    counter += countLine(firstDiagonal, player)
    counter += countLine(secDiagonal, player)

    return counter

def metaLinePotential(meta: np.ndarray, player: int) -> int:
    counter = 0

    ## Check rows and columns
    for i in range(3):
        row = meta[i, :]
        col = meta[:, i]
        if np.count_nonzero(row == player) == 1 and np.count_nonzero(row == 0) == 2:
            counter += 1
        if np.count_nonzero(col == player) == 1 and np.count_nonzero(col == 0) == 2:
            counter += 1

    ## Check diagonals
    firstDiagonal = [0, 0, 0]
    secDiagonal = [0, 0, 0]
    for i in range(3):
        firstDiagonal[i] = meta[i, i]
        secDiagonal[i] = meta[i, 2 - i]

    if firstDiagonal.count(player) == 1 and firstDiagonal.count(0) == 2:
        counter += 1
    if secDiagonal.count(player) == 1 and secDiagonal.count(0) == 2:
        counter += 1

    return counter

def emptyCellCount(board: np.ndarray) -> int:
    return np.count_nonzero(board == 0)


def hasFreedomMove(localStatus: np.ndarray, lastMove: tuple) -> int:
    if lastMove is None:
        return 1  # first move
    lr, lc = lastMove
    return int(localStatus[lr][lc] != 0)

def forcedBoardOneHot(lastAction: tuple) -> list:
    """
    Returns a one-hot vector of length 9 indicating which local board
    the next move is forced to (if not closed).
    If lastAction is None (start of game), returns all zeros.
    """
    arr = [0] * 9
    if lastAction is not None:
        r, c = lastAction
        index = r * 3 + c
        arr[index] = 1
    return arr

def countCenterCells(board: np.ndarray, localBoardStatus: np.ndarray, player: int) -> int:
    """
    Counts how many open local boards have the center cell belonging to player
    """
    counter = 0
    for i in range(3):
        for j in range(3):
            if localBoardStatus[i][j] == 0:  # board must be open
                localBoardCenter = board[i][j][1, 1]
                if localBoardCenter == player:
                    counter += 1
    return counter


def countTotalLocalThreats(board: np.ndarray, localBoardStatus: np.ndarray, player: int) -> int:
    """
    Sums the two-in-a-row threats for player across all open local boards.
    Uses 'countTwoInARow' on each open local board
    """
    totalThreats = 0
    for i in range(3):
        for j in range(3):
            if localBoardStatus[i][j] == 0:  # open local board
                localBoard = board[i][j]
                totalThreats += countTwoInARow(localBoard, player)
    return totalThreats

def countLocalBoardsOneMoveFromWin(board, localBoardStatus, player):
    count = 0
    for big_r in range(3):
        for big_c in range(3):
            ## skip closed boards
            if localBoardStatus[big_r][big_c] != 0:
                continue
            sub_board = board[big_r][big_c]
            if is_board_one_move_away(sub_board, player):
                count += 1
    return count


def is_board_one_move_away(sub_board: np.ndarray, player: int) -> bool:
    """
    Returns True if there's at least one row/column/diagonal in the 3x3 sub_board
    that contains exactly 2 player tokens and 1 empty cell
    Otherwise, returns False.
    """
    ## Check rows and columns
    for r in range(3):
        row = sub_board[r, :]
        if np.count_nonzero(row == player) == 2 and np.count_nonzero(row == 0) == 1:
            return True

    for c in range(3):
        col = sub_board[:, c]
        if np.count_nonzero(col == player) == 2 and np.count_nonzero(col == 0) == 1:
            return True

    ## Check diagonals
    firstDiag = sub_board.diagonal()
    secDiag = np.array([sub_board[0, 2], sub_board[1, 1], sub_board[2, 0]])

    if (np.count_nonzero(firstDiag == player) == 2 and np.count_nonzero(firstDiag == 0) == 1):
        return True
    if (np.count_nonzero(secDiag == player) == 2 and np.count_nonzero(secDiag == 0) == 1):
        return True

    return False



# state = State(
#     board=np.array([
#         [
#             [[1, 0, 2], [0, 1, 0], [0, 0, 1]],
#             [[2, 0, 0], [0, 0, 0], [0, 0, 0]],
#             [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
#         ],
#         [
#             [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
#             [[2, 0, 0], [0, 0, 0], [0, 0, 0]],
#             [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
#         ],
#         [
#             [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
#             [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
#             [[2, 0, 0], [0, 0, 0], [0, 0, 0]],
#         ],
#     ]),
#     fill_num=1,
#     prev_action=(2, 2, 0, 0)
# )
# start_time = time.time()
# student_agent = SimplestAgent()
# constructor_time = time.time()
# action = student_agent.choose_action(state)
# end_time = time.time()
# assert state.is_valid_action(action)
# print(f"Constructor time: {constructor_time - start_time}")
# print(f"Action time: {end_time - constructor_time}")
# assert constructor_time - start_time < 1
# assert end_time - constructor_time < 3