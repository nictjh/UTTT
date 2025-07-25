from utils import State, load_data, Action
import numpy as np
import time
import random
from typing import Optional, Tuple

class StudentAgent:
    def __init__(self, searchDepth: int = 5):
        """Instantiates your agent.
        """
        self.searchDepth = searchDepth
        self.timeLimit = 2.97
        self.startTime = None
        self.transpositionTable = {}
        ## From training (hardcoded for now)
        self.weights = np.array([
            [
                2.63846181e-02, -4.84127726e-04,  1.72024578e-01, -2.03120902e-01,
                -2.78002173e-02,  1.39335498e-01, -2.30715387e-02,  2.11378157e-01,
                -2.22294822e-01, -7.20849708e-02,  3.45746689e-02, -1.23919314e-02,
                2.26355158e-02, -4.54940367e-03, -3.48353549e-03, -1.13967806e-02,
                -3.11757158e-03, -1.52657665e-02, -2.34105391e-03, -1.70872193e-02,
                -9.27404314e-03, -8.40950012e-03, -9.45400074e-02,  8.66745561e-02,
                -3.67267057e-02,  1.83778778e-01, -3.65473703e-02,  2.97409035e-02,
                1.05117917e-01,  4.97692823e-02,  5.86513989e-03, -1.24060281e-03,
                9.45399422e-03,  3.98962945e-03, -1.04771694e-02, -5.71270986e-03,
                7.00664893e-03,  1.80023238e-02, -1.69641193e-04, -2.57642940e-02,
                3.23661952e-03, -2.09874772e-02, -8.20159018e-01
            ]
        ], dtype=np.float32)
        self.bias = np.array([-0.02859862], dtype=np.float32)

    def negamax(self, state: 'State', depth: int, alpha: float = 0.0, beta: float = 1.0) -> Tuple[float, Optional['Action']]:
        """
        Negamax-like search in [0..1] range.
        Returns (value, bestAction) from the perspective of the current player.
        """
        ## Timeout check
        if time.time() - self.startTime >= self.timeLimit:
            raise TimeoutError("Time limit exceeded in negamax")
        ## Check cached values first
        stateKey = self.getStateHash(state)
        if stateKey in self.transpositionTable:
            cachedValue, cachedDepth, cachedBestMove = self.transpositionTable[stateKey]
            ## If we have a cached value at >= depth, we can use it
            if cachedDepth >= depth:
                return (cachedValue, None)
        else:
            cachedBestMove = None
        ## Check Terminal / depth limit
        if state.is_terminal():
            ## Return the 0..1 terminal utility from the CURRENT player's perspective
            terminalValue = state.terminal_utility()  # must be in [0..1]
            self.transpositionTable[stateKey] = (terminalValue, depth, None)
            return (terminalValue, None)
        if depth == 0:
            ## Just evaluate and return
            val = self.evaluateState(state)
            self.transpositionTable[stateKey] = (val, depth, None)
            return (val, None)
        bestValue = -float('inf')
        bestAction = None
        actions = state.get_all_valid_actions()

        if cachedBestMove and cachedBestMove in actions:
            val = self._evaluate_single_move(state, cachedBestMove, depth, alpha, beta)
            if val > bestValue:
                bestValue = val
                bestAction = cachedBestMove
            alpha = max(alpha, bestValue)
            if alpha >= beta:
                # We got a cutoff using the TT best move alone
                self.transpositionTable[stateKey] = (bestValue, depth, bestAction)
                return (bestValue, bestAction)
            actions.remove(cachedBestMove) ## Dont evaluate again
        np.random.shuffle(actions)
        for action in actions:
            val = self._evaluate_single_move(state, action, depth, alpha, beta)
            if val > bestValue:
                bestValue = val
                bestAction = action
            alpha = max(alpha, bestValue)
            if alpha >= beta:
                break
        self.transpositionTable[stateKey] = (bestValue, depth, bestAction)
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
        features = extract_features(state)
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
        # if np.count_nonzero(state.board) == 0 and state.fill_num == 1:
        #     return Action(1, 1, 1, 1)

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

###### Feature Extraction Functions ######
def extract_features(state: State) -> np.ndarray:
    """
    Extract features from the state for training.
    Total Feature list:
    1. Player token count
    2. Opponent token count
    3. Player local wins
    4. Opponent local wins
    5. Draw local board count
    6. Difference in local wins
    7. Unclaimed boards
    8. Meta board center control to player
    9. Meta board center control to opponent
    10. Meta line potential for player
    11. Meta line potential for opponent
    12. Center board control in local board to player
    13. Center board control in local board to opponent
    14. One-hot encoding of forced board
    23. Count total local threats for player
    24. Count total local threats for opponent
    25. Difference in local threats
    26. Player turn
    27. Whether the last move was forced
    28. Number of valid moves
    29. Board fill ratio
    """
    features = []
    ## Constant definitions
    board = state.board
    localBoardStatus = state.local_board_status
    player = state.fill_num
    opponent = 3 - player
    lastAction = state.prev_local_action
    ## Token Counts
    playerCount = np.sum(board == 1)
    opponentCount = np.sum(board == 2)
    features.append(playerCount)
    features.append(opponentCount)
    ## p1, p2 local wins and draw count
    p1LocalWins = np.sum(localBoardStatus == 1)
    p2LocalWins = np.sum(localBoardStatus == 2)
    drawLbsCount = np.sum(localBoardStatus == 3)
    features.extend([p1LocalWins, p2LocalWins, drawLbsCount])
    ## Difference in local wins
    diffLocalWins = p1LocalWins - p2LocalWins
    features.append(diffLocalWins)
    ## Unclaimed boards
    emptyBoards = np.sum(localBoardStatus == 0)
    features.append(emptyBoards)
    ## meta board center control
    features.append(1 if localBoardStatus[1][1] == 1 else 0)
    features.append(1 if localBoardStatus[1][1] == 2 else 0)
    ## Meta Line potential
    metaThreatsP1 = metaLinePotential(localBoardStatus, 1)
    metaThreatsP2 = metaLinePotential(localBoardStatus, 2)
    features.extend([metaThreatsP1, metaThreatsP2])
    ## Center board control in local board
    centerCountP1 = countCenterCells(board, localBoardStatus, 1)
    centerCountP2 = countCenterCells(board, localBoardStatus, 2)
    features.extend([centerCountP1, centerCountP2])
    ## One-hot encoding of forced board if there is
    forced_board_9d = forcedBoardOneHot(lastAction)
    features.extend(forced_board_9d)
    ## Count total local threats across all open local boards
    totalLocalThreatsP1 = countTotalLocalThreats(board, localBoardStatus, 1)
    totalLocalThreatsP2 = countTotalLocalThreats(board, localBoardStatus, 2)
    features.extend([totalLocalThreatsP1, totalLocalThreatsP2])
    # Difference in local threats
    diffTotalLocalThreats = totalLocalThreatsP1 - totalLocalThreatsP2
    features.append(diffTotalLocalThreats)
    ## Player turn
    features.append(player)
    ## whether the last move was forced
    freedomMove = hasFreedomMove(localBoardStatus, lastAction)
    features.append(freedomMove)
    ## Number of valid moves
    valid_moves = state.get_all_valid_actions()
    features.append(len(valid_moves))
    ## Board fill ratio
    boardFillRatio = np.count_nonzero(board) / 81.0
    features.append(boardFillRatio)
    ## Expanded local features
    features.append(p1LocalWins * diffTotalLocalThreats)
    features.append(boardFillRatio * len(valid_moves))
    features.append((metaThreatsP1 - metaThreatsP2) * (totalLocalThreatsP1 - totalLocalThreatsP2))
    features.append(metaThreatsP1 * totalLocalThreatsP1)
    # features.append(boardFillRatio * p1LocalWins) ## Got worse...
    features.append(centerCountP1 * metaThreatsP1)
    features.append(centerCountP2 * metaThreatsP2) ## no difference at all ##Add back later
    features.append(p1LocalWins * metaThreatsP1)
    features.append(p2LocalWins * metaThreatsP2) ##Add back later
    ## Above this is already 0.2526 best
    features.append(emptyBoards * diffTotalLocalThreats)
    features.append(len(valid_moves) * diffTotalLocalThreats)
    ## Above best is 0.2524
    features.append(diffLocalWins * diffTotalLocalThreats)
    features.append(diffLocalWins * (metaThreatsP1 - metaThreatsP2))
    features.append(player * len(valid_moves)) ## Strong feature
    features.append(player * boardFillRatio)
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
    Counts how many open local boards have the center cell belonging to 'player_id'.
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
    Sums the two-in-a-row threats for 'player_id' across all open local boards.
    Uses 'countTwoInARow' on each open local board
    """
    totalThreats = 0
    for i in range(3):
        for j in range(3):
            if localBoardStatus[i][j] == 0:  # open local board
                localBoard = board[i][j]
                totalThreats += countTwoInARow(localBoard, player)
    return totalThreats



state = State(
    board=np.array([
        [
            [[1, 0, 2], [0, 1, 0], [0, 0, 1]],
            [[2, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
        ],
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[2, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ],
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[2, 0, 0], [0, 0, 0], [0, 0, 0]],
        ],
    ]),
    fill_num=1,
    prev_action=(2, 2, 0, 0)
)
start_time = time.time()
student_agent = StudentAgent()
constructor_time = time.time()
action = student_agent.choose_action(state)
end_time = time.time()
assert state.is_valid_action(action)
print(f"Constructor time: {constructor_time - start_time}")
print(f"Action time: {end_time - constructor_time}")
assert constructor_time - start_time < 1
assert end_time - constructor_time < 3