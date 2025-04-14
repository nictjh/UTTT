from utils import State, load_data, Action
import numpy as np
import time
import random
from typing import Optional, Tuple

class HuberLossAgent:
    def __init__(self, searchDepth: int = 6):
        """Instantiates your agent.
        """
        self.searchDepth = searchDepth
        self.timeLimit = 2.95
        self.startTime = None
        self.transpositionTable = {}

        self.weights = np.array([
            [
                7.45331571e-02,  6.63284659e-02, -4.30941060e-02,  9.44342539e-02,
                -9.35389921e-02,  2.43891716e-01, -2.33127862e-01, -1.22456532e-02,
                1.81713142e-02, -1.28765637e-02,  1.37973860e-01,  2.39836182e-02,
                -1.66359842e-02,  5.39807901e-02, -3.73200923e-02,  5.50543845e-01,
                1.73693746e-02,  4.05496284e-02,  5.15195318e-02,  9.81016755e-02,
                -1.02771714e-01,  5.53233549e-03, -1.69599766e-03,  9.63509083e-03,
                -5.59633225e-03, -8.06251287e-01, -1.43164000e-03,  1.01557858e-02,
                3.00204698e-02, -4.22811843e-02,  1.05814740e-01, -3.53328767e-03,
                9.38229356e-03,  9.35406052e-03,  1.98401641e-02,  1.09952542e-14,
                5.42764934e-41, -4.57274739e-11,  2.66423832e-39, -1.34671654e-17,
                -4.56991455e-40,  5.62269608e-40,  7.46223662e-40, -3.91734578e-03
            ]
        ], dtype=np.float32)

        self.bias = np.array([0.09446607], dtype=np.float32)

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
        features = extract_features6(state)
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
        #     return (1, 1, 1, 1)

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


def extract_features6(state: State) -> np.ndarray:
    local_status = state.local_board_status
    board = state.board
    features = []
    features.append(state.fill_num)

    validMoves = state.get_all_valid_actions()
    # Previous 8 features
    p1Wins = np.sum(local_status == 1)
    p2Wins = np.sum(local_status == 2)
    diffLocalWins = p1Wins - p2Wins
    features.append(np.sum(local_status == 1))  # P1 local wins
    features.append(np.sum(local_status == 2))  # P2 local wins
    features.append(1 if local_status[1][1] == 1 else 0)
    features.append(1 if local_status[1][1] == 2 else 0)
    features.append(countTwoInARow(local_status, 1))
    features.append(countTwoInARow(local_status, 2))

    p1_centers = 0
    p2_centers = 0
    for i in range(3):
        for j in range(3):
            if local_status[i][j] == 0:
                center = board[i][j][1][1]
                if center == 1:
                    p1_centers += 1
                elif center == 2:
                    p2_centers += 1
    features.append(p1_centers)
    features.append(p2_centers)

    # New features start here

    # 8. Empty local boards
    features.append(np.sum(local_status == 0))

    # 9. Forced board open (if you're forced into a finished board, you can move freely)
    features.append(hasFreedomMove(state.local_board_status, state.prev_local_action))

    # 10-11. Threats in local boards (2-in-a-rows)
    p1_threats = 0
    p2_threats = 0
    for i in range(3):
        for j in range(3):
            if local_status[i][j] == 0:
                sub_board = board[i][j]
                p1_threats += countTwoInARow(sub_board, 1)
                p2_threats += countTwoInARow(sub_board, 2)
    features.append(p1_threats)
    features.append(p2_threats)

    # 12-13. Meta board lines with exactly 1 win by each player
    features.append(metaLinePotential(local_status, player=1))
    features.append(metaLinePotential(local_status, player=2))

    boardFillRatio = np.count_nonzero(board) / 81.0
    features.append(boardFillRatio)

    # New: Game phase indicators
    is_early_game = 1.0 if boardFillRatio < 0.3 else 0.0
    is_mid_game = 1.0 if 0.3 <= boardFillRatio <= 0.7 else 0.0
    is_end_game = 1.0 if boardFillRatio > 0.7 else 0.0
    features.extend([is_early_game, is_mid_game, is_end_game])

    # One-move-away counts for both players
    oneMoveAwayP1 = countLocalBoardsOneMoveFromWin(board, state.local_board_status, player=1)
    oneMoveAwayP2 = countLocalBoardsOneMoveFromWin(board, state.local_board_status, player=2)
    features.append(oneMoveAwayP1)
    features.append(oneMoveAwayP2)

    # Nearly full local boards overall
    nearlyFull = countNearlyFullLocalBoards(board, state.local_board_status)
    features.append(nearlyFull)

    forced_empty, forced_threat_diff = analyzeForcedBoard(state)
    features.append(forced_empty)
    features.append(forced_threat_diff)

    features.append(state.fill_num * len(validMoves)) ## Strong feature
    features.append(state.fill_num * boardFillRatio)

    cornerCountP1 = countCornerTokens(board, state.local_board_status, 1)
    cornerCountP2 = countCornerTokens(board, state.local_board_status, 2)
    features.append(cornerCountP1)
    features.append(cornerCountP2)

    features.append(is_end_game * (p1_threats - p2_threats))
    features.append(is_mid_game * diffLocalWins)
    features.append(diffLocalWins)


    ##.2256 above
    totalMoves = np.count_nonzero(board)  # All tokens on board
    openingFlag = 1.0 if totalMoves < 5 else 0.0
    features.append(openingFlag)

    local_counts = []
    for i in range(3):
        for j in range(3):
            local_counts.append(np.count_nonzero(board[i][j]))
    local_balance = np.std(local_counts)
    features.append(local_balance)

    contested = 0
    for i in range(3):
        for j in range(3):
            if local_status[i][j] == 0:
                sub_board = board[i][j]
                if (np.count_nonzero(sub_board == 1) > 0 and np.count_nonzero(sub_board == 2) > 0):
                    contested += 1
    features.append(contested)

    volatility = 0.0
    count_open = 0
    for i in range(3):
        for j in range(3):
            if local_status[i][j] == 0:
                sub_board = board[i][j]
                diff = abs(np.count_nonzero(sub_board == 1) - np.count_nonzero(sub_board == 2))
                volatility += diff
                count_open += 1
    if count_open > 0:
        volatility /= count_open
    features.append(volatility)

    meta_line_crit = metaLineCriticalIndicators(board)  # local_status is the meta board
    features.extend(meta_line_crit)

    forced_quality = forcedBoardQuality(state)
    features.append(forced_quality)

    return np.array(features, dtype=np.float32)

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

def countNearlyFullLocalBoards(board: np.ndarray, localBoardStatus: np.ndarray) -> int:
    count = 0
    for big_r in range(3):
        for big_c in range(3):
            if localBoardStatus[big_r][big_c] != 0:
                continue
            sub_board = board[big_r][big_c]
            empties = np.count_nonzero(sub_board == 0)
            if empties <= 3:  # Board nearly full: 6 or more tokens placed
                count += 1
    return count

def analyzeForcedBoard(state: State) -> Tuple[int, int]:
    lastAction = state.prev_local_action
    if lastAction is None:
        return (0, 0)
    forced_r = lastAction[0] % 3
    forced_c = lastAction[1] % 3
    if state.local_board_status[forced_r][forced_c] != 0:
        return (0, 0)
    forced_board = state.board[forced_r][forced_c]
    empty_in_forced = np.count_nonzero(forced_board == 0)

    # Count two-in-a-row threats in forced board (simple version)
    threatP1 = countTwoInARow(forced_board, 1)
    threatP2 = countTwoInARow(forced_board, 2)
    threat_diff = threatP1 - threatP2
    return (empty_in_forced, threat_diff)

def is_board_one_move_away(sub_board: np.ndarray, player: int) -> bool:
    """
    Returns True if there's at least one row/column/diagonal in the 3x3 sub_board
    that contains exactly 2 'player' tokens and 1 empty cell
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

def countCornerTokens(board: np.ndarray, localBoardStatus: np.ndarray, player: int) -> int:
    count = 0
    corners = [(0,0), (0,2), (2,0), (2,2)]
    for big_r in range(3):
        for big_c in range(3):
            if localBoardStatus[big_r][big_c] != 0:
                continue
            sub_board = board[big_r][big_c]
            for (i, j) in corners:
                if sub_board[i, j] == player:
                    count += 1
    return count


def metaLineCriticalIndicators(meta: np.ndarray) -> list[int]:
    """
    For each meta-line (row, column, or diagonal) in the 3x3 meta board (local_board_status),
    returns 1 if the line is "critical" (i.e., either player has exactly 2 wins and 1 open),
    otherwise 0. Returns a list of 8 binary indicators (3 rows, 3 columns, 2 diagonals).
    """
    crit = []
    # Rows
    for i in range(3):
        row = meta[i, :]
        if ((np.count_nonzero(row == 1) == 2 and np.count_nonzero(row == 0) == 1) or
            (np.count_nonzero(row == 2) == 2 and np.count_nonzero(row == 0) == 1)):
            crit.append(1)
        else:
            crit.append(0)
    # Columns
    for j in range(3):
        col = meta[:, j]
        if ((np.count_nonzero(col == 1) == 2 and np.count_nonzero(col == 0) == 1) or
            (np.count_nonzero(col == 2) == 2 and np.count_nonzero(col == 0) == 1)):
            crit.append(1)
        else:
            crit.append(0)
    # Diagonals
    diag1 = np.array([meta[0,0], meta[1,1], meta[2,2]])
    if ((np.count_nonzero(diag1 == 1) == 2 and np.count_nonzero(diag1 == 0) == 1) or
        (np.count_nonzero(diag1 == 2) == 2 and np.count_nonzero(diag1 == 0) == 1)):
        crit.append(1)
    else:
        crit.append(0)
    diag2 = np.array([meta[0,2], meta[1,1], meta[2,0]])
    if ((np.count_nonzero(diag2 == 1) == 2 and np.count_nonzero(diag2 == 0) == 1) or
        (np.count_nonzero(diag2 == 2) == 2 and np.count_nonzero(diag2 == 0) == 1)):
        crit.append(1)
    else:
        crit.append(0)
    return crit

def forcedBoardQuality(state: State) -> int:
    """
    Returns 1 if the forced board (determined by state.prev_local_action) is open and
    has 3 or fewer empty cells (i.e. likely to be decided quickly), otherwise 0.
    """
    if state.prev_local_action is None:
        return 0
    forced_r = state.prev_local_action[0] % 3
    forced_c = state.prev_local_action[1] % 3
    # Check if the forced board is open
    if state.local_board_status[forced_r][forced_c] != 0:
        return 0
    forced_board = state.board[forced_r][forced_c]
    empties = np.count_nonzero(forced_board == 0)
    return 1 if empties <= 3 else 0
