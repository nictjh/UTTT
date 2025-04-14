## First Attempt
## Trialed with a very comprehensive evaluation function ##
## Got the inspiration from a youtube video ##
## Attempt didn't work for the project because it was too slow ##

import numpy as np
from dataclasses import dataclass
import pickle
import time


Action = tuple[int, int, int, int]
LocalBoardAction = tuple[int, int]


def convert_board_to_string(board):
    output = []
    horizontal_separator = "-" * 21

    for super_row in range(3):
        for sub_row in range(3):
            line_parts = []
            for super_col in range(3):
                sub_board = board[super_row][super_col]
                sub_line = " ".join(str(sub_board[sub_row][sub_col]) for sub_col in range(3))
                line_parts.append(sub_line)
            full_line = " | ".join(line_parts)
            output.append(full_line)
        if super_row != 2:
            output.append(horizontal_separator)
    
    return '\n'.join(output)

ENDLINE = '\n'


@dataclass(frozen=True)
class ImmutableState:
    board: np.ndarray
    prev_local_action: LocalBoardAction | None
    fill_num: 1 | 2
    local_board_status: np.ndarray = None

    def __post_init__(self):
        object.__setattr__(self, 'local_board_status', get_local_board_status(self.board))

    def __eq__(self, other):
        return np.all(self.board == other.board) and self.prev_local_action == other.prev_local_action and self.fill_num == other.fill_num

    def __repr__(self):
        return f"""State(
    board=
        {convert_board_to_string(self.board).replace(ENDLINE, ENDLINE+'        ')}, 
    local_board_status=
        {str(self.local_board_status).replace(ENDLINE, ENDLINE+'        ')}, 
    prev_local_action={self.prev_local_action}, 
    fill_num={self.fill_num}
)
"""


def get_local_board_action(action: Action) -> LocalBoardAction:
    meta_row, meta_col, local_row, local_col = action
    return LocalBoardAction((local_row, local_col))

## Checks whether its a win / loss / ongoing / tie
def board_status(board: np.ndarray) -> int:
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != 0:
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != 0:
            return board[0][i]
    if board[0][0] == board[1][1] == board[2][2] != 0:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != 0:
        return board[0][2]
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                return 0
    return 3


def get_local_board_status(board: np.ndarray) -> None:
    # print(f"DEBUG: board.shape = {board.shape}")
    local_board_status: np.ndarray = np.array([[0 for i in range(3)] for j in range(3)])
    for i in range(3):
        for j in range(3):
            local_board_status[i][j] = board_status(board[i][j])
    return local_board_status


def is_valid_action(state: ImmutableState, action: Action) -> bool:
    if not isinstance(action, tuple):
        return False
    if len(action) != 4:
        return False
    i, j, k, l = action
    if type(i) != int or type(j) != int or type(k) != int or type(l) != int:
        return False
    if state.local_board_status[i][j] != 0:
        return False
    if state.board[i][j][k][l] != 0:
        return False
    if state.prev_local_action is None:
        return True
    prev_row, prev_col = state.prev_local_action
    if prev_row == i and prev_col == j:
        return True
    return state.local_board_status[prev_row][prev_col] != 0


def _get_all_valid_free_actions(state: ImmutableState) -> list[Action]:
    valid_actions: list[Action] = []
    for i in range(3):
        for j in range(3):
            if state.local_board_status[i][j] != 0:
                continue
            for k in range(3):
                for l in range(3):
                    if state.board[i][j][k][l] == 0:
                        valid_actions.append((i, j, k, l))
    return valid_actions


def get_all_valid_actions(state: ImmutableState) -> list[Action]:
    if state.prev_local_action is None:
        return _get_all_valid_free_actions(state)
    prev_row, prev_col = state.prev_local_action
    if state.local_board_status[prev_row][prev_col] != 0:
        return _get_all_valid_free_actions(state)
    valid_actions: list[Action] = []
    for i in range(3):
        for j in range(3):
            if state.board[prev_row][prev_col][i][j] == 0:
                valid_actions.append((prev_row, prev_col, i, j))
    return valid_actions


def get_next_turn_fill_num(fill_num):
    return 3 - fill_num


def get_random_valid_action(state: ImmutableState) -> Action:
    valid_actions = get_all_valid_actions(state)
    return valid_actions[np.random.randint(len(valid_actions))]


def change_state(state: ImmutableState, action: Action, check_valid_action = True) -> ImmutableState:
    if check_valid_action:
        assert is_valid_action(state, action), f"Invalid action: {action}"
    i, j, k, l = action
    new_board = state.board.copy()
    new_board[i][j][k][l] = state.fill_num
    new_state = ImmutableState(board=new_board, fill_num=get_next_turn_fill_num(state.fill_num), prev_local_action=get_local_board_action(action))
    return new_state


def is_terminal(state: ImmutableState) -> bool:
    return board_status(state.local_board_status) != 0


def terminal_utility(state: ImmutableState) -> float:
    status = board_status(state.local_board_status)
    if status == 1:
        return 1.0
    if status == 2:
        return 0.0
    if status == 3:
        return 0.5
    assert False, "Board is not terminal"


def invert(state: ImmutableState) -> ImmutableState:
    board = np.zeros_like(state.board)
    board[state.board == 1] = 2
    board[state.board == 2] = 1
    return ImmutableState(
        board = board,
        prev_local_action = state.prev_local_action,
        fill_num = get_next_turn_fill_num(state.fill_num)
    )


class State: # Wrapper for ImmutableState
    def __init__(self,
                 board: np.ndarray = None,
                 fill_num: 1 | 2 = 1,
                 prev_action: Action | None = None,
                 prev_local_action: LocalBoardAction | None = None,
                 local_board_status: np.ndarray = None, # deprecated: only for backward compatibility
                 ):
        if board is None:
            board = np.array([[[[0 for i in range(3)]for j in range(3)] for k in range(3)] for l in range(3)])
        if prev_local_action is None and prev_action is not None:
            prev_local_action = get_local_board_action(prev_action)
        if local_board_status is not None:
            print("Warning: The use of local_board_status is deprecated and will be ignored.")
        self._state = ImmutableState(board=board, fill_num=fill_num, prev_local_action=prev_local_action)

    def __eq__(self, other):
        return self._state == other._state

    def __repr__(self):
        return self._state.__repr__()

    @property
    def board(self) -> np.array:
        return self._state.board

    @property
    def fill_num(self) -> 1 | 2:
        return self._state.fill_num

    @property
    def local_board_status(self) -> np.array:
        return self._state.local_board_status

    @property
    def prev_local_action(self) -> np.array:
        return self._state.prev_local_action

    def update_local_board_status(self) -> None:
        pass # Does nothing, only for backward compatibility

    def get_backward_compatible_state(self, prev_action: Action | None = None) -> ImmutableState:
        state = self._state
        if prev_action is not None and get_local_board_action(prev_action) != self._state.prev_local_action:
            print("Warning: The prev_action you specified contains a local action that differs from the one in the current state.")
            state = ImmutableState(board=self._state.board, fill_num=self._state.fill_num,
                                prev_local_action=get_local_board_action(prev_action))
        return state

    def is_valid_action(self, action: Action, prev_action: Action | None = None) -> bool:
        return is_valid_action(self.get_backward_compatible_state(prev_action), action)

    def get_all_valid_actions(self, prev_action: Action | None = None) -> list[Action]:
        return get_all_valid_actions(self.get_backward_compatible_state(prev_action))

    def _get_all_valid_free_actions(self) -> list[Action]:
        return _get_all_valid_free_actions(self._state)

    def get_random_valid_action(self, prev_action: Action | None = None) -> Action:
        return get_random_valid_action(self.get_backward_compatible_state(prev_action))

    def change_state(self, action: Action, prev_action: Action | None = None, in_place: bool = False, check_valid_action = True) -> "State":
        if in_place: # break backward compatibility
            raise NotImplementedError
        new_state = change_state(self.get_backward_compatible_state(prev_action), action)
        return State(board=new_state.board, fill_num=new_state.fill_num, prev_local_action=new_state.prev_local_action)

    def is_terminal(self) -> bool:
        return is_terminal(self._state)

    def terminal_utility(self) -> float:
        return terminal_utility(self._state)

    def invert(self) -> "State":
        new_state = invert(self._state)
        return State(board=new_state.board, fill_num=new_state.fill_num, prev_local_action=new_state.prev_local_action)

    def clone(self) -> "State":
        return State(
            board=self._state.board.copy(),
            fill_num=self._state.fill_num,
            prev_local_action=self._state.prev_local_action
        )


def load_data() -> list[tuple[State, float]]:
    with open("data.pkl", "rb") as f:
        data = pickle.load(f)
    new_data = []
    for row in data:
        row_data, utility = row
        board, fill_num, prev_local_action = row_data
        state = State(board=board, fill_num=fill_num, prev_local_action=prev_local_action)
        new_data.append((state, utility))
    return new_data

##### Student Agent ######

class StudentAgent:
    def __init__(self):
        """Instantiates your agent.
        """
        # self.maxDepth = 2 ## Trialed 5 and it was too long
        self.player = 1 ## Assumed to be 1 always

    @staticmethod
    def evaluateLocalBoard(board: np.ndarray, player: int) -> float:
        """
        Evaluates the local board for the player.
        Give positional weight to the center, corners and edges.
        Evaluate the lines to give a more accurate score for local evaluation


        Parameters
        ---------------
        board: The local board to evaluate.
        player: The player to evaluate for.
        """

        # if not isinstance(board, np.ndarray) or board.shape != (3, 3):
        #     raise ValueError(f"❌ ERROR: Expected 3×3 board, but got shape: {getattr(board, 'shape', 'no shape')} and value:\n{board}")


        if (player == 1):
            opponent = 2
        else:
            opponent = 1

        evalScore = 0.0
        ## According to online sources, taking center > corners > edges
        positionalWeights = np.array([
            [3, 2, 3],
            [2, 4, 2],
            [3, 2, 3]
        ])

        # print(f"DEBUG LOCAL BOARD: board.shape = {board.shape}, board =\n{board}")

        ## Loops through the board and add up positional weights
        for i in range(3):
            for j in range(3):
                if board[i][j] == player:
                    evalScore += positionalWeights[i][j]
                elif board[i][j] == opponent:
                    evalScore -= positionalWeights[i][j]

        ## Potential winning combinations lines
        lines = [
            ## Straight row combos
            [(0,0), (0,1), (0,2)],
            [(1,0), (1,1), (1,2)],
            [(2,0), (2,1), (2,2)],
            ## Straight column combos
            [(0,0), (1,0), (2,0)],
            [(0,1), (1,1), (2,1)],
            [(0,2), (1,2), (2,2)],
            ## Diagonal combos
            [(0,0), (1,1), (2,2)],
            [(0,2), (1,1), (2,0)]
        ]

        for line in lines:
            lineValues = []
            for coordinate in line:
                i, j = coordinate
                lineValues.append(board[i][j])
            ## Count the players in each iteration of line
            playerCount = lineValues.count(player)
            opponentCount = lineValues.count(opponent)

            ## Win potential, bonus
            if (playerCount == 2 and opponentCount == 1):
                evalScore += 10
            ## Blocking potential
            if (opponentCount == 2 and playerCount == 1):
                evalScore -= 10
            ## Win
            if (playerCount == 3):
                evalScore += 20
            ## Lose
            if (opponentCount == 3):
                evalScore -= 20

        return evalScore

    @staticmethod
    def evaluateGame(state: State, player: int) -> float:
        """
        Evaluates the current game state for the player.
        High == means good for my Agent
        Low == means bad for my agent

        Strategy:
        1. Find all local boards eval scores and scale them accordingly to their meta position
        2. Find the total evaluation score for the meta board

        Parameters
        ---------------
        state: The game state to evaluate.

        Returns
        ---------------
        The evaluation score for the game state.
        """
        if (player == 1):
            opponent = 2
        else:
            opponent = 1

        board = state.prev_local_action  ## Get the current board (if there is a forced one)
        if board is None:
            board = (-1, -1) ## Fallback when is the opening / the forced board is already won

        metaBoardWEights = np.array([
            [1.4, 1.0, 1.4],
            [1.0, 1.75, 1.0],
            [1.4, 1.0, 1.4]
        ]) ## Follows similar logic to the local board evaluation

        totalEval = 0.0
        metaBoard = np.zeros((3, 3), dtype=int) ## Represents the status of each board, initialise to all 0s

        ## Evaluates all the boards in metaboard
        for i in range(3):
            for j in range(3):
                localBoard = state.board[i][j]
                localEval = StudentAgent.evaluateLocalBoard(localBoard, player)
                totalEval += localEval * 1.5 * metaBoardWEights[i][j] ## Scale the evaluation values up

                ## Boost the score abit since im forced to play in this board
                if ((i,j) == board):
                    totalEval += localEval * metaBoardWEights[i][j]

                currentStatus = state.local_board_status[i][j]
                metaBoard[i][j] = currentStatus ## updates the localboardStatus in metaboard

                ## If the local board is won, add a bonus to the score
                if (currentStatus == player):
                    totalEval += 20 * metaBoardWEights[i][j]
                elif (currentStatus == opponent):
                    totalEval -= 20 * metaBoardWEights[i][j]


        metaStatus = board_status(metaBoard)
        if metaStatus == player:
            totalEval += 5000
        elif metaStatus == opponent:
            totalEval -= 5000
        else:
            totalEval += StudentAgent.evaluateLocalBoard(metaBoard, player) * 150

        return totalEval


    def minimax(self, state: State, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        """
        Minimax algorithm to find the best action to take.
        """

        if depth == 0 or state.is_terminal():
            return StudentAgent.evaluateGame(state, self.player)

        if maximizing: ## Maximizing player
            v = float('-inf')
            for action in state.get_all_valid_actions():
                v = max(v, self.minimax(state.change_state(action), depth - 1, alpha, beta, False))
                alpha = max(alpha, v)
                if v >= beta:
                    return v
            return v
        else: ## Minimizing player
            v = float('inf')
            for action in state.get_all_valid_actions():
                v = min(v, self.minimax(state.change_state(action), depth - 1, alpha, beta, True))
                beta = min(beta, v)
                if v <= alpha:
                    return v
            return v


    def choose_action(self, state: State) -> Action:
        """Returns a valid action to be played on the board.
        Assuming that you are filling in the board with number 1.

        Parameters
        ---------------
        state: The board to make a move on.
        """

        move_count = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        if state.board[i][j][k][l] != 0:
                            move_count += 1

        # Adapt depth based on move count
        if move_count < 20:
            depth = 2
        elif move_count < 45:
            depth = 3
        else:
            depth = 4  # late game, go full depth

        ## Initialise
        bestScore = float('-inf')
        bestAction = None

        for action in state.get_all_valid_actions():
            nextState = state.change_state(action)
            score = self.minimax(nextState, depth, float('-inf'), float('inf'), False)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction


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
            [[0, 2, 0], [0, 0, 0], [0, 0, 0]],
        ],
    ]),
    fill_num=1,
    prev_action=(2, 2, 0, 1),
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
