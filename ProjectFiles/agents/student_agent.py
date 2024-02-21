# Student agent: Add your own agent here

from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    Conventions in our code:
    1. function prefixes:
    -a/b/c/d for the section of the function
    -p for "private" methods that are only used by their section
    -h for specific helper methods that check the validity of a move (there are many slightly different ones)

    2. "hard-coded" elements:
    A heuristic of 1000 or -1000 represents a winning or a losing move.
    These numbers are hard-coded since this is a relatively small project, and it helps code readability
    to have actual numbers in the code.

    3. FIXME: TIMEOUT
    This is a comment added next to the timeout checker method when it is called.
    Nothing needs to be fixed!
    It just highlights the comment in yellow and helps us know which code is not part of the code logic,
    but is just checking the time left in our turn.
    """
    # MY CODE BELOW-------------------------------------------------------
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # SECTION 0:

    # Moves (Up, Right, Down, Left)
    MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))
    # Opposite Directions
    OPPOSITES = {0: 2, 1: 3, 2: 0, 3: 1}
    EMPTY_SET = set()

    def time_taken(self):  # this is used for testing purposes
        return time.time() - self.start_time

    def timeout(self):  # are we almost out of time?
        return (time.time() - self.start_time) > 1.9

    # --------------------------------------------------------------------
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # SECTION A:

    def ap_get_distance(self, pos1, pos2):
        # get the Manhattan distance
        r1, c1 = pos1
        r2, c2 = pos2
        return abs(r1 - r2) + abs(c1 - c2)

    def ah_is_move_valid_promising_squares_helper(
            self, chess_board, old_pos, new_pos, direction, adv_pos, found_squares):
        old_r, old_c = old_pos  # row col
        new_r, new_c = new_pos  # row col
        if new_r >= chess_board.shape[0] or new_c >= chess_board.shape[1] or new_r < 0 or new_c < 0:
            return False  # is not on the board
        if new_pos == adv_pos:
            return False  # is the adversary position
        if new_pos in found_squares:
            return False  # is already found
        if chess_board[old_r, old_c, direction]:
            return False  # wall between old_pos, new_pos
        return True

    def ap_get_promising_squares(self, chess_board, my_pos, adv_pos, max_step):
        # PROMISING SQUARES ------------------------------------------------------------------
        furthest_right = furthest_left = furthest_up = furthest_down = my_pos
        opponent_right = opponent_left = opponent_up = opponent_down = my_pos
        middle_closest = my_pos
        middle_of_the_board = (int(chess_board.shape[0] / 2), int(chess_board.shape[1] / 2))
        # ------------------------------------------------------------------------------------
        possible_squares = {my_pos}  # obviously my position is possible
        visiting = []  # queue1 for bfs
        visiting.append(my_pos)
        next_visiting = []  # queue2 for bfs
        for i in range(max_step):  # run bfs at range max_step
            while len(visiting) > 0:  # visit every position at this level of bfs
                if self.timeout():  # FIXME: TIMEOUT
                    return
                current_square = visiting.pop()
                r, c = current_square
                moves = ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1))  # u,r,d,l in order
                for direction in range(4):  # try to move in 4 directions
                    square = moves[direction]
                    if self.ah_is_move_valid_promising_squares_helper(chess_board, current_square,
                                                                      square, direction,
                                                                      adv_pos, possible_squares):
                        possible_squares.add(square)  # this move is reachable
                        next_visiting.append(square)  # continue bfs at this move for the next iteration
                        # PROMISING SQUARES -----------------------------------------------
                        if (self.ap_get_distance(middle_of_the_board, square) <
                                self.ap_get_distance(middle_of_the_board, middle_closest)):
                            middle_closest = square
                        # ---
                        if (square[0] < adv_pos[0] and square[1] >= adv_pos[1] and
                                (self.ap_get_distance(adv_pos, square) <=
                                 self.ap_get_distance(adv_pos, opponent_up))):
                            opponent_up = square
                        elif (square[0] >= adv_pos[0] and square[1] > adv_pos[1] and
                                (self.ap_get_distance(adv_pos, square) <=
                                 self.ap_get_distance(adv_pos, opponent_right))):
                            opponent_right = square
                        elif (square[0] > adv_pos[0] and square[1] <= adv_pos[1] and
                                (self.ap_get_distance(adv_pos, square) <=
                                 self.ap_get_distance(adv_pos, opponent_down))):
                            opponent_down = square
                        elif (square[0] <= adv_pos[0] and square[1] < adv_pos[1] and
                                (self.ap_get_distance(adv_pos, square) <=
                                 self.ap_get_distance(adv_pos, opponent_left))):
                            opponent_left = square
                        # ---
                        if (square[0] < my_pos[0] and square[1] >= my_pos[1] and
                                (self.ap_get_distance(my_pos, square) >
                                 self.ap_get_distance(my_pos, furthest_up))):
                            furthest_up = square
                        elif (square[0] >= my_pos[0] and square[1] > my_pos[1] and
                                (self.ap_get_distance(my_pos, square) >
                                 self.ap_get_distance(my_pos, furthest_right))):
                            furthest_right = square
                        elif (square[0] > my_pos[0] and square[1] <= my_pos[1] and
                                (self.ap_get_distance(my_pos, square) >
                                 self.ap_get_distance(my_pos, furthest_down))):
                            furthest_down = square
                        elif (square[0] <= my_pos[0] and square[1] < my_pos[1] and
                                (self.ap_get_distance(my_pos, square) >
                                 self.ap_get_distance(my_pos, furthest_left))):
                            furthest_left = square
                        # ----------------------------------------------------------------
            temp = visiting  # old queue
            visiting = next_visiting  # update queue
            next_visiting = temp  # reuse the old empty queue to add new squares
        # PROMISING SQUARES ---------------------------------------------------------------
        promising_squares_set = {furthest_right, furthest_left, furthest_up, furthest_down,
                                 opponent_right, opponent_left, opponent_up, opponent_down, middle_closest}
        promising_squares = [s for s in promising_squares_set]
        return promising_squares
        # ---------------------------------------------------------------------------------

    def ah_is_move_valid_best_squares_helper(
            self, chess_board, old_pos, new_pos, direction, my_squares, op_squares):
        old_r, old_c = old_pos  # row col
        new_r, new_c = new_pos  # row col
        if new_r >= chess_board.shape[0] or new_c >= chess_board.shape[1] or new_r < 0 or new_c < 0:
            return False  # is not on the board
        if new_pos in my_squares or new_pos in op_squares:  # already found
            return False  # is already found
        if chess_board[old_r, old_c, direction]:
            return False  # wall between old_pos, new_pos
        return True

    def a_get_best_squares(self, chess_board, my_pos, adv_pos, max_step):
        if self.timeout():  # FIXME: TIMEOUT
            return
        promising_positions = self.ap_get_promising_squares(chess_board, my_pos, adv_pos, max_step)
        if self.timeout():  # FIXME: TIMEOUT
            return
        number_of_positions = len(promising_positions)
        my_squares_per_position = [{pos} for pos in promising_positions]
        my_visiting_per_position = [[] for _ in promising_positions]
        for i in range(number_of_positions):
            my_visiting_per_position[i].append(promising_positions[i])
        my_next_visiting_per_position = [[] for _ in promising_positions]
        my_square_count_per_position = [1 for _ in range(number_of_positions)]
        neutral_square_count_per_position = [0 for _ in range(number_of_positions)]

        op_squares = {adv_pos}
        op_squares_preview = {adv_pos}
        op_visiting = []
        op_visiting.append(adv_pos)
        op_next_visiting = []
        op_square_count = 1

        while True:
            # PART 1, get next squares
            while len(op_visiting) > 0:  # opponent BFS
                if self.timeout():  # FIXME: TIMEOUT
                    return
                current_square = op_visiting.pop()
                r, c = current_square
                moves = ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1))  # u,r,d,l in order
                for direction in range(4):  # try to move in 4 directions
                    if self.ah_is_move_valid_best_squares_helper(chess_board, current_square, moves[direction],
                                                                 direction, op_squares_preview, self.EMPTY_SET):
                        op_square_count += 1
                        op_next_visiting.append(moves[direction])
                        op_squares_preview.add(moves[direction])
            for i in range(number_of_positions):  # Simultaneous BFS for each square
                my_visiting = my_visiting_per_position[i]
                my_next_visiting = my_next_visiting_per_position[i]
                my_squares = my_squares_per_position[i]
                while len(my_visiting) > 0:
                    if self.timeout():  # FIXME: TIMEOUT
                        return
                    current_square = my_visiting.pop()
                    r, c = current_square
                    moves = ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1))  # u,r,d,l in order
                    for direction in range(4):  # try to move in 4 directions
                        if self.ah_is_move_valid_best_squares_helper(chess_board, current_square,
                                                                     moves[direction], direction,
                                                                     my_squares, op_squares):
                            if moves[direction] in op_squares_preview:
                                neutral_square_count_per_position[i] = neutral_square_count_per_position[i] + 1
                            else:
                                my_square_count_per_position[i] = my_square_count_per_position[i] + 1
                            my_squares.add(moves[direction])
                            my_next_visiting.append(moves[direction])
            # PART 2, reset
            for sq in op_next_visiting:
                op_squares.add(sq)
            temp = op_visiting  # empty
            op_visiting = op_next_visiting
            op_next_visiting = temp
            finished = 0
            for i in range(number_of_positions):
                temp = my_visiting_per_position[i]
                my_visiting_per_position[i] = my_next_visiting_per_position[i]
                my_next_visiting_per_position[i] = temp
                if len(my_visiting_per_position[i]) == 0:
                    finished += 1
            if finished >= number_of_positions and len(op_visiting) == 0:
                break
        # PART 3, heuristics
        heuristics = [(my_square_count_per_position[i] -
                      (op_square_count - my_square_count_per_position[i] - neutral_square_count_per_position[i]),
                       promising_positions[i])
                      for i in range(number_of_positions)]
        heuristics.sort()
        if my_pos not in op_squares:
            if heuristics[0][0] == 0:
                return (), None
            elif heuristics[0][0] > 0:
                return (), 1000
            return (), -1000
        if number_of_positions == 1:
            return [heuristics[0][1]], (heuristics[0][0],)
        if number_of_positions == 2:
            return [heuristics[-1][1], heuristics[-2][1]], (heuristics[-1][0], heuristics[-2][0])
        return ([heuristics[-1][1], heuristics[-2][1], heuristics[-3][1]],
                (heuristics[-1][0], heuristics[-2][0], heuristics[-3][0]))

    # --------------------------------------------------------------------
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # SECTION B:

    def bp_get_opposite_barrier(self, r, c, d):
        move = self.MOVES[d]
        return r + move[0], c + move[1], self.OPPOSITES[d]

    def bp_does_wall_exist(self, chess_board, size, r, c, d):
        return 0 <= r < size and 0 <= c < size and chess_board[r, c, d]

    def bp_get_wall_neighbors(self, chess_board, size, r, c, d):
        if d == 0:
            left_neighbors = []
            if self.bp_does_wall_exist(chess_board, size, r, c, 3):
                left_neighbors.append((r, c, 3))
            if self.bp_does_wall_exist(chess_board, size, r - 1, c, 3):
                left_neighbors.append((r-1, c, 3))
            if self.bp_does_wall_exist(chess_board, size, r, c - 1, 0):
                left_neighbors.append((r, c-1, 0))
            right_neighbors = []
            if self.bp_does_wall_exist(chess_board, size, r, c, 1):
                right_neighbors.append((r, c, 1))
            if self.bp_does_wall_exist(chess_board, size, r - 1, c, 1):
                right_neighbors.append((r-1, c, 1))
            if self.bp_does_wall_exist(chess_board, size, r, c + 1, 0):
                right_neighbors.append((r, c+1, 0))
            return left_neighbors, right_neighbors
        elif d == 3:
            top_neighbors = []
            if self.bp_does_wall_exist(chess_board, size, r, c, 0):
                top_neighbors.append((r, c, 0))
            if self.bp_does_wall_exist(chess_board, size, r, c - 1, 0):
                top_neighbors.append((r, c-1, 0))
            if self.bp_does_wall_exist(chess_board, size, r - 1, c, 3):
                top_neighbors.append((r-1, c, 3))
            bot_neighbors = []
            if self.bp_does_wall_exist(chess_board, size, r, c, 2):
                bot_neighbors.append((r, c, 2))
            if self.bp_does_wall_exist(chess_board, size, r, c - 1, 2):
                bot_neighbors.append((r, c-1, 2))
            if self.bp_does_wall_exist(chess_board, size, r + 1, c, 3):
                bot_neighbors.append((r+1, c, 3))
            return top_neighbors, bot_neighbors

    def bp_get_wall_set(self, chess_board):
        size = chess_board.shape[0]
        edge_set = set()
        wall_sets = []
        for c in range(size):
            edge_set.add((0, c, 0))
        for c in range(size):
            edge_set.add((size - 1, c, 2))
        for r in range(size):
            edge_set.add((r, 0, 3))
            edge_set.add((r, size - 1, 1))
        for r in range(size):  # every row
            for c in range(size):  # every col
                if self.timeout():  # FIXME: TIMEOUT
                    return
                for d in (0, 3):  # top and left walls
                    if (r, c, d) in edge_set or not chess_board[r, c, d]:
                        continue
                    opposite = self.bp_get_opposite_barrier(r, c, d)
                    if not self.bp_does_wall_exist(chess_board, size, opposite[0], opposite[1], opposite[2]):
                        opposite = None
                    n1, n2 = self.bp_get_wall_neighbors(chess_board, size, r, c, d)
                    wall_set1 = None
                    wall_set2 = None
                    # check what sets neighbors belong to
                    for n in n1:
                        if n in edge_set:
                            wall_set1 = edge_set
                            break
                    for n in n2:
                        if n in edge_set:
                            wall_set2 = edge_set
                            break
                    if wall_set1 is None and len(n1) > 0:
                        for wall_set in wall_sets:
                            for n in n1:
                                if n in wall_set:
                                    wall_set1 = wall_set
                                    break
                    if wall_set2 is None and len(n2) > 0:
                        for wall_set in wall_sets:
                            for n in n2:
                                if n in wall_set:
                                    wall_set2 = wall_set
                                    break
                    full_set = None
                    old_set = None
                    # add the new wall to a set and possibly merge some sets
                    if wall_set1 is not None and wall_set2 is not None:
                        if wall_set1 is wall_set2:
                            full_set = wall_set1
                        elif wall_set1 is edge_set:
                            edge_set.update(wall_set2)
                            full_set = edge_set
                            old_set = wall_set2
                        elif wall_set2 is edge_set:
                            edge_set.update(wall_set1)
                            full_set = edge_set
                            old_set = wall_set1
                        else:
                            wall_set1.update(wall_set2)
                            full_set = wall_set1
                            old_set = wall_set2
                        if old_set is not None:
                            wall_sets.remove(old_set)
                    elif wall_set1 is not None:
                        full_set = wall_set1
                    elif wall_set2 is not None:
                        full_set = wall_set2
                    else:
                        full_set = set()
                    full_set.add((r, c, d))
                    if opposite is not None:
                        full_set.add(opposite)
                    if wall_set1 is None and wall_set2 is None:
                        wall_sets.append(full_set)
        return edge_set

    def bp_is_square_game_ending(self, chess_board, r, c, wall_set):
        size = chess_board.shape[0]
        top_left = False
        top_right = False
        bot_left = False
        bot_right = False
        if ((r - 1, c - 1, 2) in wall_set or
            (r - 1, c - 1, 1) in wall_set or
            (r - 1, c, 3) in wall_set or
                (r, c - 1, 0) in wall_set):
            top_left = True
        if ((r - 1, c + 1, 3) in wall_set or
            (r - 1, c + 1, 2) in wall_set or
            (r - 1, c, 1) in wall_set or
                (r, c + 1, 0) in wall_set):
            top_right = True
        if ((r + 1, c + 1, 0) in wall_set or
            (r + 1, c + 1, 3) in wall_set or
            (r, c + 1, 2) in wall_set or
                (r + 1, c, 1) in wall_set):
            bot_right = True
        if ((r + 1, c - 1, 0) in wall_set or
            (r + 1, c - 1, 1) in wall_set or
            (r, c - 1, 2) in wall_set or
                (r + 1, c, 3) in wall_set):
            bot_left = True
        if (r, c, 0) in wall_set:
            top_right = top_left = True
        if (r, c, 1) in wall_set:
            top_right = bot_right = True
        if (r, c, 2) in wall_set:
            bot_right = bot_left = True
        if (r, c, 3) in wall_set:
            bot_left = top_left = True
        top = self.bp_does_wall_exist(chess_board, size, r, c, 0)
        right = self.bp_does_wall_exist(chess_board, size, r, c, 1)
        left = self.bp_does_wall_exist(chess_board, size, r, c, 3)
        bot = self.bp_does_wall_exist(chess_board, size, r, c, 2)
        n_orthogonal = len([0 for i in (top, right, left, bot) if i])
        n_diagonal = len([0 for i in (top_left, top_right, bot_right, bot_left) if i])
        if n_orthogonal >= 3 or n_diagonal <= 1:
            return False
        if n_diagonal == 4 or (n_diagonal >= 3 and n_orthogonal <= 1) or (n_diagonal >= 2 and n_orthogonal == 0):
            return True
        if top and bot:
            return (top_left or top_right) and (bot_left or bot_right)
        if right and left:
            return (top_right or bot_right) and (top_left or bot_left)
        if top and right:
            return n_diagonal >= 2 and bot_left
        if right and bot:
            return n_diagonal >= 2 and top_left
        if bot and left:
            return n_diagonal >= 2 and top_right
        if left and top:
            return n_diagonal >= 2 and bot_right
        # we have exactly 1 orthogonal and 2 diagonal
        if left:
            return not (top_left and bot_left)
        if right:
            return not (top_right and bot_right)
        if top:
            return not (top_right and top_left)
        if bot:
            return not (bot_right and bot_left)
        return False

    def bh_is_move_valid_game_ending_squares_helper(
            self, chess_board, old_pos, new_pos, direction, adv_pos, found_squares):
        old_r, old_c = old_pos  # row col
        new_r, new_c = new_pos  # row col
        if new_r >= chess_board.shape[0] or new_c >= chess_board.shape[1] or new_r < 0 or new_c < 0:
            return False  # is not on the board
        if new_pos == adv_pos:
            return False  # is the adversary position
        if new_pos in found_squares:
            return False  # is already found
        if chess_board[old_r, old_c, direction]:
            return False  # wall between old_pos, new_pos
        return True

    def bp_get_game_ending_squares(self, chess_board, my_pos, adv_pos, max_step, wall_set):
        # GAME ENDING SQUARES ------------------------------------------------------------------
        game_ending_squares = set()
        # ------------------------------------------------------------------------------------
        possible_squares = {my_pos}  # obviously my position is possible
        visiting = []  # queue1 for bfs
        visiting.append(my_pos)
        next_visiting = []  # queue2 for bfs
        if self.bp_is_square_game_ending(chess_board, my_pos[0], my_pos[1], wall_set):
            game_ending_squares.add(my_pos)
        for i in range(max_step):  # run bfs at range max_step
            while len(visiting) > 0:  # visit every position at this level of bfs
                if self.timeout():  # FIXME: TIMEOUT
                    return
                current_square = visiting.pop()
                r, c = current_square
                moves = ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1))  # u,r,d,l in order
                for direction in range(4):  # try to move in 4 directions
                    square = moves[direction]
                    if self.bh_is_move_valid_game_ending_squares_helper(chess_board, current_square,
                                                                        square, direction,
                                                                        adv_pos, possible_squares):
                        possible_squares.add(square)  # this move is reachable
                        next_visiting.append(square)  # continue bfs at this move for the next iteration
                        # GAME ENDING SQUARES -----------------------------------------------
                        if self.bp_is_square_game_ending(chess_board, square[0], square[1], wall_set):
                            game_ending_squares.add(square)
                        # ----------------------------------------------------------------
            temp = visiting  # old queue
            visiting = next_visiting  # update queue
            next_visiting = temp  # reuse the old empty queue to add new squares
        # GAME ENDING SQUARES ---------------------------------------------------------------
        return game_ending_squares
        # ---------------------------------------------------------------------------------

    def bh_is_move_valid_possible_move_helper(
            self, chess_board, old_pos, new_pos, direction, my_squares):
        old_r, old_c = old_pos  # row col
        new_r, new_c = new_pos  # row col
        if new_r >= chess_board.shape[0] or new_c >= chess_board.shape[1] or new_r < 0 or new_c < 0:
            return False  # is not on the board
        if new_pos in my_squares:  # already found
            return False  # is already found
        if chess_board[old_r, old_c, direction]:
            return False  # wall between old_pos, new_pos
        return True

    def bp_get_possible_game_winning_moves(self, chess_board, my_pos, adv_pos, max_step):
        if self.timeout():  # FIXME: TIMEOUT
            return
        wall_set = self.bp_get_wall_set(chess_board)
        if self.timeout():  # FIXME: TIMEOUT
            return
        game_ending_squares = self.bp_get_game_ending_squares(chess_board, my_pos, adv_pos, max_step, wall_set)
        if self.timeout():  # FIXME: TIMEOUT
            return
        possible_game_winning_squares = set()
        possible_game_winning_moves = []

        op_squares = {adv_pos}
        op_visiting = []
        op_visiting.append(adv_pos)
        op_square_count = 1

        while len(op_visiting) > 0:  # run dfs and get the game ending moves that are reached by the opponent
            if self.timeout():  # FIXME: TIMEOUT
                return
            current_square = op_visiting.pop()
            r, c = current_square
            moves = ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1))  # u,r,d,l in order
            for direction in range(4):  # try to move in 4 directions
                if self.bh_is_move_valid_possible_move_helper(
                        chess_board, current_square, moves[direction], direction, op_squares):
                    if (moves[direction] in game_ending_squares and
                            moves[direction] not in possible_game_winning_squares):
                        possible_game_winning_squares.add(moves[direction])
                        possible_game_winning_moves.append((moves[direction], self.OPPOSITES[direction]))
                    if moves[direction] in game_ending_squares:
                        continue
                    op_square_count += 1
                    op_visiting.append(moves[direction])
                    op_squares.add(moves[direction])
        return possible_game_winning_moves, possible_game_winning_squares, op_squares, op_square_count

    def bh_is_move_valid_winning_move_helper_op(
            self, chess_board, old_pos, new_pos, direction, my_squares, op_squares):
        old_r, old_c = old_pos  # row col
        new_r, new_c = new_pos  # row col
        if new_r >= chess_board.shape[0] or new_c >= chess_board.shape[1] or new_r < 0 or new_c < 0:
            return False  # is not on the board
        if new_pos in my_squares or new_pos in op_squares:  # already found
            return False  # is already found
        if chess_board[old_r, old_c, direction]:
            return False  # wall between old_pos, new_pos
        return True

    def b_get_winning_move(self, chess_board, my_pos, adv_pos, max_step):
        if self.timeout():  # FIXME: TIMEOUT
            return
        var_time_safe = self.bp_get_possible_game_winning_moves(
            chess_board, my_pos, adv_pos, max_step)
        if self.timeout():  # FIXME: TIMEOUT
            return
        possible_moves, possible_squares, op_squares, op_count = var_time_safe
        for possible_move in possible_moves:
            if self.timeout():  # FIXME: TIMEOUT
                return

            op_squares = op_squares.copy()
            op_visiting = []
            for sq in possible_squares:
                if sq != possible_move[0]:
                    op_visiting.append(sq)
            op_square_count = op_count

            while len(op_visiting) > 0:  # count opponent squares with BFS
                if self.timeout():  # FIXME: TIMEOUT
                    return
                current_square = op_visiting.pop()
                r, c = current_square
                moves = ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1))  # u,r,d,l in order
                for direction in range(4):  # try to move in 4 directions
                    if self.bh_is_move_valid_winning_move_helper_op(
                            chess_board, current_square, moves[direction], direction, op_squares, self.EMPTY_SET):
                        if moves[direction] == possible_move[0]:
                            continue
                        op_square_count += 1
                        op_visiting.append(moves[direction])
                        op_squares.add(moves[direction])

            my_squares = {possible_move[0]}
            my_visiting = []
            my_visiting.append(possible_move[0])
            my_square_count = 1

            while len(my_visiting) > 0:  # count our squares with BFS
                if self.timeout():  # FIXME: TIMEOUT
                    return
                current_square = my_visiting.pop()
                r, c = current_square
                moves = ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1))  # u,r,d,l in order
                for direction in range(4):  # try to move in 4 directions
                    if self.bh_is_move_valid_winning_move_helper_op(
                            chess_board, current_square, moves[direction], direction, my_squares, op_squares):
                        my_square_count += 1
                        my_visiting.append(moves[direction])
                        my_squares.add(moves[direction])

            if my_square_count > op_square_count:
                return possible_move
        return None

    # --------------------------------------------------------------------
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # SECTION C:

    def c_get_possible_walls(self, chess_board, pos):
        r, c = pos
        allowed_barriers = [i for i in range(0, 4) if not chess_board[r, c, i]]
        return allowed_barriers

    def c_get_possible_moves(self, chess_board, all_squares):
        all_moves = []
        for square in all_squares:
            for wall in self.c_get_possible_walls(chess_board, square):
                all_moves.append((square, wall))
        return all_moves

    def cp_set_barrier(self, chess_board, r, c, d):
        # Set the barrier to True
        chess_board[r, c, d] = True
        # Set the opposite barrier to True
        move = self.MOVES[d]
        chess_board[r + move[0], c + move[1], self.OPPOSITES[d]] = True

    def c_new_board(self, chess_board, move):
        new_chess_board = deepcopy(chess_board)
        (r, c), d = move
        self.cp_set_barrier(new_chess_board, r, c, d)
        return new_chess_board

    # --------------------------------------------------------------------
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # SECTION D:

    def d_leaf_max(self, max_step, moves):
        # return the h for moves. We are assuming that the base move is their move
        # (move, board, [sub_moves/sub_moves])
        move_heuristics = []
        for move in moves:  # these are our formatted moves
            if self.timeout():  # FIXME: TIMEOUT
                return
            if move[2] is None:
                move_heuristics.append(0)
                continue
            elif move[2] == 1000 or move[2] == -1000:
                move_heuristics.append(move[2])
                continue
            sub_moves = self.c_get_possible_moves(move[1], move[2])
            formatted_sub_moves_h = []
            sub_move_heuristics = []
            for sub_move in sub_moves:  # these are opponent moves
                sub_board = self.c_new_board(move[1], sub_move)
                # my best squares in response
                if self.timeout():  # FIXME: TIMEOUT
                    return
                winning_move = self.b_get_winning_move(sub_board, move[0][0], sub_move[0], max_step)
                if self.timeout():  # FIXME: TIMEOUT
                    return
                if winning_move is not None:  # we found a winning move
                    sub_move_h = 1000
                    sub_move_heuristics.append(sub_move_h)
                    formatted_sub_move = (sub_move, sub_board, sub_move_h * -1)
                    formatted_sub_moves_h.append((sub_move_h, formatted_sub_move))
                    continue
                if self.timeout():  # FIXME: TIMEOUT
                    return
                var_time_safe = self.a_get_best_squares(sub_board, move[0][0], sub_move[0], max_step)
                if self.timeout():  # FIXME: TIMEOUT
                    return
                sub_squares, sub_square_hs = var_time_safe
                if sub_square_hs == 1000 or sub_square_hs == -1000 or sub_square_hs is None:
                    sub_move_h = 0 if sub_square_hs is None else sub_square_hs
                    sub_move_heuristics.append(sub_move_h)
                    formatted_sub_move = (sub_move, sub_board, None if sub_square_hs is None else sub_move_h * -1)
                    formatted_sub_moves_h.append((sub_move_h, formatted_sub_move))
                    continue
                sub_move_h = max(sub_square_hs)  # our best response is the Heuristic
                sub_move_heuristics.append(sub_move_h)
                formatted_sub_move = (sub_move, sub_board, sub_squares)
                formatted_sub_moves_h.append((sub_move_h, formatted_sub_move))
            formatted_sub_moves_h.sort()
            if len(formatted_sub_moves_h) >= 4:
                del formatted_sub_moves_h[4:]
            formatted_sub_moves = [i[1] for i in formatted_sub_moves_h]
            move[2][:] = formatted_sub_moves
            move_heuristics.append(min(sub_move_heuristics))  # the opponent's best reaction to our move
        return max(move_heuristics)  # the best of our moves

    def d_recursive(self, max_step, moves, depth):
        if moves is None:
            return 0
        if depth == 0:
            if moves == 1000 or moves == -1000:
                return -moves
            return self.d_leaf_max(max_step, moves)
        if depth % 2 == 0:  # even depth -> low heuristic is good
            if moves == 1000 or moves == -1000:
                return -moves
            h = []
            for move in moves:
                if self.timeout():  # FIXME: TIMEOUT
                    return
                h.append(self.d_recursive(max_step, move[2], depth - 1))
                if self.timeout():  # FIXME: TIMEOUT
                    return
            return max(h)
        else:  # odd depth -> high heuristic is good
            if moves == 1000 or moves == -1000:
                return moves
            h = []
            for move in moves:
                if self.timeout():  # FIXME: TIMEOUT
                    return
                h.append(self.d_recursive(max_step, move[2], depth - 1))
                if self.timeout():  # FIXME: TIMEOUT
                    return
            return min(h)

    def d_get_best(self, chess_board, my_pos, adv_pos, max_step):
        # initialise data for depths 1 and 2
        if self.timeout():  # FIXME: TIMEOUT
            return
        winning_move = self.b_get_winning_move(chess_board, my_pos, adv_pos, max_step)
        if self.timeout():  # FIXME: TIMEOUT
            return
        if winning_move is not None:
            self.best_move = winning_move
            return
        if self.timeout():  # FIXME: TIMEOUT
            return
        var_time_safe = self.a_get_best_squares(chess_board, my_pos, adv_pos, max_step)
        if self.timeout():  # FIXME: TIMEOUT
            return
        squares, _ = var_time_safe
        self.best_move = squares, self.c_get_possible_walls(chess_board, squares[0])[0]
        # depth 1
        dummy_moves = [((adv_pos,), chess_board, squares)]
        if self.timeout():  # FIXME: TIMEOUT
            return
        self.d_leaf_max(max_step, dummy_moves)
        if self.timeout():  # FIXME: TIMEOUT
            return
        top_moves = dummy_moves[0][2]
        self.best_move = top_moves[0][0]
        # depth 2
        new_best_move = self.best_move
        max_h = -1000
        for move in top_moves:
            dummy_moves = [move]
            if self.timeout():  # FIXME: TIMEOUT
                return
            h = self.d_leaf_max(max_step, dummy_moves)
            if self.timeout():  # FIXME: TIMEOUT
                return
            if h > max_h:
                new_best_move = move[0]
                max_h = h
        self.best_move = new_best_move
        # depth 3+
        depth = 0
        while depth <= 100:
            new_best_move = self.best_move
            if depth % 2 == 0:  # depth is even: get min heuristic -> opponent moves are expanded
                min_h = 1000
                for move in top_moves:
                    if self.timeout():  # FIXME: TIMEOUT
                        return
                    h = self.d_recursive(max_step, move[2], depth)
                    if self.timeout():  # FIXME: TIMEOUT
                        return
                    if h < min_h:
                        new_best_move = move[0]
                        min_h = h
            else:  # depth is odd: get max heuristic -> our moves are expanded
                max_h = -1000
                for move in top_moves:
                    if self.timeout():  # FIXME: TIMEOUT
                        return
                    h = self.d_recursive(max_step, move[2], depth)
                    if self.timeout():  # FIXME: TIMEOUT
                        return
                    if h > max_h:
                        new_best_move = move[0]
                        max_h = h
            self.best_move = new_best_move
            depth += 1

    # --------------------------------------------------------------------
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # SECTION TEST:

    # Testing code
    def t_print_all_barriers(self, chess_board):
        # (square/barrier + sanity check for matching barriers)
        assert chess_board.shape[0] == chess_board.shape[1] and chess_board.shape[2] == 4
        size = chess_board.shape[0]
        barriers = set()
        barriers2 = set()
        for r in range(size):
            for c in range(size):
                if chess_board[r, c, 0] and r != 0:  # u
                    barriers.add(((r, c), 0))
                if chess_board[r, c, 1] and c != size - 1:  # r
                    barriers.add(((r, c), 1))
        # SANITY CHECK
        for r in range(size):
            for c in range(size):
                if chess_board[r, c, 2] and r != size - 1:  # d
                    assert ((r + 1, c), 0) in barriers
                    barriers2.add(((r, c), 2))
                if chess_board[r, c, 3] and c != 0:  # l
                    assert ((r, c - 1), 1) in barriers
                    barriers2.add(((r, c), 3))
        barriers = sorted(barriers)
        for barrier in barriers:
            p, d = barrier
            print("bar: " + str(p) + " " + ("u" if d == 0 else "r"))

    def t_print_info(self, chess_board, my_pos, adv_pos, max_step):
        # max step, player pos, op pos, board size.
        print("max step: " + str(max_step) + ", board_size: " + str(chess_board.shape[0]) +
              ", my_pos: " + str(my_pos) + ", op_pos: " + str(adv_pos))

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # MY CODE UP^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        # MY CODE BELOW---------------------------------------------------
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

        self.best_move = None
        self.start_time = 0

        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # MY CODE UP^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        self.start_time = time.time()

        # MY CODE BELOW-------------------------------------------------------
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

        self.d_get_best(chess_board, my_pos, adv_pos, max_step)  # run our main function
        return self.best_move  # return the best move found

        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # MY CODE UP^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
