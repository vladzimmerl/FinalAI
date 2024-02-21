'''
# Student agent: Add your own agent here
import collections

from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time


@register_agent("second_agent")
class SecondAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """
    # MY CODE DOWN--------------------------------------------------------
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # SECTION 0:
    is_testing = False

    def time_taken(self):
        return time.time() - self.start_time

    def timeout(self):
        return (time.time() - self.start_time) > 1.9

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # SECTION A:

    @staticmethod
    def ap_is_move_valid_heuristic_helper(chess_board, old_pos, new_pos, direction, adv_pos,
                                          my_squares, op_squares, neutral_squares):
        old_r, old_c = old_pos  # row col
        new_r, new_c = new_pos  # row col
        if new_r >= chess_board.shape[0] or new_c >= chess_board.shape[1] or new_r < 0 or new_c < 0:
            return False  # is not on the board
        if new_pos == adv_pos:
            return False  # is the adversary position
        if new_pos in my_squares or new_pos in neutral_squares:  # we check op squares on its own
            return False  # is already found
        if chess_board[old_r, old_c, direction]:
            return False  # wall between old_pos, new_pos
        return True

    @staticmethod
    def a_heuristic(chess_board, my_pos, adv_pos):
        my_squares = {my_pos}
        op_squares = {adv_pos}
        my_visiting = collections.deque()
        op_visiting = collections.deque()
        my_visiting.append(my_pos)
        op_visiting.append(adv_pos)
        my_possible_squares = collections.deque()
        op_possible_squares = collections.deque()
        neutral_squares = set()

        game_over = True

        while len(my_visiting) > 0 or len(op_visiting) > 0:
            # PART 1, get next squares
            while len(my_visiting) > 0:
                current_square = my_visiting.pop()
                r, c = current_square
                moves = ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1))  # u,r,d,l in order
                for direction in range(4):  # try to move in 4 directions
                    if SecondAgent.ap_is_move_valid_heuristic_helper(chess_board, current_square,
                                                                      moves[direction], direction, adv_pos,
                                                                      my_squares, op_squares, neutral_squares):
                        if moves[direction] in op_squares:
                            game_over = False
                            continue
                        my_possible_squares.append(moves[direction])
            while len(op_visiting) > 0:
                current_square = op_visiting.pop()
                r, c = current_square
                moves = ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1))  # u,r,d,l in order
                for direction in range(4):  # try to move in 4 directions
                    if SecondAgent.ap_is_move_valid_heuristic_helper(chess_board, current_square,
                                                                      moves[direction], direction, my_pos,
                                                                      op_squares, my_squares, neutral_squares):
                        if moves[direction] in my_squares:
                            continue
                        op_possible_squares.append(moves[direction])
            # PART 2, eliminate neutral squares
            while len(my_possible_squares) > 0:
                current_square = my_possible_squares.pop()
                if current_square in op_possible_squares:
                    neutral_squares.add(current_square)
                else:
                    my_visiting.append(current_square)
                    my_squares.add(current_square)
            while len(op_possible_squares) > 0:
                current_square = op_possible_squares.pop()
                if current_square in neutral_squares:
                    continue
                else:
                    op_visiting.append(current_square)
                    op_squares.add(current_square)
        heuristic = len(my_squares) - len(op_squares)
        if len(neutral_squares) > 0:
            game_over = False
        if game_over:
            return 0 if heuristic == 0 else (heuristic / abs(heuristic)) * 1000
        else:
            return heuristic

    @staticmethod
    def ap_get_distance(pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        return abs(r1 - r2) + abs(c1 - c2)

    @staticmethod
    def ap_is_move_valid_possible_squares_helper(chess_board, old_pos, new_pos, direction, adv_pos, found_squares):
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

    @staticmethod
    def ap_get_possible_squares(chess_board, my_pos, adv_pos, max_step):
        possible_squares = {my_pos}  # obviously my position is possible
        visiting = collections.deque()  # queue1 for bfs
        visiting.append(my_pos)
        next_visiting = collections.deque()  # queue2 for bfs
        for i in range(max_step):  # run bfs at range max_step
            while len(visiting) > 0:  # visit every position at this level of bfs
                current_square = visiting.pop()
                r, c = current_square
                moves = ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1))  # u,r,d,l in order
                for direction in range(4):  # try to move in 4 directions
                    if SecondAgent.ap_is_move_valid_possible_squares_helper(chess_board, current_square,
                                                                             moves[direction], direction,
                                                                             adv_pos, possible_squares):
                        possible_squares.add(moves[direction])  # this move is reachable
                        next_visiting.append(moves[direction])  # continue bfs at this move for the next iteration
            temp = visiting  # old queue
            visiting = next_visiting  # update queue
            next_visiting = temp  # reuse the old empty queue to add new squares
        return possible_squares

    @staticmethod
    def ap_get_promising_squares(chess_board, my_pos, adv_pos, possible_squares):
        furthest = [my_pos, my_pos, my_pos]
        opponent = [(1000, 1000), (1000, 1000), (1000, 1000), (1000, 1000)]
        middle = (1000, 1000)
        middle_of_the_board = (chess_board.shape[0] / 2, chess_board.shape[1] / 2)
        for square in possible_squares:
            if SecondAgent.ap_get_distance(my_pos, square) > SecondAgent.ap_get_distance(my_pos, furthest[0]):
                furthest[0] = square
                furthest.sort(key=lambda x: SecondAgent.ap_get_distance(my_pos, x), reverse=False)
            if SecondAgent.ap_get_distance(adv_pos, square) < SecondAgent.ap_get_distance(adv_pos, opponent[0]):
                opponent[0] = square
                opponent.sort(key=lambda x: SecondAgent.ap_get_distance(adv_pos, x), reverse=True)
            if (SecondAgent.ap_get_distance(middle_of_the_board, square) <
                    SecondAgent.ap_get_distance(middle_of_the_board, middle)):
                middle = square
        promising_squares = set()
        for square in furthest:
            promising_squares.add(square)
        for square in opponent:
            promising_squares.add(square)
        promising_squares.add(middle)
        if (1000, 1000) in promising_squares:
            promising_squares.remove((1000, 1000))
        return promising_squares

    @staticmethod
    def a_get_best_squares(chess_board, my_pos, adv_pos, max_step):
        possible_squares = SecondAgent.ap_get_possible_squares(chess_board, my_pos, adv_pos, max_step)
        promising_squares = SecondAgent.ap_get_promising_squares(chess_board, my_pos, adv_pos, possible_squares)
        squares_with_heuristics = []
        for square in promising_squares:
            heuristic = SecondAgent.a_heuristic(chess_board, square, adv_pos)
            squares_with_heuristics.append((square, heuristic))
        squares_with_heuristics.sort(key=lambda x: x[1], reverse=True)
        if len(squares_with_heuristics) == 1:
            return [squares_with_heuristics[0][0]]
        return [squares_with_heuristics[0][0], squares_with_heuristics[1][0]]

    # --------------------------------------------------------------------
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # SECTION A.2:

    def ap2_get_promising_squares(self, chess_board, my_pos, adv_pos, max_step):
        # PROMISING SQUARES ------------------------------------------------------------------
        furthest_right = furthest_left = furthest_up = furthest_down = my_pos
        opponent_right = opponent_left = opponent_up = opponent_down = my_pos
        middle_closest = my_pos
        middle_of_the_board = (int(chess_board.shape[0] / 2), int(chess_board.shape[1] / 2))
        # ------------------------------------------------------------------------------------
        possible_squares = {my_pos}  # obviously my position is possible
        visiting = collections.deque()  # queue1 for bfs
        visiting.append(my_pos)
        next_visiting = collections.deque()  # queue2 for bfs
        for i in range(max_step):  # run bfs at range max_step
            while len(visiting) > 0:  # visit every position at this level of bfs
                if self.time_taken() > 1.9:  # FIXME: TIMEOUT
                    return
                current_square = visiting.pop()
                r, c = current_square
                moves = ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1))  # u,r,d,l in order
                for direction in range(4):  # try to move in 4 directions
                    square = moves[direction]
                    if SecondAgent.ap_is_move_valid_possible_squares_helper(chess_board, current_square,
                                                                             square, direction,
                                                                             adv_pos, possible_squares):
                        possible_squares.add(square)  # this move is reachable
                        next_visiting.append(square)  # continue bfs at this move for the next iteration
                        # PROMISING SQUARES -----------------------------------------------
                        if (SecondAgent.ap_get_distance(middle_of_the_board, square) <
                                SecondAgent.ap_get_distance(middle_of_the_board, middle_closest)):
                            middle_closest = square
                        # ---
                        if (square[0] < adv_pos[0] and square[1] >= adv_pos[1] and
                                (SecondAgent.ap_get_distance(adv_pos, square) <=
                                 SecondAgent.ap_get_distance(adv_pos, opponent_up))):
                            opponent_up = square
                        elif (square[0] >= adv_pos[0] and square[1] > adv_pos[1] and
                                (SecondAgent.ap_get_distance(adv_pos, square) <=
                                 SecondAgent.ap_get_distance(adv_pos, opponent_right))):
                            opponent_right = square
                        elif (square[0] > adv_pos[0] and square[1] <= adv_pos[1] and
                                (SecondAgent.ap_get_distance(adv_pos, square) <=
                                 SecondAgent.ap_get_distance(adv_pos, opponent_down))):
                            opponent_down = square
                        elif (square[0] <= adv_pos[0] and square[1] < adv_pos[1] and
                                (SecondAgent.ap_get_distance(adv_pos, square) <=
                                 SecondAgent.ap_get_distance(adv_pos, opponent_left))):
                            opponent_left = square
                        # ---
                        if (square[0] < my_pos[0] and square[1] >= my_pos[1] and
                                (SecondAgent.ap_get_distance(my_pos, square) >
                                 SecondAgent.ap_get_distance(my_pos, furthest_up))):
                            furthest_up = square
                        elif (square[0] >= my_pos[0] and square[1] > my_pos[1] and
                                (SecondAgent.ap_get_distance(my_pos, square) >
                                 SecondAgent.ap_get_distance(my_pos, furthest_right))):
                            furthest_right = square
                        elif (square[0] > my_pos[0] and square[1] <= my_pos[1] and
                                (SecondAgent.ap_get_distance(my_pos, square) >
                                 SecondAgent.ap_get_distance(my_pos, furthest_down))):
                            furthest_down = square
                        elif (square[0] <= my_pos[0] and square[1] < my_pos[1] and
                                (SecondAgent.ap_get_distance(my_pos, square) >
                                 SecondAgent.ap_get_distance(my_pos, furthest_left))):
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

    @staticmethod
    def ap2_is_move_valid_best_squares_helper(chess_board, old_pos, new_pos, direction, my_squares, op_squares):
        old_r, old_c = old_pos  # row col
        new_r, new_c = new_pos  # row col
        if new_r >= chess_board.shape[0] or new_c >= chess_board.shape[1] or new_r < 0 or new_c < 0:
            return False  # is not on the board
        if new_pos in my_squares or new_pos in op_squares:  # already found
            return False  # is already found
        if chess_board[old_r, old_c, direction]:
            return False  # wall between old_pos, new_pos
        return True

    def a2_get_best_squares(self, chess_board, my_pos, adv_pos, max_step):
        if self.time_taken() > 1.9:  # FIXME: TIMEOUT
            return
        promising_positions = self.ap2_get_promising_squares(chess_board, my_pos, adv_pos, max_step)
        if self.time_taken() > 1.9:  # FIXME: TIMEOUT
            return
        number_of_positions = len(promising_positions)
        my_squares_per_position = [{pos} for pos in promising_positions]
        my_visiting_per_position = [collections.deque() for _ in promising_positions]
        for i in range(number_of_positions):
            my_visiting_per_position[i].append(promising_positions[i])
        my_next_visiting_per_position = [collections.deque() for _ in promising_positions]
        my_square_count_per_position = [1 for _ in range(number_of_positions)]
        neutral_square_count_per_position = [0 for _ in range(number_of_positions)]

        op_squares = {adv_pos}
        op_squares_preview = {adv_pos}
        op_visiting = collections.deque()
        op_visiting.append(adv_pos)
        op_next_visiting = collections.deque()
        op_square_count = 1
        empty_set = set()

        while True:
            # PART 1, get next squares
            while len(op_visiting) > 0:
                if self.time_taken() > 1.9:  # FIXME: TIMEOUT
                    return
                current_square = op_visiting.pop()
                r, c = current_square
                moves = ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1))  # u,r,d,l in order
                for direction in range(4):  # try to move in 4 directions
                    if SecondAgent.ap2_is_move_valid_best_squares_helper(chess_board, current_square, moves[direction],
                                                                          direction, op_squares_preview, empty_set):
                        op_square_count += 1
                        op_next_visiting.append(moves[direction])
                        op_squares_preview.add(moves[direction])
            for i in range(number_of_positions):
                my_visiting = my_visiting_per_position[i]
                my_next_visiting = my_next_visiting_per_position[i]
                my_squares = my_squares_per_position[i]
                while len(my_visiting) > 0:
                    if self.time_taken() > 1.9:  # FIXME: TIMEOUT
                        return
                    current_square = my_visiting.pop()
                    r, c = current_square
                    moves = ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1))  # u,r,d,l in order
                    for direction in range(4):  # try to move in 4 directions
                        if SecondAgent.ap2_is_move_valid_best_squares_helper(chess_board, current_square,
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
    # SECTION A.3:

    @staticmethod
    def a3_get_opposite_barrier(r, c, d):
        move = SecondAgent.MOVES[d]
        return r + move[0], c + move[1], SecondAgent.OPPOSITES[d]

    @staticmethod
    def a3_does_wall_exist(chess_board, size, r, c, d):
        return 0 <= r < size and 0 <= c < size and chess_board[r, c, d]

    @staticmethod
    def a3_get_wall_neighbors(chess_board, size, r, c, d):
        if d == 0:
            left_neighbors = []
            if SecondAgent.a3_does_wall_exist(chess_board, size, r, c, 3):
                left_neighbors.append((r, c, 3))
            if SecondAgent.a3_does_wall_exist(chess_board, size, r-1, c, 3):
                left_neighbors.append((r-1, c, 3))
            if SecondAgent.a3_does_wall_exist(chess_board, size, r, c-1, 0):
                left_neighbors.append((r, c-1, 0))
            right_neighbors = []
            if SecondAgent.a3_does_wall_exist(chess_board, size, r, c, 1):
                right_neighbors.append((r, c, 1))
            if SecondAgent.a3_does_wall_exist(chess_board, size, r-1, c, 1):
                right_neighbors.append((r-1, c, 1))
            if SecondAgent.a3_does_wall_exist(chess_board, size, r, c+1, 0):
                right_neighbors.append((r, c+1, 0))
            return left_neighbors, right_neighbors
        elif d == 3:
            top_neighbors = []
            if SecondAgent.a3_does_wall_exist(chess_board, size, r, c, 0):
                top_neighbors.append((r, c, 0))
            if SecondAgent.a3_does_wall_exist(chess_board, size, r, c-1, 0):
                top_neighbors.append((r, c-1, 0))
            if SecondAgent.a3_does_wall_exist(chess_board, size, r-1, c, 3):
                top_neighbors.append((r-1, c, 3))
            bot_neighbors = []
            if SecondAgent.a3_does_wall_exist(chess_board, size, r, c, 2):
                bot_neighbors.append((r, c, 2))
            if SecondAgent.a3_does_wall_exist(chess_board, size, r, c-1, 2):
                bot_neighbors.append((r, c-1, 2))
            if SecondAgent.a3_does_wall_exist(chess_board, size, r+1, c, 3):
                bot_neighbors.append((r+1, c, 3))
            return top_neighbors, bot_neighbors

    @staticmethod
    def a3_get_wall_set(chess_board):
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
        #
        if SecondAgent.is_testing:
            print("#########")
        for r in range(size):
            for c in range(size):
                for d in (0, 3):
                    if (r, c, d) in edge_set or not chess_board[r, c, d]:
                        continue
                    opposite = SecondAgent.a3_get_opposite_barrier(r, c, d)
                    if not SecondAgent.a3_does_wall_exist(chess_board, size, opposite[0], opposite[1], opposite[2]):
                        opposite = None
                    n1, n2 = SecondAgent.a3_get_wall_neighbors(chess_board, size, r, c, d)
                    if SecondAgent.is_testing:
                        print("-----")
                        print(str(r) + ", " + str(c) + ", " + str(d))
                        print(n1)
                        print(n2)
                    wall_set1 = None
                    wall_set2 = None
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
                    if SecondAgent.is_testing:
                        if wall_set1 is edge_set:
                            print("edge")
                        else:
                            print(wall_set1)
                        if wall_set2 is edge_set:
                            print("edge")
                        else:
                            print(wall_set2)
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
        return sorted(edge_set)  # TODO : WE DO NOT NEED TO SORT!!

    @staticmethod
    def a3_is_square_game_ending(chess_board, r, c, wall_set):
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
        top = SecondAgent.a3_does_wall_exist(chess_board, size, r, c, 0)
        right = SecondAgent.a3_does_wall_exist(chess_board, size, r, c, 1)
        left = SecondAgent.a3_does_wall_exist(chess_board, size, r, c, 3)
        bot = SecondAgent.a3_does_wall_exist(chess_board, size, r, c, 2)
        n_orthogonal = len([0 for i in (top, right, left, bot) if i])
        n_diagonal = len([0 for i in (top_left, top_right, bot_right, bot_left) if i])
        if SecondAgent.is_testing:
            print("--------")
            print(str(r) + ", " + str(c))
            print("top_left: " + str(top_left))
            print("top_right: " + str(top_right))
            print("bot_left: " + str(bot_left))
            print("bot_right: " + str(bot_right))
            print("top: " + str(top) + ", bot: " + str(bot) + ", left: " + str(left) + ", right: " + str(right))
            print("orth: " + str(n_orthogonal) + ", diag: " + str(n_diagonal))
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

    @staticmethod
    def a3_get_game_ending_squares(chess_board, my_pos, adv_pos, max_step, wall_set):
        # PROMISING SQUARES ------------------------------------------------------------------
        game_ending_squares = set()
        # ------------------------------------------------------------------------------------
        possible_squares = {my_pos}  # obviously my position is possible
        visiting = collections.deque()  # queue1 for bfs
        visiting.append(my_pos)
        next_visiting = collections.deque()  # queue2 for bfs
        if SecondAgent.a3_is_square_game_ending(chess_board, my_pos[0], my_pos[1], wall_set):
            game_ending_squares.add(my_pos)
        for i in range(max_step):  # run bfs at range max_step
            while len(visiting) > 0:  # visit every position at this level of bfs
                current_square = visiting.pop()
                r, c = current_square
                moves = ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1))  # u,r,d,l in order
                for direction in range(4):  # try to move in 4 directions
                    square = moves[direction]
                    if SecondAgent.ap_is_move_valid_possible_squares_helper(chess_board, current_square,
                                                                             square, direction,
                                                                             adv_pos, possible_squares):
                        possible_squares.add(square)  # this move is reachable
                        next_visiting.append(square)  # continue bfs at this move for the next iteration
                        # PROMISING SQUARES -----------------------------------------------
                        if SecondAgent.a3_is_square_game_ending(chess_board, square[0], square[1], wall_set):
                            game_ending_squares.add(square)
                        # ----------------------------------------------------------------
            temp = visiting  # old queue
            visiting = next_visiting  # update queue
            next_visiting = temp  # reuse the old empty queue to add new squares
        # PROMISING SQUARES ---------------------------------------------------------------
        return game_ending_squares
        # ---------------------------------------------------------------------------------

    @staticmethod
    def a3_get_possible_game_winning_moves(chess_board, my_pos, adv_pos, max_step):
        wall_set = wall_set = SecondAgent.a3_get_wall_set(chess_board)
        game_ending_squares = SecondAgent.a3_get_game_ending_squares(chess_board, my_pos, adv_pos, max_step, wall_set)
        possible_game_winning_squares = set()
        possible_game_winning_moves = []

        op_squares = {adv_pos}
        op_visiting = collections.deque()
        op_visiting.append(adv_pos)
        op_square_count = 1
        empty_set = set()

        while len(op_visiting) > 0:
            current_square = op_visiting.pop()
            r, c = current_square
            moves = ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1))  # u,r,d,l in order
            for direction in range(4):  # try to move in 4 directions
                if SecondAgent.ap3_is_move_valid_winning_move_helper_op(
                        chess_board, current_square, moves[direction], direction, op_squares, empty_set):
                    if (moves[direction] in game_ending_squares and
                            moves[direction] not in possible_game_winning_squares):
                        possible_game_winning_squares.add(moves[direction])
                        possible_game_winning_moves.append((moves[direction], SecondAgent.OPPOSITES[direction]))
                    if moves[direction] in game_ending_squares:
                        continue
                    op_square_count += 1
                    op_visiting.append(moves[direction])
                    op_squares.add(moves[direction])
        return possible_game_winning_moves, possible_game_winning_squares, op_squares, op_square_count

    @staticmethod
    def ap3_is_move_valid_winning_move_helper_op(chess_board, old_pos, new_pos, direction, my_squares, op_squares):
        old_r, old_c = old_pos  # row col
        new_r, new_c = new_pos  # row col
        if new_r >= chess_board.shape[0] or new_c >= chess_board.shape[1] or new_r < 0 or new_c < 0:
            return False  # is not on the board
        if new_pos in my_squares or new_pos in op_squares:  # already found
            return False  # is already found
        if chess_board[old_r, old_c, direction]:
            return False  # wall between old_pos, new_pos
        return True

    @staticmethod
    def a3_get_winning_move(chess_board, my_pos, adv_pos, max_step):
        possible_moves, possible_squares, op_squares, op_count = SecondAgent.a3_get_possible_game_winning_moves(
            chess_board, my_pos, adv_pos, max_step)
        empty_set = set()
        for possible_move in possible_moves:

            op_squares = op_squares.copy()
            op_visiting = collections.deque()
            for sq in possible_squares:
                if sq != possible_move[0]:
                    op_visiting.append(sq)
            op_square_count = op_count

            while len(op_visiting) > 0:
                current_square = op_visiting.pop()
                r, c = current_square
                moves = ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1))  # u,r,d,l in order
                for direction in range(4):  # try to move in 4 directions
                    if SecondAgent.ap3_is_move_valid_winning_move_helper_op(
                            chess_board, current_square, moves[direction], direction, op_squares, empty_set):
                        if moves[direction] == possible_move[0]:
                            continue
                        op_square_count += 1
                        op_visiting.append(moves[direction])
                        op_squares.add(moves[direction])

            my_squares = {possible_move[0]}
            my_visiting = collections.deque()
            my_visiting.append(possible_move[0])
            my_square_count = 1

            while len(my_visiting) > 0:
                current_square = my_visiting.pop()
                r, c = current_square
                moves = ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1))  # u,r,d,l in order
                for direction in range(4):  # try to move in 4 directions
                    if SecondAgent.ap3_is_move_valid_winning_move_helper_op(
                            chess_board, current_square, moves[direction], direction, my_squares, op_squares):
                        my_square_count += 1
                        my_visiting.append(moves[direction])
                        my_squares.add(moves[direction])

            if my_square_count > op_square_count:
                return possible_move
        return None

    # --------------------------------------------------------------------
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # SECTION B:

    @staticmethod
    def b_get_possible_walls(chess_board, pos):
        r, c = pos
        allowed_barriers = [i for i in range(0, 4) if not chess_board[r, c, i]]
        return allowed_barriers

    @staticmethod
    def b_get_possible_moves(chess_board, all_squares):
        all_moves = []
        for square in all_squares:
            for wall in SecondAgent.b_get_possible_walls(chess_board, square):
                all_moves.append((square, wall))
        return all_moves

    @staticmethod
    def b_get_best_moves(chess_board, my_pos, adv_pos, max_step):
        best_squares = SecondAgent.a_get_best_squares(chess_board, my_pos, adv_pos, max_step)
        return SecondAgent.b_get_possible_moves(chess_board, best_squares)

    # Moves (Up, Right, Down, Left)
    MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))
    # Opposite Directions
    OPPOSITES = {0: 2, 1: 3, 2: 0, 3: 1}

    @staticmethod
    def bp_set_barrier(chess_board, r, c, d):
        # Set the barrier to True
        chess_board[r, c, d] = True
        # Set the opposite barrier to True
        move = SecondAgent.MOVES[d]
        chess_board[r + move[0], c + move[1], SecondAgent.OPPOSITES[d]] = True

    @staticmethod
    def b_new_board(chess_board, move):
        new_chess_board = deepcopy(chess_board)
        (r, c), d = move
        SecondAgent.bp_set_barrier(new_chess_board, r, c, d)
        return new_chess_board

    # --------------------------------------------------------------------
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # SECTION C:

    def c_get_best_move(self, chess_board, my_pos, adv_pos, max_step):
        moves = self.b_get_best_moves(chess_board, my_pos, adv_pos, max_step)
        move_heuristics = {move: 0 for move in moves}
        op_pos = {move: [] for move in moves}
        best_move = moves[0]
        for move in moves:
            my_new_pos, _ = move
            new_board = self.b_new_board(chess_board, move)
            #
            h = self.a_heuristic(new_board, my_new_pos, adv_pos)  # FIXME extra heuristic check
            if h > 100:
                return move
            elif h < -100:
                move_heuristics[move] = -1000
                continue
            #
            op_positions = self.a_get_best_squares(chess_board, adv_pos, my_new_pos, max_step)
            op_heuristics = [self.a_heuristic(new_board, my_new_pos, i) for i in op_positions]
            move_heuristics[move] = min(op_heuristics)
            op_pos[move] = op_positions
        for move in moves:
            #
            if move_heuristics[move] < -100:
                continue
            #
            my_new_pos, _ = move
            board_after_my_move = self.b_new_board(chess_board, move)
            move_h = []
            for op_move in self.b_get_possible_moves(board_after_my_move, op_pos[move]):
                op_position, _ = op_move
                board_after_op_move = self.b_new_board(board_after_my_move, op_move)
                #
                h = self.a_heuristic(board_after_op_move, my_new_pos, op_position)  # FIXME extra heuristic check
                if h > 100:
                    move_h.append(1000)
                    continue
                elif h < -100:
                    move_h.append(-1000)
                    continue
                #
                my_2nd_pos = self.a_get_best_squares(board_after_op_move, my_new_pos, op_position, max_step)
                h_best_after_op = max([self.a_heuristic(board_after_op_move, i, op_position) for i in my_2nd_pos])
                move_h.append(h_best_after_op)
            move_heuristics[move] = min(move_h)
        for move in move_heuristics:
            if move_heuristics[move] > move_heuristics[best_move]:
                best_move = move
        return best_move

    # --------------------------------------------------------------------
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # SECTION C.2ish:

    def c2_get_best_move(self, chess_board, my_pos, adv_pos, max_step):
        best_positions, _ = self.a2_get_best_squares(chess_board, my_pos, adv_pos, max_step)
        moves = self.b_get_possible_moves(chess_board, best_positions)
        move_heuristics = {move: 0 for move in moves}
        op_pos = {move: [] for move in moves}
        best_move = moves[0]
        for move in moves:
            my_new_pos, _ = move
            new_board = self.b_new_board(chess_board, move)
            #
            h = self.a_heuristic(new_board, my_new_pos, adv_pos)  # FIXME extra heuristic check
            if h > 100:
                return move
            elif h < -100:
                move_heuristics[move] = -1000
                continue
            #
            op_positions, op_heuristics = self.a2_get_best_squares(chess_board, adv_pos, my_new_pos, max_step)
            move_heuristics[move] = min(op_heuristics)
            op_pos[move] = op_positions
        for move in moves:
            #
            if move_heuristics[move] < -100:
                continue
            #
            my_new_pos, _ = move
            board_after_my_move = self.b_new_board(chess_board, move)
            move_h = []
            for op_move in self.b_get_possible_moves(board_after_my_move, op_pos[move]):
                op_position, _ = op_move
                board_after_op_move = self.b_new_board(board_after_my_move, op_move)
                #
                h = self.a_heuristic(board_after_op_move, my_new_pos, op_position)  # FIXME extra heuristic check
                if h > 100:
                    move_h.append(1000)
                    continue
                elif h < -100:
                    move_h.append(-1000)
                    continue
                #
                my_2nd_pos, h_2nd = self.a2_get_best_squares(board_after_op_move, my_new_pos, op_position, max_step)
                h_best_after_op = max(h_2nd)
                move_h.append(h_best_after_op)
            move_heuristics[move] = min(move_h)
        for move in move_heuristics:
            if move_heuristics[move] > move_heuristics[best_move]:
                best_move = move
        return best_move

    # --------------------------------------------------------------------
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # SECTION C.3:

    def c3_leaf_max(self, max_step, moves):
        # return the h for moves. We are assuming that the base move is their move
        # (move, board, [sub_moves/sub_moves])
        move_heuristics = []
        for move in moves:  # these are our formatted moves
            if self.time_taken() > 1.9:  # FIXME: TIMEOUT
                return
            if move[2] is None:
                move_heuristics.append(0)
                continue
            elif move[2] == 1000 or move[2] == -1000:
                move_heuristics.append(move[2])
                continue
            sub_moves = self.b_get_possible_moves(move[1], move[2])
            formatted_sub_moves_h = []
            sub_move_heuristics = []
            for sub_move in sub_moves:  # these are opponent moves
                sub_board = self.b_new_board(move[1], sub_move)
                # my best squares in response
                if self.time_taken() > 1.9:  # FIXME: TIMEOUT
                    return
                winning_move = self.a3_get_winning_move(sub_board, move[0][0], sub_move[0], max_step)
                if self.time_taken() > 1.9:  # FIXME: TIMEOUT
                    return
                if winning_move is not None:
                    sub_move_h = 1000
                    sub_move_heuristics.append(sub_move_h)
                    formatted_sub_move = (sub_move, sub_board, sub_move_h * -1)
                    formatted_sub_moves_h.append((sub_move_h, formatted_sub_move))
                    continue
                if self.time_taken() > 1.9:  # FIXME: TIMEOUT
                    return
                var_time_safe = self.a2_get_best_squares(sub_board, move[0][0], sub_move[0], max_step)
                if self.time_taken() > 1.9:  # FIXME: TIMEOUT
                    return
                sub_squares, sub_square_hs = var_time_safe
                if sub_square_hs == 1000 or sub_square_hs == -1000 or sub_square_hs is None:
                    sub_move_h = 0 if sub_square_hs is None else sub_square_hs
                    sub_move_heuristics.append(sub_move_h)
                    formatted_sub_move = (sub_move, sub_board, None if sub_square_hs is None else sub_move_h * -1)
                    formatted_sub_moves_h.append((sub_move_h, formatted_sub_move))
                    continue
                sub_move_h = max(sub_square_hs)  # our best response is the H
                sub_move_heuristics.append(sub_move_h)
                formatted_sub_move = (sub_move, sub_board, sub_squares)
                formatted_sub_moves_h.append((sub_move_h, formatted_sub_move))
            formatted_sub_moves_h.sort()
            if len(formatted_sub_moves_h) >= 3:
                del formatted_sub_moves_h[3:]
            formatted_sub_moves = [i[1] for i in formatted_sub_moves_h]
            move[2][:] = formatted_sub_moves
            move_heuristics.append(min(sub_move_heuristics))  # the opponent's best reaction to our move
        return max(move_heuristics)  # the best of our moves

    def c3_recursive(self, max_step, moves, depth):  # high h is good for us
        if moves is None:
            return 0
        if depth == 0:
            if moves == 1000 or moves == -1000:
                return -moves
            return self.c3_leaf_max(max_step, moves)
        if depth % 2 == 0:  # we are in a max situation, max of our moves
            if moves == 1000 or moves == -1000:
                return -moves
            h = []
            for move in moves:
                if self.time_taken() > 1.9:  # FIXME: TIMEOUT
                    return
                h.append(self.c3_recursive(max_step, move[2], depth - 1))
                if self.time_taken() > 1.9:  # FIXME: TIMEOUT
                    return
            return max(h)
        else:  # we are in a min situation, min of their moves
            if moves == 1000 or moves == -1000:
                return moves
            h = []
            for move in moves:
                if self.time_taken() > 1.9:  # FIXME: TIMEOUT
                    return
                h.append(self.c3_recursive(max_step, move[2], depth - 1))
                if self.time_taken() > 1.9:  # FIXME: TIMEOUT
                    return
            return min(h)

    def c3_get_best(self, chess_board, my_pos, adv_pos, max_step):
        winning_move = self.a3_get_winning_move(chess_board, my_pos, adv_pos, max_step)
        if winning_move is not None:
            self.best_move = winning_move
            return
        if self.time_taken() > 1.9:  # FIXME: TIMEOUT
            return
        var_time_safe = self.a2_get_best_squares(chess_board, my_pos, adv_pos, max_step)
        if self.time_taken() > 1.9:  # FIXME: TIMEOUT
            return
        squares, _ = var_time_safe
        self.best_move = squares, self.b_get_possible_walls(chess_board, squares[0])[0]
        # depth 1
        dummy_moves = [((adv_pos,), chess_board, squares)]
        if self.time_taken() > 1.9:  # FIXME: TIMEOUT
            return
        self.c3_leaf_max(max_step, dummy_moves)
        if self.time_taken() > 1.9:  # FIXME: TIMEOUT
            return
        top_moves = dummy_moves[0][2]
        self.best_move = top_moves[0][0]
        # depth 2
        new_best_move = self.best_move
        max_h = -1000
        for move in top_moves:
            dummy_moves = [move]
            if self.time_taken() > 1.9:  # FIXME: TIMEOUT
                return
            h = self.c3_leaf_max(max_step, dummy_moves)
            if self.time_taken() > 1.9:  # FIXME: TIMEOUT
                return
            if h > max_h:
                new_best_move = move[0]
                max_h = h
        self.best_move = new_best_move
        # depth 3+
        depth = 0
        while self.time_taken() < 1.9 and depth <= 100:  # FIXME: TIMEOUT
            #if depth <= 6:
            #    print(depth)
            new_best_move = self.best_move
            if depth % 2 == 0:  # depth is even: get min heuristic
                min_h = 1000
                for move in top_moves:
                    if self.time_taken() > 1.9:  # FIXME: TIMEOUT
                        return
                    h = self.c3_recursive(max_step, move[2], depth)
                    if self.time_taken() > 1.9:  # FIXME: TIMEOUT
                        return
                    if h < min_h:
                        new_best_move = move[0]
                        min_h = h
            else:  # depth is odd: get max heuristic
                max_h = -1000
                for move in top_moves:
                    if self.time_taken() > 1.9:  # FIXME: TIMEOUT
                        return
                    h = self.c3_recursive(max_step, move[2], depth)
                    if self.time_taken() > 1.9:  # FIXME: TIMEOUT
                        return
                    if h > max_h:
                        new_best_move = move[0]
                        max_h = h
            self.best_move = new_best_move
            depth += 1

    # --------------------------------------------------------------------
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # SECTION T:

    @staticmethod
    def t_print_all_barriers(chess_board):
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
        #barriers2 = sorted(barriers2)
        #for barrier in barriers2:
        #    p, d = barrier
        #    print("bar: " + str(p) + " " + ("d" if d == 2 else "l"))

    @staticmethod
    def t_print_info(chess_board, my_pos, adv_pos, max_step):
        # max step, player pos, op pos, board size.
        print("max step: " + str(max_step) + ", board_size: " + str(chess_board.shape[0]) +
              ", my_pos: " + str(my_pos) + ", op_pos: " + str(adv_pos))

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # MY CODE UP^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    def __init__(self):
        super(SecondAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        # MY CODE DOWN----------------------------------------------------
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        self.best_move = None
        self.start_time = 0
        self.has_started = False
        self.is_testing = False
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
        # MY CODE DOWN--------------------------------------------------------
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

        testing = False  # enable this for testing
        if not self.has_started and testing:
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        self.has_started = True

        if testing:
            SecondAgent.t_print_info(chess_board, my_pos, adv_pos, max_step)
            SecondAgent.t_print_all_barriers(chess_board)
            #----
            #TEST CODE
            #print(self.a3_get_wall_set(chess_board))


        #self.c3_get_best(chess_board, my_pos, adv_pos, max_step)

        time1 = time.time()
        SecondAgent.is_testing = True
        wall_set = self.a3_get_wall_set(chess_board)
        SecondAgent.is_testing = False
        game_ending = self.a3_get_game_ending_squares(chess_board, my_pos, adv_pos, max_step, wall_set)

        possible_moves, possible_squares, op_squares, op_count = SecondAgent.a3_get_possible_game_winning_moves(
            chess_board, my_pos, adv_pos, max_step)
        #SecondAgent.is_testing = True
        #for sq in game_ending:
        #    self.a3_is_square_game_ending(chess_board, sq[0], sq[1], wall_set)
        #SecondAgent.is_testing = False
        #print(SecondAgent.a3_get_wall_neighbors(chess_board, chess_board.shape[0], my_pos[0], my_pos[1], 0))
        #print(SecondAgent.a3_get_wall_neighbors(chess_board, chess_board.shape[0], my_pos[0], my_pos[1], 3))
        #print(wall_set)
        print(game_ending)
        print(possible_squares)
        print(self.a3_get_winning_move(chess_board, my_pos, adv_pos, max_step))
        #for i in range(25):
        #    SecondAgent.a3_get_winning_move(chess_board, my_pos, adv_pos, max_step)
        #print(SecondAgent.a3_get_winning_move(chess_board, my_pos, adv_pos, max_step))
        #print(time.time() - time1)

        self.c3_get_best(chess_board, my_pos, adv_pos, max_step)
        return self.best_move
            #----

        #move = self.c2_get_best_move(chess_board, my_pos, adv_pos, max_step)  # when testing this can be commented out to not waste time

        #print(move)
        #return move
        if testing:
            return my_pos, SecondAgent.b_get_possible_walls(chess_board, my_pos)[0]

        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # MY CODE UP^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    '''

# Student agent: Add your own agent here
import collections

from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time


@register_agent("second_agent")
class SecondAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """
    # MY CODE BELOW-------------------------------------------------------
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # SECTION I:

    # Moves (Up, Right, Down, Left)
    MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))
    # Opposite Directions
    OPPOSITES = {0: 2, 1: 3, 2: 0, 3: 1}

    EMPTY_SET = set()

    def time_taken(self):  # this is used for testing purposes
        return time.time() - self.start_time

    def timeout(self):
        return (time.time() - self.start_time) > 1.9

    # --------------------------------------------------------------------
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # SECTION A, version 2:

    def ap2_get_distance(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        return abs(r1 - r2) + abs(c1 - c2)

    def ah2_is_move_valid_promising_squares_helper(
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

    def ap2_get_promising_squares(self, chess_board, my_pos, adv_pos, max_step):
        # PROMISING SQUARES ------------------------------------------------------------------
        furthest_right = furthest_left = furthest_up = furthest_down = my_pos
        opponent_right = opponent_left = opponent_up = opponent_down = my_pos
        middle_closest = my_pos
        middle_of_the_board = (int(chess_board.shape[0] / 2), int(chess_board.shape[1] / 2))
        # ------------------------------------------------------------------------------------
        possible_squares = {my_pos}  # obviously my position is possible
        visiting = collections.deque()  # queue1 for bfs
        visiting.append(my_pos)
        next_visiting = collections.deque()  # queue2 for bfs
        for i in range(max_step):  # run bfs at range max_step
            while len(visiting) > 0:  # visit every position at this level of bfs
                if self.timeout():  # FIXME: TIMEOUT
                    return
                current_square = visiting.pop()
                r, c = current_square
                moves = ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1))  # u,r,d,l in order
                for direction in range(4):  # try to move in 4 directions
                    square = moves[direction]
                    if self.ah2_is_move_valid_promising_squares_helper(chess_board, current_square,
                                                                       square, direction,
                                                                       adv_pos, possible_squares):
                        possible_squares.add(square)  # this move is reachable
                        next_visiting.append(square)  # continue bfs at this move for the next iteration
                        # PROMISING SQUARES -----------------------------------------------
                        if (self.ap2_get_distance(middle_of_the_board, square) <
                                self.ap2_get_distance(middle_of_the_board, middle_closest)):
                            middle_closest = square
                        # ---
                        if (square[0] < adv_pos[0] and square[1] >= adv_pos[1] and
                                (self.ap2_get_distance(adv_pos, square) <=
                                 self.ap2_get_distance(adv_pos, opponent_up))):
                            opponent_up = square
                        elif (square[0] >= adv_pos[0] and square[1] > adv_pos[1] and
                                (self.ap2_get_distance(adv_pos, square) <=
                                 self.ap2_get_distance(adv_pos, opponent_right))):
                            opponent_right = square
                        elif (square[0] > adv_pos[0] and square[1] <= adv_pos[1] and
                                (self.ap2_get_distance(adv_pos, square) <=
                                 self.ap2_get_distance(adv_pos, opponent_down))):
                            opponent_down = square
                        elif (square[0] <= adv_pos[0] and square[1] < adv_pos[1] and
                                (self.ap2_get_distance(adv_pos, square) <=
                                 self.ap2_get_distance(adv_pos, opponent_left))):
                            opponent_left = square
                        # ---
                        if (square[0] < my_pos[0] and square[1] >= my_pos[1] and
                                (self.ap2_get_distance(my_pos, square) >
                                 self.ap2_get_distance(my_pos, furthest_up))):
                            furthest_up = square
                        elif (square[0] >= my_pos[0] and square[1] > my_pos[1] and
                                (self.ap2_get_distance(my_pos, square) >
                                 self.ap2_get_distance(my_pos, furthest_right))):
                            furthest_right = square
                        elif (square[0] > my_pos[0] and square[1] <= my_pos[1] and
                                (self.ap2_get_distance(my_pos, square) >
                                 self.ap2_get_distance(my_pos, furthest_down))):
                            furthest_down = square
                        elif (square[0] <= my_pos[0] and square[1] < my_pos[1] and
                                (self.ap2_get_distance(my_pos, square) >
                                 self.ap2_get_distance(my_pos, furthest_left))):
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

    def ah2_is_move_valid_best_squares_helper(
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

    def a2_get_best_squares(self, chess_board, my_pos, adv_pos, max_step):
        if self.timeout():  # FIXME: TIMEOUT
            return
        promising_positions = self.ap2_get_promising_squares(chess_board, my_pos, adv_pos, max_step)
        if self.timeout():  # FIXME: TIMEOUT
            return
        number_of_positions = len(promising_positions)
        my_squares_per_position = [{pos} for pos in promising_positions]
        my_visiting_per_position = [collections.deque() for _ in promising_positions]
        for i in range(number_of_positions):
            my_visiting_per_position[i].append(promising_positions[i])
        my_next_visiting_per_position = [collections.deque() for _ in promising_positions]
        my_square_count_per_position = [1 for _ in range(number_of_positions)]
        neutral_square_count_per_position = [0 for _ in range(number_of_positions)]

        op_squares = {adv_pos}
        op_squares_preview = {adv_pos}
        op_visiting = collections.deque()
        op_visiting.append(adv_pos)
        op_next_visiting = collections.deque()
        op_square_count = 1

        while True:
            # PART 1, get next squares
            while len(op_visiting) > 0:
                if self.timeout():  # FIXME: TIMEOUT
                    return
                current_square = op_visiting.pop()
                r, c = current_square
                moves = ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1))  # u,r,d,l in order
                for direction in range(4):  # try to move in 4 directions
                    if self.ah2_is_move_valid_best_squares_helper(chess_board, current_square, moves[direction],
                                                                  direction, op_squares_preview, self.EMPTY_SET):
                        op_square_count += 1
                        op_next_visiting.append(moves[direction])
                        op_squares_preview.add(moves[direction])
            for i in range(number_of_positions):
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
                        if self.ah2_is_move_valid_best_squares_helper(chess_board, current_square,
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
    # SECTION Z:

    def zp3_get_opposite_barrier(self, r, c, d):
        move = self.MOVES[d]
        return r + move[0], c + move[1], self.OPPOSITES[d]

    def zp3_does_wall_exist(self, chess_board, size, r, c, d):
        return 0 <= r < size and 0 <= c < size and chess_board[r, c, d]

    def zp3_get_wall_neighbors(self, chess_board, size, r, c, d):
        if d == 0:
            left_neighbors = []
            if self.zp3_does_wall_exist(chess_board, size, r, c, 3):
                left_neighbors.append((r, c, 3))
            if self.zp3_does_wall_exist(chess_board, size, r - 1, c, 3):
                left_neighbors.append((r-1, c, 3))
            if self.zp3_does_wall_exist(chess_board, size, r, c - 1, 0):
                left_neighbors.append((r, c-1, 0))
            right_neighbors = []
            if self.zp3_does_wall_exist(chess_board, size, r, c, 1):
                right_neighbors.append((r, c, 1))
            if self.zp3_does_wall_exist(chess_board, size, r - 1, c, 1):
                right_neighbors.append((r-1, c, 1))
            if self.zp3_does_wall_exist(chess_board, size, r, c + 1, 0):
                right_neighbors.append((r, c+1, 0))
            return left_neighbors, right_neighbors
        elif d == 3:
            top_neighbors = []
            if self.zp3_does_wall_exist(chess_board, size, r, c, 0):
                top_neighbors.append((r, c, 0))
            if self.zp3_does_wall_exist(chess_board, size, r, c - 1, 0):
                top_neighbors.append((r, c-1, 0))
            if self.zp3_does_wall_exist(chess_board, size, r - 1, c, 3):
                top_neighbors.append((r-1, c, 3))
            bot_neighbors = []
            if self.zp3_does_wall_exist(chess_board, size, r, c, 2):
                bot_neighbors.append((r, c, 2))
            if self.zp3_does_wall_exist(chess_board, size, r, c - 1, 2):
                bot_neighbors.append((r, c-1, 2))
            if self.zp3_does_wall_exist(chess_board, size, r + 1, c, 3):
                bot_neighbors.append((r+1, c, 3))
            return top_neighbors, bot_neighbors

    def zp3_get_wall_set(self, chess_board):
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
        for r in range(size):
            for c in range(size):
                if self.timeout():  # FIXME: TIMEOUT
                    return
                for d in (0, 3):
                    if (r, c, d) in edge_set or not chess_board[r, c, d]:
                        continue
                    opposite = self.zp3_get_opposite_barrier(r, c, d)
                    if not self.zp3_does_wall_exist(chess_board, size, opposite[0], opposite[1], opposite[2]):
                        opposite = None
                    n1, n2 = self.zp3_get_wall_neighbors(chess_board, size, r, c, d)
                    wall_set1 = None
                    wall_set2 = None
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

    def zp3_is_square_game_ending(self, chess_board, r, c, wall_set):
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
        top = self.zp3_does_wall_exist(chess_board, size, r, c, 0)
        right = self.zp3_does_wall_exist(chess_board, size, r, c, 1)
        left = self.zp3_does_wall_exist(chess_board, size, r, c, 3)
        bot = self.zp3_does_wall_exist(chess_board, size, r, c, 2)
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

    def zh3_is_move_valid_game_ending_squares_helper(
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

    def zp3_get_game_ending_squares(self, chess_board, my_pos, adv_pos, max_step, wall_set):
        # PROMISING SQUARES ------------------------------------------------------------------
        game_ending_squares = set()
        # ------------------------------------------------------------------------------------
        possible_squares = {my_pos}  # obviously my position is possible
        visiting = collections.deque()  # queue1 for bfs
        visiting.append(my_pos)
        next_visiting = collections.deque()  # queue2 for bfs
        if self.zp3_is_square_game_ending(chess_board, my_pos[0], my_pos[1], wall_set):
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
                    if self.zh3_is_move_valid_game_ending_squares_helper(chess_board, current_square,
                                                                         square, direction,
                                                                         adv_pos, possible_squares):
                        possible_squares.add(square)  # this move is reachable
                        next_visiting.append(square)  # continue bfs at this move for the next iteration
                        # PROMISING SQUARES -----------------------------------------------
                        if self.zp3_is_square_game_ending(chess_board, square[0], square[1], wall_set):
                            game_ending_squares.add(square)
                        # ----------------------------------------------------------------
            temp = visiting  # old queue
            visiting = next_visiting  # update queue
            next_visiting = temp  # reuse the old empty queue to add new squares
        # PROMISING SQUARES ---------------------------------------------------------------
        return game_ending_squares
        # ---------------------------------------------------------------------------------

    def zh3_is_move_valid_possible_move_helper(
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

    def zp3_get_possible_game_winning_moves(self, chess_board, my_pos, adv_pos, max_step):
        if self.timeout():  # FIXME: TIMEOUT
            return
        wall_set = self.zp3_get_wall_set(chess_board)
        if self.timeout():  # FIXME: TIMEOUT
            return
        game_ending_squares = self.zp3_get_game_ending_squares(chess_board, my_pos, adv_pos, max_step, wall_set)
        if self.timeout():  # FIXME: TIMEOUT
            return
        possible_game_winning_squares = set()
        possible_game_winning_moves = []

        op_squares = {adv_pos}
        op_visiting = collections.deque()
        op_visiting.append(adv_pos)
        op_square_count = 1

        while len(op_visiting) > 0:
            if self.timeout():  # FIXME: TIMEOUT
                return
            current_square = op_visiting.pop()
            r, c = current_square
            moves = ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1))  # u,r,d,l in order
            for direction in range(4):  # try to move in 4 directions
                if self.zh3_is_move_valid_possible_move_helper(
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

    def zh3_is_move_valid_winning_move_helper_op(
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

    def z3_get_winning_move(self, chess_board, my_pos, adv_pos, max_step):
        if self.timeout():  # FIXME: TIMEOUT
            return
        var_time_safe = self.zp3_get_possible_game_winning_moves(
            chess_board, my_pos, adv_pos, max_step)
        if self.timeout():  # FIXME: TIMEOUT
            return
        possible_moves, possible_squares, op_squares, op_count = var_time_safe
        for possible_move in possible_moves:
            if self.timeout():  # FIXME: TIMEOUT
                return

            op_squares = op_squares.copy()
            op_visiting = collections.deque()
            for sq in possible_squares:
                if sq != possible_move[0]:
                    op_visiting.append(sq)
            op_square_count = op_count

            while len(op_visiting) > 0:
                if self.timeout():  # FIXME: TIMEOUT
                    return
                current_square = op_visiting.pop()
                r, c = current_square
                moves = ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1))  # u,r,d,l in order
                for direction in range(4):  # try to move in 4 directions
                    if self.zh3_is_move_valid_winning_move_helper_op(
                            chess_board, current_square, moves[direction], direction, op_squares, self.EMPTY_SET):
                        if moves[direction] == possible_move[0]:
                            continue
                        op_square_count += 1
                        op_visiting.append(moves[direction])
                        op_squares.add(moves[direction])

            my_squares = {possible_move[0]}
            my_visiting = collections.deque()
            my_visiting.append(possible_move[0])
            my_square_count = 1

            while len(my_visiting) > 0:
                if self.timeout():  # FIXME: TIMEOUT
                    return
                current_square = my_visiting.pop()
                r, c = current_square
                moves = ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1))  # u,r,d,l in order
                for direction in range(4):  # try to move in 4 directions
                    if self.zh3_is_move_valid_winning_move_helper_op(
                            chess_board, current_square, moves[direction], direction, my_squares, op_squares):
                        my_square_count += 1
                        my_visiting.append(moves[direction])
                        my_squares.add(moves[direction])

            if my_square_count > op_square_count:
                return possible_move
        return None

    # --------------------------------------------------------------------
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # SECTION B:

    def b_get_possible_walls(self, chess_board, pos):
        r, c = pos
        allowed_barriers = [i for i in range(0, 4) if not chess_board[r, c, i]]
        return allowed_barriers

    def b_get_possible_moves(self, chess_board, all_squares):
        all_moves = []
        for square in all_squares:
            for wall in self.b_get_possible_walls(chess_board, square):
                all_moves.append((square, wall))
        return all_moves

    def bp_set_barrier(self, chess_board, r, c, d):
        # Set the barrier to True
        chess_board[r, c, d] = True
        # Set the opposite barrier to True
        move = self.MOVES[d]
        chess_board[r + move[0], c + move[1], self.OPPOSITES[d]] = True

    def b_new_board(self, chess_board, move):
        new_chess_board = deepcopy(chess_board)
        (r, c), d = move
        self.bp_set_barrier(new_chess_board, r, c, d)
        return new_chess_board

    # --------------------------------------------------------------------
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # SECTION C, version 3:

    def c3_leaf_max(self, max_step, moves):
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
            sub_moves = self.b_get_possible_moves(move[1], move[2])
            formatted_sub_moves_h = []
            sub_move_heuristics = []
            for sub_move in sub_moves:  # these are opponent moves
                sub_board = self.b_new_board(move[1], sub_move)
                # my best squares in response
                if self.timeout():  # FIXME: TIMEOUT
                    return
                winning_move = self.z3_get_winning_move(sub_board, move[0][0], sub_move[0], max_step)
                if self.timeout():  # FIXME: TIMEOUT
                    return
                if winning_move is not None:
                    sub_move_h = 1000
                    sub_move_heuristics.append(sub_move_h)
                    formatted_sub_move = (sub_move, sub_board, sub_move_h * -1)
                    formatted_sub_moves_h.append((sub_move_h, formatted_sub_move))
                    continue
                if self.timeout():  # FIXME: TIMEOUT
                    return
                var_time_safe = self.a2_get_best_squares(sub_board, move[0][0], sub_move[0], max_step)
                if self.timeout():  # FIXME: TIMEOUT
                    return
                sub_squares, sub_square_hs = var_time_safe
                if sub_square_hs == 1000 or sub_square_hs == -1000 or sub_square_hs is None:
                    sub_move_h = 0 if sub_square_hs is None else sub_square_hs
                    sub_move_heuristics.append(sub_move_h)
                    formatted_sub_move = (sub_move, sub_board, None if sub_square_hs is None else sub_move_h * -1)
                    formatted_sub_moves_h.append((sub_move_h, formatted_sub_move))
                    continue
                sub_move_h = max(sub_square_hs)  # our best response is the H
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

    def c3_recursive(self, max_step, moves, depth):  # high h is good for us
        if moves is None:
            return 0
        if depth == 0:
            if moves == 1000 or moves == -1000:
                return -moves
            return self.c3_leaf_max(max_step, moves)
        if depth % 2 == 0:  # we are in a max situation, max of our moves
            if moves == 1000 or moves == -1000:
                return -moves
            h = []
            for move in moves:
                if self.timeout():  # FIXME: TIMEOUT
                    return
                h.append(self.c3_recursive(max_step, move[2], depth - 1))
                if self.timeout():  # FIXME: TIMEOUT
                    return
            return max(h)
        else:  # we are in a min situation, min of their moves
            if moves == 1000 or moves == -1000:
                return moves
            h = []
            for move in moves:
                if self.timeout():  # FIXME: TIMEOUT
                    return
                h.append(self.c3_recursive(max_step, move[2], depth - 1))
                if self.timeout():  # FIXME: TIMEOUT
                    return
            return min(h)

    def c3_get_best(self, chess_board, my_pos, adv_pos, max_step):
        if self.timeout():  # FIXME: TIMEOUT
            return
        winning_move = self.z3_get_winning_move(chess_board, my_pos, adv_pos, max_step)
        if self.timeout():  # FIXME: TIMEOUT
            return
        if winning_move is not None:
            self.best_move = winning_move
            return
        if self.timeout():  # FIXME: TIMEOUT
            return
        var_time_safe = self.a2_get_best_squares(chess_board, my_pos, adv_pos, max_step)
        if self.timeout():  # FIXME: TIMEOUT
            return
        squares, _ = var_time_safe
        self.best_move = squares, self.b_get_possible_walls(chess_board, squares[0])[0]
        # depth 1
        dummy_moves = [((adv_pos,), chess_board, squares)]
        if self.timeout():  # FIXME: TIMEOUT
            return
        self.c3_leaf_max(max_step, dummy_moves)
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
            h = self.c3_leaf_max(max_step, dummy_moves)
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
            if depth % 2 == 0:  # depth is even: get min heuristic
                min_h = 1000
                for move in top_moves:
                    if self.timeout():  # FIXME: TIMEOUT
                        return
                    h = self.c3_recursive(max_step, move[2], depth)
                    if self.timeout():  # FIXME: TIMEOUT
                        return
                    if h < min_h:
                        new_best_move = move[0]
                        min_h = h
            else:  # depth is odd: get max heuristic
                max_h = -1000
                for move in top_moves:
                    if self.timeout():  # FIXME: TIMEOUT
                        return
                    h = self.c3_recursive(max_step, move[2], depth)
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
        super(SecondAgent, self).__init__()
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

        self.c3_get_best(chess_board, my_pos, adv_pos, max_step)
        return self.best_move

        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # MY CODE UP^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
