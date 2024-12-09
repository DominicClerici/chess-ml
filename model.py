import chess
import random
import math
from collections import defaultdict
import time

class Node:
    def __init__(self, board, parent=None, move=None):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = list(board.legal_moves)
        
    def add_child(self, move):
        new_board = self.board.copy()
        new_board.push(move)
        child = Node(new_board, parent=self, move=move)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child
        
    def ucb1(self, exploration=1.41):
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + exploration * math.sqrt(math.log(self.parent.visits) / self.visits)

class ChessAI:
    def __init__(self, simulation_time=3):
        self.simulation_time = simulation_time
        
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        self.pst = {
            chess.PAWN: [
                0,  0,  0,  0,  0,  0,  0,  0,
                50, 50, 50, 50, 50, 50, 50, 50,
                10, 10, 20, 30, 30, 20, 10, 10,
                5,  5, 10, 25, 25, 10,  5,  5,
                0,  0,  0, 20, 20,  0,  0,  0,
                5, -5,-10,  0,  0,-10, -5,  5,
                5, 10, 10,-20,-20, 10, 10,  5,
                0,  0,  0,  0,  0,  0,  0,  0
            ],
            chess.KNIGHT: [
                -50,-40,-30,-30,-30,-30,-40,-50,
                -40,-20,  0,  0,  0,  0,-20,-40,
                -30,  0, 10, 15, 15, 10,  0,-30,
                -30,  5, 15, 20, 20, 15,  5,-30,
                -30,  0, 15, 20, 20, 15,  0,-30,
                -30,  5, 10, 15, 15, 10,  5,-30,
                -40,-20,  0,  5,  5,  0,-20,-40,
                -50,-40,-30,-30,-30,-30,-40,-50
            ],
            chess.BISHOP: [
                -20,-10,-10,-10,-10,-10,-10,-20,
                -10,  0,  0,  0,  0,  0,  0,-10,
                -10,  0,  5, 10, 10,  5,  0,-10,
                -10,  5,  5, 10, 10,  5,  5,-10,
                -10,  0, 10, 10, 10, 10,  0,-10,
                -10, 10, 10, 10, 10, 10, 10,-10,
                -10,  5,  0,  0,  0,  0,  5,-10,
                -20,-10,-10,-10,-10,-10,-10,-20
            ]
        }

        self.opening_moves = {
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1": [
                "e2e4", "d2d4"  # Common first moves
            ],
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1": [
                "e7e5", "c7c5"  # Responses to e4
            ],
            "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1": [
                "d7d5", "g8f6"  # Responses to d4
            ]
        }

    def evaluate_piece_position(self, board, piece, square):
        """Evaluate the position of a specific piece"""
        if piece.piece_type not in self.pst:
            return 0
            
        if piece.color == chess.WHITE:
            return self.pst[piece.piece_type][square]
        else:
            return -self.pst[piece.piece_type][chess.square_mirror(square)]

    def evaluate_position(self, board):
        if board.is_checkmate():
            return -20000 if board.turn else 20000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
            
        score = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    score += value
                else:
                    score -= value
                    
                score += self.evaluate_piece_position(board, piece, square)
        
        w_king_square = board.king(chess.WHITE)
        b_king_square = board.king(chess.BLACK)
        if w_king_square:
            score += self._evaluate_king_safety(board, w_king_square, chess.WHITE)
        if b_king_square:
            score -= self._evaluate_king_safety(board, b_king_square, chess.BLACK)
        
        w_mobility = len(list(board.legal_moves))
        board.turn = chess.BLACK
        b_mobility = len(list(board.legal_moves))
        board.turn = chess.WHITE
        score += (w_mobility - b_mobility) * 5
        
        score += self._evaluate_pawn_structure(board)
        
        return score

    def _evaluate_king_safety(self, board, king_square, color):
        score = 0
        
        rank = chess.square_rank(king_square)
        file = chess.square_file(king_square)
        
        for rank_offset in [-1, 0, 1]:
            for file_offset in [-1, 0, 1]:
                if rank_offset == 0 and file_offset == 0:
                    continue
                
                new_rank = rank + rank_offset
                new_file = file + file_offset
                
                if 0 <= new_rank < 8 and 0 <= new_file < 8:
                    square = chess.square(new_file, new_rank)
                    if board.is_attacked_by(not color, square):
                        score -= 10
            
        # Bonus for pawn shield
        shield_squares = []
        if color == chess.WHITE and rank < 7:
            # in front
            for f in range(max(0, file - 1), min(8, file + 2)):
                shield_squares.append(chess.square(f, rank + 1))
        elif color == chess.BLACK and rank > 0:
            # behind
            for f in range(max(0, file - 1), min(8, file + 2)):
                shield_squares.append(chess.square(f, rank - 1))
        
        for square in shield_squares:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                score += 15
        
        return score

    def _evaluate_pawn_structure(self, board):
        score = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                if self._is_passed_pawn(board, square, piece.color):
                    score += 50 if piece.color == chess.WHITE else -50
        
        return score

    def _is_passed_pawn(self, board, square, color):
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        
        if color == chess.WHITE:
            for r in range(rank + 1, 8):
                for f in range(max(0, file - 1), min(8, file + 2)):
                    s = chess.square(f, r)
                    p = board.piece_at(s)
                    if p and p.piece_type == chess.PAWN and p.color == chess.BLACK:
                        return False
        else:
            for r in range(rank - 1, -1, -1):
                for f in range(max(0, file - 1), min(8, file + 2)):
                    s = chess.square(f, r)
                    p = board.piece_at(s)
                    if p and p.piece_type == chess.PAWN and p.color == chess.WHITE:
                        return False
        return True

    def get_move_from_opening_book(self, board):
        """Try to get a move from the opening book"""
        fen = board.fen().split(' ')[0] 
        if fen in self.opening_moves:
            legal_book_moves = [move for move in self.opening_moves[fen] 
                              if chess.Move.from_uci(move) in board.legal_moves]
            if legal_book_moves:
                return chess.Move.from_uci(random.choice(legal_book_moves))
        return None

    def select_node(self, node):
        while not node.board.is_game_over() and not node.untried_moves:
            if not node.children:
                return node
            node = max(node.children, key=lambda n: n.ucb1())
        return node

    def expand_node(self, node):
        if node.untried_moves and not node.board.is_game_over():
            moves = []
            for move in node.untried_moves:
                node.board.push(move)
                eval = self.evaluate_position(node.board)
                node.board.pop()
                moves.append((move, eval))
            
            # top 3 moves
            moves.sort(key=lambda x: x[1], reverse=node.board.turn)
            selected_move = random.choice(moves[:3])[0]
            return node.add_child(selected_move)
        return node

    def simulate(self, board):
        """Smart simulation with basic tactics"""
        board = board.copy()
        playout_depth = 50
        
        for _ in range(playout_depth):
            if board.is_game_over():
                break
                
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
                
            if random.random() < 0.9:  # 90% of the time use evaluation
                best_moves = []
                best_eval = float('-inf')
                
                for move in legal_moves:
                    board.push(move)
                    eval = -self.evaluate_position(board)
                    board.pop()
                    
                    if eval > best_eval:
                        best_moves = [move]
                        best_eval = eval
                    elif eval == best_eval:
                        best_moves.append(move)
                
                board.push(random.choice(best_moves))
            else:  # 10% random moves for exploration
                board.push(random.choice(legal_moves))
        
        return self.evaluate_position(board)

    def backpropagate(self, node, result):
        while node:
            node.visits += 1
            node.wins += (math.tanh(result / 1000) + 1) / 2
            result = -result
            node = node.parent

    def get_best_move(self, board):
     
        book_move = self.get_move_from_opening_book(board)
        if book_move:
            return book_move
            
        root = Node(board)
        end_time = time.time() + self.simulation_time
        

        while time.time() < end_time:
            node = self.select_node(root)
            node = self.expand_node(node)
            result = self.simulate(node.board)
            self.backpropagate(node, result)
        
        if root.children:
            return max(root.children, key=lambda c: c.visits).move
            
        return random.choice(list(board.legal_moves))

    def set_simulation_time(self, seconds):
        self.simulation_time = seconds