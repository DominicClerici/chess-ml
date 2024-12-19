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
        
    def ucb1(self, exploration=0.7):
        if self.visits == 0:
            return float('inf')
        # Add progressive bias term
        if self.parent:
            move_value = self._get_move_value(self.move, self.parent.board)
            return (self.wins / self.visits) + \
                   exploration * math.sqrt(math.log(self.parent.visits) / self.visits) + \
                   move_value / (self.visits + 1)
        return float('inf')

    def _get_move_value(self, move, board):
        """Heuristic value of a move"""
        if move is None:
            return 0
        
        value = 0
        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            if captured_piece:
                value += 10 

        board.push(move)
        if board.is_check():
            value += 5
        board.pop()
        
        return value

class ChessAI:
    def __init__(self, simulation_time=10):
        self.simulation_time = simulation_time
        self.metrics = {
            'total_games_played': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'average_moves_per_game': [],
            'average_time_per_move': [],
            'nodes_explored_per_move': [],
            'capture_moves_played': 0,
            'check_moves_played': 0,
            'opening_book_moves_used': 0
        }
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
                ("e2e4", 0.6),  # King's pawn
                ("d2d4", 0.3),  # Queen's pawn
                ("c2c4", 0.1)   # English Opening
            ],
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1": [
                ("e7e5", 0.5),    # Open game
                ("c7c5", 0.3),    # Sicilian Defense
                ("e7e6", 0.1),    # French Defense
                ("c7c6", 0.1)     # Caro-Kann
            ],
            "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": [
                ("d2d4", 0.6),
                ("d2d3", 0.2),
                ("g1f3", 0.2)
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

        score += self._evaluate_tactical_patterns(board)
        
        score += self._evaluate_center_control(board)
        score += self._evaluate_pawn_structure(board)
        
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
            
        shield_squares = []
        if color == chess.WHITE and rank < 7:
            for f in range(max(0, file - 1), min(8, file + 2)):
                shield_squares.append(chess.square(f, rank + 1))
        elif color == chess.BLACK and rank > 0:
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
        """Get a weighted random move from the opening book"""
        fen = board.fen().split(' ')[0]
        if fen in self.opening_moves:
            legal_book_moves = [(move, weight) for move, weight in self.opening_moves[fen] 
                              if chess.Move.from_uci(move) in board.legal_moves]
            if legal_book_moves:
                total_weight = sum(weight for _, weight in legal_book_moves)
                choice = random.uniform(0, total_weight)
                current = 0
                for move, weight in legal_book_moves:
                    current += weight
                    if current >= choice:
                        return chess.Move.from_uci(move)
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
            
            moves.sort(key=lambda x: x[1], reverse=node.board.turn)
            selected_move = random.choice(moves[:3])[0]
            return node.add_child(selected_move)
        return node

    def simulate(self, board):
        """Smart simulation with improved piece protection"""
        board = board.copy()
        playout_depth = 50
        
        for _ in range(playout_depth):
            if board.is_game_over():
                break
            
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            
            if random.random() < 0.99:
                best_moves = []
                best_eval = float('-inf')
                
                for move in legal_moves:
                    board.push(move)
                    
                    
                    piece_safety_penalty = 0
                    for square in chess.SQUARES:
                        piece = board.piece_at(square)
                        if piece and piece.color == board.turn:
                            if board.is_attacked_by(not board.turn, square):
                                defenders = len(board.attackers(board.turn, square))
                                attackers = len(board.attackers(not board.turn, square))
                                if attackers > defenders:
                                    piece_safety_penalty -= self.piece_values[piece.piece_type] * 2.5
                                elif attackers == defenders and piece.piece_type != chess.PAWN:
                                    piece_safety_penalty -= self.piece_values[piece.piece_type] * 0.3
                    
                    eval = -self.evaluate_position(board) + piece_safety_penalty
                    board.pop()
                    
                    if eval > best_eval:
                        best_moves = [move]
                        best_eval = eval
                    elif eval == best_eval:
                        best_moves.append(move)
                
                board.push(random.choice(best_moves))
            else:
                board.push(random.choice(legal_moves))
        
        return self.evaluate_position(board)

    def backpropagate(self, node, result):
        while node:
            node.visits += 1
            node.wins += (math.tanh(result / 1000) + 1) / 2
            result = -result
            node = node.parent

    def get_best_move(self, board):
        start_time = time.time()
        nodes_explored = 0
        
        book_move = self.get_move_from_opening_book(board)
        if book_move:
            self.metrics['opening_book_moves_used'] += 1
            return book_move
            
        root = Node(board)
        end_time = time.time() + self.simulation_time
        
        while time.time() < end_time:
            node = self.select_node(root)
            node = self.expand_node(node)
            result = self.simulate(node.board)
            self.backpropagate(node, result)
        
        if root.children:
            nodes_explored = sum(child.visits for child in root.children) if root.children else 0
            self.metrics['nodes_explored_per_move'].append(nodes_explored)
            
            move_time = time.time() - start_time
            self.metrics['average_time_per_move'].append(move_time)
            
            best_move = max(root.children, key=lambda c: c.visits).move if root.children else random.choice(list(board.legal_moves))
            
            if board.is_capture(best_move):
                self.metrics['capture_moves_played'] += 1
            board.push(best_move)
            if board.is_check():
                self.metrics['check_moves_played'] += 1
            board.pop()
            
            return best_move
            
        return random.choice(list(board.legal_moves))

    def set_simulation_time(self, seconds):
        self.simulation_time = seconds

    def _is_endgame(self, board):
        """Determine if the position is in endgame"""
        queens = 0
        minor_pieces = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                if piece.piece_type == chess.QUEEN:
                    queens += 1
                elif piece.piece_type in [chess.BISHOP, chess.KNIGHT]:
                    minor_pieces += 1
        
        return queens == 0 or (queens == 2 and minor_pieces <= 2)

    def _evaluate_endgame(self, board):
        """Special evaluation terms for endgame positions"""
        score = 0
        
        w_king_square = board.king(chess.WHITE)
        b_king_square = board.king(chess.BLACK)
        
        w_king_file, w_king_rank = chess.square_file(w_king_square), chess.square_rank(w_king_square)
        b_king_file, b_king_rank = chess.square_file(b_king_square), chess.square_rank(b_king_square)
        
        w_king_center_dist = abs(3.5 - w_king_file) + abs(3.5 - w_king_rank)
        b_king_center_dist = abs(3.5 - b_king_file) + abs(3.5 - b_king_rank)
        
        score += (7 - w_king_center_dist) * 10
        score -= (7 - b_king_center_dist) * 10
        
        return score

    def _is_tactical_position(self, board):
        """Detect if position is tactical (lots of captures/checks available)"""
        tactical_count = 0
        for move in board.legal_moves:
            if board.is_capture(move):
                tactical_count += 1
            board.push(move)
            if board.is_check():
                tactical_count += 1
            board.pop()
        return tactical_count >= 3
    def record_game_result(self, board, my_color):
        """Record the game result and update metrics"""
        self.metrics['total_games_played'] += 1
        self.metrics['average_moves_per_game'].append(board.fullmove_number)
        
        if board.is_checkmate():
            if board.turn != my_color:
                self.metrics['wins'] += 1
            else:
                self.metrics['losses'] += 1
        elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition():
            self.metrics['draws'] += 1

    def get_metrics_summary(self):
        """Return a formatted summary of the AI's performance metrics"""
        if self.metrics['total_games_played'] == 0:
            return "No games played yet"
            
        win_rate = (self.metrics['wins'] / self.metrics['total_games_played']) * 100
        avg_moves = sum(self.metrics['average_moves_per_game']) / len(self.metrics['average_moves_per_game']) if self.metrics['average_moves_per_game'] else 0
        avg_time = sum(self.metrics['average_time_per_move']) / len(self.metrics['average_time_per_move']) if self.metrics['average_time_per_move'] else 0
        avg_nodes = sum(self.metrics['nodes_explored_per_move']) / len(self.metrics['nodes_explored_per_move']) if self.metrics['nodes_explored_per_move'] else 0
        
        return {
            'Games Played': self.metrics['total_games_played'],
            'Win Rate': f"{win_rate:.2f}%",
            'Win/Loss/Draw': f"{self.metrics['wins']}/{self.metrics['losses']}/{self.metrics['draws']}",
            'Average Moves per Game': f"{avg_moves:.1f}",
            'Average Time per Move': f"{avg_time:.3f}s",
            'Average Nodes Explored': f"{avg_nodes:.0f}",
            'Tactical Stats': {
                'Captures': self.metrics['capture_moves_played'],
                'Checks': self.metrics['check_moves_played'],
                'Opening Book Usage': self.metrics['opening_book_moves_used']
            }
        }

    def _evaluate_tactical_patterns(self, board):
        """Evaluate tactical patterns like pins, forks, and discovered attacks"""
        score = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                if self._is_pinned(board, square):
                    if piece.color == chess.WHITE:
                        score -= self.piece_values[piece.piece_type] * 0.3
                    else:
                        score += self.piece_values[piece.piece_type] * 0.3
        
        score += self._evaluate_forks(board, chess.WHITE)
        score -= self._evaluate_forks(board, chess.BLACK)
        
        score += self._evaluate_discovered_attacks(board, chess.WHITE)
        score -= self._evaluate_discovered_attacks(board, chess.BLACK)
        
        return score

    def _is_pinned(self, board, square):
        """Check if a piece is pinned to its king"""
        piece = board.piece_at(square)
        if not piece:
            return False
        
        color = piece.color
        king_square = board.king(color)
        if not king_square:
            return False
        
        return board.is_pinned(color, square)

    def _evaluate_forks(self, board, color):
        """Evaluate potential and actual forks"""
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                attacked_squares = list(board.attacks(square))
                attacked_pieces = []
                
                for attacked_square in attacked_squares:
                    attacked_piece = board.piece_at(attacked_square)
                    if attacked_piece and attacked_piece.color != color:
                        attacked_pieces.append(attacked_piece)
                
                # Score potential forks
                if len(attacked_pieces) >= 2:
                    fork_value = sum(self.piece_values[p.piece_type] for p in attacked_pieces)
                    score += fork_value * 0.2
        
        return score

    def _evaluate_discovered_attacks(self, board, color):
        """Evaluate potential discovered attacks"""
        score = 0
        king_square = board.king(not color)
        
        if not king_square:
            return 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                for move in board.legal_moves:
                    if move.from_square == square:
                        board.push(move)
                        if board.is_check():
                            score += 50
                        board.pop()
        
        return score

    def _evaluate_center_control(self, board):
        """Evaluate control of the center squares"""
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        extended_center = [chess.C3, chess.C4, chess.C5, chess.C6,
                          chess.D3, chess.D6, chess.E3, chess.E6,
                          chess.F3, chess.F4, chess.F5, chess.F6]
        
        score = 0
        
        for square in center_squares:
            if board.piece_at(square):
                piece = board.piece_at(square)
                value = 30 if piece.color == chess.WHITE else -30
                score += value
            white_attackers = len(board.attackers(chess.WHITE, square))
            black_attackers = len(board.attackers(chess.BLACK, square))
            score += (white_attackers - black_attackers) * 10

        for square in extended_center:
            if board.piece_at(square):
                piece = board.piece_at(square)
                value = 15 if piece.color == chess.WHITE else -15
                score += value
            white_attackers = len(board.attackers(chess.WHITE, square))
            black_attackers = len(board.attackers(chess.BLACK, square))
            score += (white_attackers - black_attackers) * 5
        
        return score

    def _evaluate_pawn_structure(self, board):
        """Enhanced pawn structure evaluation"""
        score = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                multiplier = 1 if piece.color == chess.WHITE else -1
                
                if self._is_passed_pawn(board, square, piece.color):
                    score += 50 * multiplier
                
                if self._is_doubled_pawn(board, square, piece.color):
                    score -= 20 * multiplier
                
                if self._is_isolated_pawn(board, square, piece.color):
                    score -= 15 * multiplier
                
                if self._is_backward_pawn(board, square, piece.color):
                    score -= 15 * multiplier
                
                if self._is_connected_pawn(board, square, piece.color):
                    score += 10 * multiplier
        
        return score

    def _is_doubled_pawn(self, board, square, color):
        """Check if pawn is doubled"""
        file = chess.square_file(square)
        for rank in range(8):
            test_square = chess.square(file, rank)
            if test_square != square:
                piece = board.piece_at(test_square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    return True
        return False

    def _is_isolated_pawn(self, board, square, color):
        """Check if pawn is isolated"""
        file = chess.square_file(square)
        for adjacent_file in [file - 1, file + 1]:
            if 0 <= adjacent_file < 8:
                for rank in range(8):
                    test_square = chess.square(adjacent_file, rank)
                    piece = board.piece_at(test_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        return False
        return True

    def _is_backward_pawn(self, board, square, color):
        """Check if pawn is backward"""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        
        direction = 1 if color == chess.WHITE else -1
        
        for adjacent_file in [file - 1, file + 1]:
            if 0 <= adjacent_file < 8:
                found_advanced_pawn = False
                for r in range(rank + direction, 7 if direction > 0 else 0, direction):
                    test_square = chess.square(adjacent_file, r)
                    piece = board.piece_at(test_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        found_advanced_pawn = True
                        break
                if found_advanced_pawn:
                    return True
        return False

    def _is_connected_pawn(self, board, square, color):
        """Check if pawn is connected to friendly pawns"""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        
        for adjacent_file in [file - 1, file + 1]:
            if 0 <= adjacent_file < 8:
                test_square = chess.square(adjacent_file, rank)
                piece = board.piece_at(test_square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    return True
        return False