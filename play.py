import tkinter as tk
from tkinter import messagebox, font
import chess
import torch
from model import ChessModel, create_move_mapping
import numpy as np

class ChessGUI:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("Chess vs AI")
        
        # Load the model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Create move mappings
        self.move_to_idx, self.idx_to_move = create_move_mapping()
        
        # Initialize chess board
        self.board = chess.Board()
        self.selected_square = None
        
        # Unicode chess pieces
        self.piece_symbols = {
            'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
            'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
        }
        
        # Create the chess board GUI
        self.square_size = 64
        self.canvas = tk.Canvas(root, width=self.square_size * 8, height=self.square_size * 8)
        self.canvas.pack()
        
        # Create custom font for chess pieces
        self.piece_font = font.Font(family='Arial', size=32)
        
        # Bind mouse clicks
        self.canvas.bind('<Button-1>', self.on_square_click)
        
        # Draw initial board
        self.draw_board()
        
    def load_model(self, model_path):
        model = ChessModel().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def draw_board(self):
        self.canvas.delete("all")
        
        # Draw squares
        for row in range(8):
            for col in range(8):
                x1 = col * self.square_size
                y1 = (7-row) * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                
                # Light squares: #F0D9B5, Dark squares: #B58863
                color = "#F0D9B5" if (row + col) % 2 == 0 else "#B58863"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
                
                # Draw piece if present
                square = chess.square(col, row)
                piece = self.board.piece_at(square)
                if piece:
                    piece_symbol = self.piece_symbols[piece.symbol()]
                    # Center the piece in the square
                    self.canvas.create_text(
                        x1 + self.square_size/2,
                        y1 + self.square_size/2,
                        text=piece_symbol,
                        font=self.piece_font,
                        fill="#000000" if piece.color else "#000000"
                    )
        
        # Highlight selected square
        if self.selected_square is not None:
            col = chess.square_file(self.selected_square)
            row = chess.square_rank(self.selected_square)
            x1 = col * self.square_size
            y1 = (7-row) * self.square_size
            self.canvas.create_rectangle(x1, y1, x1 + self.square_size, y1 + self.square_size, 
                                      outline="#ff0000", width=2)
        
        # Add coordinate labels
        for i in range(8):
            # Rank numbers (1-8)
            self.canvas.create_text(
                2, 
                i * self.square_size + self.square_size/2,
                text=str(8-i),
                anchor="w",
                fill="#000000"
            )
            # File letters (a-h)
            self.canvas.create_text(
                i * self.square_size + self.square_size/2,
                self.square_size * 8 - 2,
                text=chr(97 + i),
                anchor="s",
                fill="#000000"
            )
    
    def board_to_tensor(self):
        """Convert current board position to tensor format"""
        tensor = torch.zeros(16, 8, 8, dtype=torch.float32)
        
        piece_idx = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                rank = square // 8
                file = square % 8
                idx = piece_idx[piece.symbol()]
                tensor[idx][rank][file] = 1.0
        
        tensor[12][0][0] = float(self.board.has_kingside_castling_rights(chess.WHITE))
        tensor[12][0][1] = float(self.board.has_queenside_castling_rights(chess.WHITE))
        tensor[12][7][0] = float(self.board.has_kingside_castling_rights(chess.BLACK))
        tensor[12][7][1] = float(self.board.has_queenside_castling_rights(chess.BLACK))
        
        if self.board.ep_square is not None:
            rank = self.board.ep_square // 8
            file = self.board.ep_square % 8
            tensor[13][rank][file] = 1.0
        
        tensor[14].fill_(float(self.board.turn))
        tensor[15].fill_(float(self.board.fullmove_number) / 100.0)
        
        return tensor
    
    def get_ai_move(self):
        """Get the AI's move using the trained model"""
        with torch.no_grad():
            board_tensor = self.board_to_tensor().unsqueeze(0).to(self.device)
            global_features = torch.zeros(1, 8, dtype=torch.float32).to(self.device)
            
            # Fill global features
            global_features[0][0] = board_tensor[0][14][0][0]  # Turn
            global_features[0][1] = board_tensor[0][12][0][0]  # White kingside castling
            global_features[0][2] = board_tensor[0][12][0][1]  # White queenside castling
            global_features[0][3] = board_tensor[0][12][7][0]  # Black kingside castling
            global_features[0][4] = board_tensor[0][12][7][1]  # Black queenside castling
            global_features[0][5] = float(board_tensor[0][13].sum() > 0)  # En passant available
            global_features[0][6] = board_tensor[0][15][0][0]  # Move number
            
            try:
                move_logits, _ = self.model(board_tensor, global_features)
                
                # Get dimensions for debugging
                print(f"Move logits shape: {move_logits.shape}")
                
                # Ensure move_logits is the right shape
                if len(move_logits.shape) != 2:
                    move_logits = move_logits.view(1, -1)
                
                move_probs = torch.softmax(move_logits, dim=1)
                
                # Filter only legal moves
                legal_moves = list(self.board.legal_moves)
                legal_move_indices = [self.move_to_idx.get(move.uci(), -1) for move in legal_moves]
                legal_move_indices = [i for i in legal_move_indices if i != -1]
                
                if not legal_move_indices:
                    # If no legal moves found in our mapping, choose a random legal move
                    if legal_moves:
                        return legal_moves[0]
                    return None
                
                # Ensure we're not accessing invalid indices
                max_idx = move_probs.shape[1] - 1
                legal_move_indices = [i for i in legal_move_indices if i <= max_idx]
                
                if not legal_move_indices:
                    # If still no valid moves, choose a random legal move
                    if legal_moves:
                        return legal_moves[0]
                    return None
                
                legal_probs = move_probs[0][legal_move_indices]
                selected_idx = legal_move_indices[legal_probs.argmax().item()]
                
                # Convert back to UCI move
                move_uci = self.idx_to_move[selected_idx]
                return chess.Move.from_uci(move_uci)
                
            except Exception as e:
                print(f"Error in get_ai_move: {str(e)}")
                # Fallback to choosing first legal move
                legal_moves = list(self.board.legal_moves)
                return legal_moves[0] if legal_moves else None
    
    def on_square_click(self, event):
        col = event.x // self.square_size
        row = 7 - (event.y // self.square_size)
        square = chess.square(col, row)
        
        if self.selected_square is None:
            # First click - select piece
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                self.selected_square = square
        else:
            # Second click - try to make move
            move = chess.Move(self.selected_square, square)
            
            # Check if promotion is needed
            selected_piece = self.board.piece_at(self.selected_square)
            if (selected_piece and 
                selected_piece.piece_type == chess.PAWN and 
                ((self.board.turn and row == 7) or (not self.board.turn and row == 0))):
                move = chess.Move(self.selected_square, square, promotion=chess.QUEEN)
            
            # Check if move is legal
            if move in self.board.legal_moves:
                self.board.push(move)
                
                # AI's turn
                ai_move = self.get_ai_move()
                if ai_move:
                    self.board.push(ai_move)
                
                # Check game state
                if self.board.is_game_over():
                    result = self.board.outcome()
                    messagebox.showinfo("Game Over", f"Game Over! Result: {result.result()}")
                    self.board.reset()
            
            self.selected_square = None
        
        self.draw_board()

def main():
    # Specify your model path here
    model_path = "best_model.pth"  # Replace with your model path
    
    root = tk.Tk()
    app = ChessGUI(root, model_path)
    root.mainloop()

if __name__ == "__main__":
    main()