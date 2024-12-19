import tkinter as tk
from tkinter import font, messagebox, ttk
import chess
from model import ChessAI
import threading

class GameOverDialog:
    def __init__(self, parent, winner, restart_callback):
        self.top = tk.Toplevel(parent)
        self.top.title("Game Over")
        
        self.top.transient(parent)
        self.top.grab_set()

        window_width = 300
        window_height = 200
        screen_width = parent.winfo_screenwidth()
        screen_height = parent.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.top.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        frame = ttk.Frame(self.top, padding="20")
        frame.pack(fill='both', expand=True)
        
        trophy = "🏆"
        trophy_label = ttk.Label(frame, text=trophy, font=('TkDefaultFont', 48))
        trophy_label.pack(pady=(0, 10))
        
        winner_text = f"{winner} wins!" if winner != "Draw" else "It's a draw!"
        winner_label = ttk.Label(frame, text=winner_text, font=('TkDefaultFont', 14))
        winner_label.pack(pady=(0, 20))
        
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill='x', pady=(0, 10))
        
        restart_btn = ttk.Button(button_frame, text="Play Again", 
                               command=lambda: self.handle_restart(restart_callback))
        restart_btn.pack(side='left', padx=5, expand=True)
        
        quit_btn = ttk.Button(button_frame, text="Quit", 
                            command=lambda: parent.quit())
        quit_btn.pack(side='right', padx=5, expand=True)

    def handle_restart(self, restart_callback):
        self.top.destroy()
        restart_callback()

class ChessGUI:
    def __init__(self, root):
        self.root = root
        self.setup_game()

    def setup_game(self):
        self.root.title("Chess AI")
        
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(padx=10, pady=10)
        
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill='x', pady=(0, 10))
        
        self.thinking_label = ttk.Label(self.control_frame, text="")
        self.thinking_label.pack(side='left', padx=5)
        
        self.board = chess.Board()
        self.ai = ChessAI(simulation_time=3)
        self.selected_square = None
        
        self.pieces_unicode = {
            'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
            'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙',
        }
        
        self.piece_font = font.Font(family="Arial", size=32)
        
        self.square_size = 64
        self.canvas = tk.Canvas(self.main_frame, width=self.square_size * 8, 
                              height=self.square_size * 8)
        self.canvas.pack()
        
        self.piece_ids = {}
        
        self.create_board()
        self.create_pieces()
        
        self.canvas.bind('<Button-1>', self.on_square_click)

    def reset_game(self):
        self.canvas.delete("all")
        
        self.board = chess.Board()
        self.selected_square = None
        
        self.create_board()
        self.create_pieces()

    def create_board(self):
        for row in range(8):
            for col in range(8):
                x1 = col * self.square_size
                y1 = row * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                
                color = "#FFE9C5" if (row + col) % 2 == 0 else "#D18B47"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, tags="square")

    def create_pieces(self):
        self.canvas.delete("piece")
        self.piece_ids.clear()
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                self.create_piece(square, piece)

    def create_piece(self, square, piece):
        x = (chess.square_file(square) * self.square_size) + self.square_size/2
        y = ((7 - chess.square_rank(square)) * self.square_size) + self.square_size/2
        piece_symbol = self.pieces_unicode[piece.symbol()]
        piece_id = self.canvas.create_text(x, y, text=piece_symbol, font=self.piece_font,
                                         fill="black", tags="piece")
        self.piece_ids[square] = piece_id

    def update_board_display(self):
        self.create_pieces()

    def get_square_from_coords(self, x, y):
        file = x // self.square_size
        rank = 7 - (y // self.square_size)
        return chess.square(file, rank)

    def show_promotion_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Promote Pawn")
        dialog.transient(self.root)
        dialog.grab_set()
        
        result = tk.StringVar()
        
        def choose(piece):
            result.set(piece)
            dialog.destroy()
        
        pieces = [('Queen', 'q'), ('Rook', 'r'), ('Bishop', 'b'), ('Knight', 'n')]
        for name, piece in pieces:
            ttk.Button(dialog, text=name, command=lambda p=piece: choose(p)).pack(pady=5)
            
        self.root.wait_window(dialog)
        return result.get()

    def is_promotion_move(self, move):
        piece = self.board.piece_at(move.from_square)
        return (piece is not None and 
                piece.piece_type == chess.PAWN and 
                chess.square_rank(move.to_square) in [0, 7])

    def highlight_square(self, square):
        """Highlight a square on the board"""
        x1 = (square % 8) * self.square_size
        y1 = (7 - square // 8) * self.square_size
        x2 = x1 + self.square_size
        y2 = y1 + self.square_size
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="green", width=2, tags="highlight")

    def clear_highlights(self):
        """Clear all highlighted squares"""
        self.canvas.delete("highlight")

    def check_game_over(self):
        """Check if the game is over and show appropriate dialog"""
        if self.board.is_game_over():
            if self.board.is_checkmate():
                winner = "Black" if self.board.turn else "White"
            else:
                winner = "Draw"
            GameOverDialog(self.root, winner, self.reset_game)
            return True
        return False

    def on_square_click(self, event):
        if self.board.is_game_over():
            return
            
        square = self.get_square_from_coords(event.x, event.y)
        
        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece and piece.color == chess.WHITE:
                self.selected_square = square
                self.clear_highlights()
                self.highlight_square(square)
        else:
            move = chess.Move(self.selected_square, square)
            
            if self.is_promotion_move(move):
                promotion_piece = self.show_promotion_dialog()
                if promotion_piece:
                    move = chess.Move(self.selected_square, square, 
                                    promotion=chess.Piece.from_symbol(promotion_piece).piece_type)
            
            if move in self.board.legal_moves:
                self.make_move(move)
            
            self.selected_square = None
            self.clear_highlights()

    def ai_move(self):
        self.thinking_label.config(text="AI is thinking...")
        self.root.update()
        
        ai_move = self.ai.get_best_move(self.board)
        
        if ai_move:
            self.board.push(ai_move)
            self.update_board_display()
            
        self.thinking_label.config(text="")
        
        self.check_game_over()

    def make_move(self, move):
        self.board.push(move)
        self.update_board_display()
        
        if not self.check_game_over() and not self.board.is_game_over():
            threading.Thread(target=self.ai_move, daemon=True).start()

def main():
    root = tk.Tk()
    ChessGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()