from stockfish import Stockfish
import chess
import chess.engine
import time
from model import ChessAI
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from colorama import Fore, Style, init 

init()

class ChessAITester:
    def __init__(self):
        """
        Initialize the tester with paths to engines
        """

        stockfish_path = "stockfish.exe"

        if not Path(stockfish_path).exists():
            raise FileNotFoundError("Stockfish not found. Please provide valid path to Stockfish executable")
            
        self.stockfish = Stockfish(path=stockfish_path)
        self.ai = ChessAI()
        self.results_data = []

    def print_board(self, board, last_move=None):
        """Print the chess board with coordinates and last move highlighted"""
        print("\n   a b c d e f g h")
        print("   ---------------")
        for rank in range(7, -1, -1):
            print(f"{rank + 1}|", end=" ")
            for file in range(8):
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                
                if last_move and (square == last_move.from_square or square == last_move.to_square):
                    color = Fore.YELLOW
                else:
                    color = Fore.WHITE
                    
                if piece is None:
                    print(color + "." + Style.RESET_ALL, end=" ")
                else:
                    print(color + piece.symbol() + Style.RESET_ALL, end=" ")
            print(f"|{rank + 1}")
        print("   ---------------")
        print("   a b c d e f g h\n")

    def get_move_description(self, board, move):
        """Generate a human-readable move description"""
        piece = board.piece_at(move.from_square)
        capture = board.piece_at(move.to_square) is not None
        check = board.gives_check(move)
        
        piece_name = {
            chess.PAWN: "",
            chess.KNIGHT: "N",
            chess.BISHOP: "B",
            chess.ROOK: "R",
            chess.QUEEN: "Q",
            chess.KING: "K"
        }[piece.piece_type]
        
        move_str = f"{piece_name}{chess.square_name(move.from_square)}-{chess.square_name(move.to_square)}"
        
        if capture:
            move_str += " captures"
        if check:
            move_str += " +"
        if board.is_castling(move):
            move_str = "O-O" if move.to_square > move.from_square else "O-O-O"
            
        return move_str

    def play_single_game(self, elo_rating, ai_color):
        """Play a single game against Stockfish with move-by-move display"""
        self.stockfish.set_elo_rating(elo_rating)
        board = chess.Board()
        moves_played = 0
        game_start_time = time.time()
        
        print("\n" + "="*50)
        print(f"Starting new game - AI playing as {'White' if ai_color else 'Black'}")
        print(f"Stockfish ELO: {elo_rating}")
        print("="*50 + "\n")
        
        self.print_board(board)

        while not board.is_game_over():
            if board.turn == ai_color:
                print(f"{Fore.CYAN}AI thinking...{Style.RESET_ALL}")
                move_start_time = time.time()
                move = self.ai.get_best_move(board)
                move_time = time.time() - move_start_time
                
                move_desc = self.get_move_description(board, move)
                print(f"{Fore.CYAN}AI plays: {move_desc} ({move_time:.2f}s){Style.RESET_ALL}")
                
                self.ai.metrics['average_time_per_move'].append(move_time)
                if board.is_capture(move):
                    self.ai.metrics['capture_moves_played'] += 1
                
            else:
                print(f"{Fore.GREEN}Stockfish thinking...{Style.RESET_ALL}")
                self.stockfish.set_position([move.uci() for move in board.move_stack])
                stockfish_move = self.stockfish.get_best_move()
                move = chess.Move.from_uci(stockfish_move)
                
                move_desc = self.get_move_description(board, move)
                print(f"{Fore.GREEN}Stockfish plays: {move_desc}{Style.RESET_ALL}")
            
            board.push(move)
            moves_played += 1
            
            print(f"\nMove {moves_played}:")
            self.print_board(board, move)
            
            if board.is_check():
                print(f"{Fore.RED}CHECK!{Style.RESET_ALL}")
            

        print("\n" + "="*50)
        print("Game Over!")
        if board.is_checkmate():
            winner = "AI" if board.turn != ai_color else "Stockfish"
            print(f"{Fore.YELLOW}Checkmate! {winner} wins!{Style.RESET_ALL}")
        elif board.is_stalemate():
            print(f"{Fore.YELLOW}Stalemate! Game is drawn.{Style.RESET_ALL}")
        elif board.is_insufficient_material():
            print(f"{Fore.YELLOW}Draw by insufficient material.{Style.RESET_ALL}")
        elif board.is_fifty_moves():
            print(f"{Fore.YELLOW}Draw by fifty-move rule.{Style.RESET_ALL}")
        elif board.is_repetition():
            print(f"{Fore.YELLOW}Draw by repetition.{Style.RESET_ALL}")
        print("="*50 + "\n")

        game_duration = time.time() - game_start_time

        self.ai.record_game_result(board, ai_color)
        
        result = {
            'elo_rating': elo_rating,
            'ai_color': 'White' if ai_color else 'Black',
            'moves_played': moves_played,
            'game_duration': game_duration,
            'result': self._get_result_string(board, ai_color)
        }
        self.results_data.append(result)
        
        return result

    def test_against_stockfish(self, elo_ratings=None, games_per_level=10):
        """
        Test AI against Stockfish at multiple ELO levels
        :param elo_ratings: List of ELO ratings to test against
        :param games_per_level: Number of games to play at each level
        """
        if elo_ratings is None:
            elo_ratings = [100, 200, 300, 400, 500, 600]
        
        results = []
        
        for elo in elo_ratings:
            print(f"\nTesting against ELO {elo}...")
            level_results = []
            
            for game_num in range(games_per_level):
                ai_color = chess.WHITE if game_num % 2 == 0 else chess.BLACK
                color_str = "White" if ai_color else "Black"
                print(f"Playing game {game_num + 1}/{games_per_level} as {color_str}...")
                
                result = self.play_single_game(elo, ai_color)
                level_results.append(result)
                
                print(f"Game finished: {result['result']}")
            
            results.extend(level_results)
            
            self._print_level_summary(level_results, elo)
        
        return results

    def generate_report(self, output_file="chess_ai_results.html"):
        """Generate a detailed HTML report of the results"""
        df = pd.DataFrame(self.results_data)
        
        plt.figure(figsize=(10, 6))
        win_rates = df.groupby('elo_rating').apply(
            lambda x: (x['result'] == 'Win').mean() * 100
        )
        win_rates.plot(kind='line', marker='o')
        plt.title('Win Rate vs ELO Rating')
        plt.xlabel('ELO Rating')
        plt.ylabel('Win Rate (%)')
        plt.savefig('win_rate_plot.png')
        plt.close()
        
        html_content = f"""
        <html>
        <head>
            <title>Chess AI Test Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Chess AI Test Results</h1>
            
            <h2>Overall Statistics</h2>
            {self._generate_overall_stats()}
            
            <h2>Performance by ELO Rating</h2>
            <img src='win_rate_plot.png' alt='Win Rate Plot'>
            
            <h2>Detailed Results</h2>
            {df.to_html()}
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"Report generated: {output_file}")

    def _get_result_string(self, board, ai_color):
        """Convert game result to string"""
        if board.is_checkmate():
            return "Win" if board.turn != ai_color else "Loss"
        elif board.is_stalemate() or board.is_insufficient_material() or \
             board.is_fifty_moves() or board.is_repetition():
            return "Draw"
        return "Unknown"

    def _print_level_summary(self, results, elo):
        """Print summary statistics for a specific ELO level"""
        wins = sum(1 for r in results if r['result'] == 'Win')
        losses = sum(1 for r in results if r['result'] == 'Loss')
        draws = sum(1 for r in results if r['result'] == 'Draw')
        
        print(f"\nResults against ELO {elo}:")
        print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
        print(f"Win rate: {(wins / len(results)) * 100:.1f}%")

    def _generate_overall_stats(self):
        """Generate overall statistics HTML"""
        df = pd.DataFrame(self.results_data)
        total_games = len(df)
        wins = (df['result'] == 'Win').sum()
        draws = (df['result'] == 'Draw').sum()
        losses = (df['result'] == 'Loss').sum()
        
        return f"""
        <table>
            <tr><th>Total Games</th><td>{total_games}</td></tr>
            <tr><th>Wins</th><td>{wins} ({wins/total_games*100:.1f}%)</td></tr>
            <tr><th>Draws</th><td>{draws} ({draws/total_games*100:.1f}%)</td></tr>
            <tr><th>Losses</th><td>{losses} ({losses/total_games*100:.1f}%)</td></tr>
        </table>
        """

def main():
    try: 
        # make sure that you have stockfish installed in this directory as stockfish.exe
        tester = ChessAITester()
        
        print("Starting chess AI testing...")
        results = tester.test_against_stockfish(
            # elo_ratings=[200,400,600,800,1000],
            elo_ratings=[50,100,150,200],
            games_per_level=10
        )
        
        tester.generate_report()
        
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    main()