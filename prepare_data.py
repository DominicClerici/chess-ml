import pandas as pd
import chess.pgn
import io
from datetime import datetime
import numpy as np
from tqdm import tqdm
from tqdm.auto import tqdm

def safe_int_conversion(value, default=0):
    """
    Safely convert a value to int, returning default if conversion fails
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def read_pgn_file(file_path, num_games=None):
    """
    Read games from a local PGN file
    """
    games_data = []
    games_processed = 0
    
    with open(file_path) as pgn:
        pbar = tqdm(desc="Reading games", unit=" games")
        
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:  # End of file
                break
                
            if num_games and games_processed >= num_games:
                break
                
            try:
                # Extract game information with safe conversion for ratings
                game_data = {
                    'white_player': game.headers.get('White', 'Unknown'),
                    'black_player': game.headers.get('Black', 'Unknown'),
                    'white_rating': safe_int_conversion(game.headers.get('WhiteElo', '0')),
                    'black_rating': safe_int_conversion(game.headers.get('BlackElo', '0')),
                    'result': game.headers.get('Result', '*'),
                    'opening_name': game.headers.get('Opening', 'Unknown'),
                    'time_control': game.headers.get('TimeControl', 'Unknown'),
                    'moves': ' '.join(str(move) for move in game.mainline_moves()),
                    'event': game.headers.get('Event', 'Unknown'),
                    'date': game.headers.get('Date', 'Unknown')
                }
                
                # Only append games that have moves
                if game_data['moves'].strip():
                    games_data.append(game_data)
                    games_processed += 1
                    pbar.update(1)
                
            except Exception as e:
                print(f"Error processing game {games_processed}: {str(e)}")
                continue
                
    pbar.close()
    print(f"\nSuccessfully processed {len(games_data)} games")
    return games_data

def create_move_sequences(games_df, max_moves=50):
    """
    Convert move strings into formatted sequences suitable for ML
    """
    def process_moves(moves_str):
        moves = moves_str.split()
        if len(moves) > max_moves:
            moves = moves[:max_moves]
        else:
            moves = moves + ['PAD'] * (max_moves - len(moves))
        return moves
    
    # Create move columns
    move_cols = [f'move_{i}' for i in range(max_moves)]
    
    # Initialize tqdm for pandas
    # tqdm_pandas()
    tqdm.pandas()

    
    # Process moves with progress bar
    print("Processing move sequences...")
    moves_expanded = pd.Series(list(tqdm(map(process_moves, games_df['moves']), total=len(games_df))))
    moves_df = pd.DataFrame(moves_expanded.tolist(), columns=move_cols)
    
    # Combine with original data
    result_df = pd.concat([games_df.drop('moves', axis=1), moves_df], axis=1)
    
    return result_df

def prepare_training_data(file_path, num_games=None, max_moves=50):
    """
    Main function to prepare chess data for ML training
    """
    # Read games
    games_data = read_pgn_file(file_path, num_games)
    
    # Convert to DataFrame
    df = pd.DataFrame(games_data)
    
    # Process move sequences
    df_processed = create_move_sequences(df, max_moves)
    
    # Convert result to numerical values
    df_processed['result'] = df_processed['result'].map({'1-0': 1, '0-1': -1, '1/2-1/2': 0, '*': None})
    
    # Drop games with unknown results
    df_processed = df_processed.dropna(subset=['result'])
    
    # Ensure we have enough games for the split
    total_games = len(df_processed)
    train_size = int(total_games * 0.2)  # Use 20%
    test_size = total_games - train_size
    
    # Create train/test split
    train_df = df_processed.iloc[:train_size]
    test_df = df_processed.iloc[train_size:train_size + test_size]
    
    print(f"\nTraining set size: {len(train_df)}")
    print(f"Testing set size: {len(test_df)}")
    
    # Print data statistics
    print("\nRating statistics:")
    print(f"Average white rating (train): {train_df['white_rating'].mean():.0f}")
    print(f"Average black rating (train): {train_df['black_rating'].mean():.0f}")
    print(f"Games with valid ratings: {(train_df['white_rating'] > 0).sum()}")
    
    return train_df, test_df
