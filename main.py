import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from tqdm import tqdm
import chess
import logging
from datetime import datetime
import gc
import pandas as pd
import os
import pickle
from pathlib import Path
import time
import torch.nn.functional as F

# Import our modules
from prepare_data import prepare_training_data
from model import ChessModel, create_move_mapping

def setup_logging():
    """Setup logging configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'training_{timestamp}.log'
    
    # Clear any existing handlers
    logging.getLogger().handlers = []
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w'),
            logging.StreamHandler()
        ]
    )
    return log_filename, timestamp


class ChessDataset(Dataset):
    """Custom Dataset for chess positions with caching support"""
    def __init__(self, dataframe, move_to_idx, max_moves=1968, cache_dir='cached_data'):
        self.data = dataframe
        self.move_to_idx = move_to_idx
        self.max_moves = max_moves
        self.cache_dir = Path(cache_dir)
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(exist_ok=True)
        
        # Generate cache filename based on dataset characteristics
        
        self.cache_file = self.cache_dir / f'chess_dataset.pkl'
        
        # Try to load from cache first
        if self.cache_file.exists():
            print(f"Loading processed positions from cache: {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.positions = cached_data['positions']
                self.target_moves = cached_data['target_moves']
                self.evaluations = cached_data['evaluations']
        else:
            # Process data if cache doesn't exist
            self.positions = []
            self.target_moves = []
            self.evaluations = []
            
            # Add validation before processing
            if len(dataframe) == 0:
                raise ValueError("Empty dataframe provided to ChessDataset")
                
            self._process_games()
            
            # Save to cache
            print(f"Saving processed positions to cache: {self.cache_file}")
            with open(self.cache_file, 'wb') as f:
                pickle.dump({
                    'positions': self.positions,
                    'target_moves': self.target_moves,
                    'evaluations': self.evaluations
                }, f)
        
        # Add validation after loading/processing
        if len(self.positions) == 0:
            raise ValueError("No valid positions were processed from the games")
            
        print(f"Dataset ready with {len(self.positions)} positions")
        if self.target_moves:
            print(f"Max move index: {max(self.target_moves)}")
            print(f"Number of unique moves: {len(set(self.target_moves))}")

    
    def _process_games(self):
        """Process all games into position-move pairs"""
        skipped_games = 0
        total_moves_processed = 0
        
        for idx in tqdm(range(len(self.data)), desc="Processing games"):
            try:
                positions, moves, evals = self._process_single_game(self.data.iloc[idx])
                if positions and moves:
                    self.positions.extend(positions)
                    self.target_moves.extend(moves)
                    self.evaluations.extend(evals)
                    total_moves_processed += len(moves)
                else:
                    skipped_games += 1
            except Exception as e:
                skipped_games += 1
                print(f"Error processing game {idx}: {str(e)}")
                continue
                
        print(f"Total games processed: {len(self.data)}")
        print(f"Skipped games: {skipped_games}")
        print(f"Total moves processed: {total_moves_processed}")
        
        if total_moves_processed == 0:
            raise ValueError("No valid moves were processed from any games")

    
    def _algebraic_to_uci(self, board, move_str):
        """Convert algebraic notation to UCI format"""
        try:
            # Try to parse the move string directly
            move = board.parse_san(move_str)
            return move.uci()
        except:
            # If direct parsing fails, return None
            return None
    
    def _process_single_game(self, game_data):
        """Process a single game into a sequence of positions and moves"""
        positions = []
        moves = []
        evaluations = []
        board = chess.Board()
        
        # Get moves from the game
        move_list = []
        for i in range(50):  # Use same max_moves as in data processing
            move_col = f'move_{i}'
            if move_col in game_data and game_data[move_col] != 'PAD':
                move_list.append(game_data[move_col])
        
        # Get game result for evaluation
        result = game_data.get('result', 0)  # Default to draw if not found
        
        for move_idx, move_str in enumerate(move_list):
            try:
                # Store current position before making the move
                current_pos = self._board_to_tensor(board)
                
                # Convert move to UCI format if needed
                if len(move_str) < 4:  # Algebraic notation
                    move_uci = self._algebraic_to_uci(board, move_str)
                    if not move_uci:
                        continue
                else:  # UCI notation
                    move_uci = move_str
                
                # Create and validate move
                try:
                    move = chess.Move.from_uci(move_uci)
                except ValueError:
                    continue
                
                # Check if move is legal
                if move not in board.legal_moves:
                    continue
                
                # Get move index
                move_idx = self._move_to_index(move)
                if move_idx >= self.max_moves:
                    continue
                
                # Calculate position evaluation
                eval_score = self.calculate_position_score(board, move_idx, len(move_list), result)
                
                # Flip evaluation for black's perspective
                if not board.turn:
                    eval_score = -eval_score
                
                # Store position and move
                positions.append(current_pos)
                moves.append(move_idx)
                evaluations.append(eval_score)
                
                # Make the move
                board.push(move)
                
            except Exception as e:
                continue
        
        return positions, moves, evaluations
    
    def _board_to_tensor(self, board):
        """Convert board position to tensor format"""
        tensor = torch.zeros(20, 8, 8, dtype=torch.float32)
        
        # Piece placement
        piece_idx = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        
        # Fill piece positions
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank = square // 8
                file = square % 8
                idx = piece_idx[piece.symbol()]
                tensor[idx][rank][file] = 1.0
        
        # Additional features
        # Castling rights
        tensor[12][0][0] = float(board.has_kingside_castling_rights(chess.WHITE))
        tensor[12][0][1] = float(board.has_queenside_castling_rights(chess.WHITE))
        tensor[12][7][0] = float(board.has_kingside_castling_rights(chess.BLACK))
        tensor[12][7][1] = float(board.has_queenside_castling_rights(chess.BLACK))
        
        # En passant
        if board.ep_square is not None:
            rank = board.ep_square // 8
            file = board.ep_square % 8
            tensor[13][rank][file] = 1.0
        
        # Turn
        tensor[14].fill_(float(board.turn))
        
        # Move number
        tensor[15].fill_(float(board.fullmove_number) / 100.0)  # Normalized
        
        # Attacked squares
        for square in chess.SQUARES:
            if board.is_attacked_by(chess.WHITE, square):
                tensor[16][square // 8][square % 8] = 1.0
            if board.is_attacked_by(chess.BLACK, square):
                tensor[17][square // 8][square % 8] = 1.0

        # Piece mobility (number of legal moves for each piece)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                legal_moves = len([move for move in board.legal_moves if move.from_square == square])
                tensor[18][square // 8][square % 8] = min(legal_moves / 8.0, 1.0)  # Normalize

        # Center control
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        for square in center_squares:
            tensor[19][square // 8][square % 8] = 1.0
        
        return tensor
    
    def _move_to_index(self, move):
        """Convert a chess.Move to an index"""
        move_str = move.uci()
        return self.move_to_idx.get(move_str, 0)
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        board_tensor = self.positions[idx]
        target = self.target_moves[idx]
        evaluation = self.evaluations[idx]
        
        # Create global features
        global_features = torch.zeros(8, dtype=torch.float32)
        # Fill with basic features (can be expanded)
        global_features[0] = board_tensor[14][0][0]  # Turn
        global_features[1] = board_tensor[12][0][0]  # White kingside castling
        global_features[2] = board_tensor[12][0][1]  # White queenside castling
        global_features[3] = board_tensor[12][7][0]  # Black kingside castling
        global_features[4] = board_tensor[12][7][1]  # Black queenside castling
        global_features[5] = float(board_tensor[13].sum() > 0)  # En passant available
        global_features[6] = board_tensor[15][0][0]  # Move number
        global_features[7] = evaluation
        
        return board_tensor, global_features, target, torch.tensor(evaluation, dtype=torch.float32)

    def calculate_position_score(self, board, move_idx, total_moves, result):
        """Calculate a more comprehensive position evaluation score"""
        # Material value dictionary
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        
        # Calculate material balance
        material_balance = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                material_balance += value if piece.color else -value
        
        # Normalize material balance to [-1, 1]
        material_score = np.tanh(material_balance / 30.0)
        
        # Base evaluation from game result
        if result == 1:  # White win
            base_eval = 1.0 - (0.5 * move_idx / total_moves)
        elif result == -1:  # Black win
            base_eval = -1.0 + (0.5 * move_idx / total_moves)
        else:  # Draw
            base_eval = 0.0
        
        # Combine scores (0.6 weight to material, 0.4 to game result)
        eval_score = 0.6 * material_score + 0.4 * base_eval
        
        return eval_score

def calculate_repetition_penalty(move_logits, board_history):
    """Penalize repetitive moves"""
    penalty = 0
    if len(board_history) >= 4:
        last_positions = board_history[-4:]
        if len(set(last_positions)) < len(last_positions):
            penalty = 0.5  # Significant penalty for repetition
    return penalty

def train_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, device):
    """Train for one epoch with mixed precision training"""
    if len(train_loader) == 0:
        raise ValueError("Empty train_loader provided to train_epoch")
        
    model.train()
    total_loss = 0
    move_loss = 0
    eval_loss = 0
    correct_moves = 0
    total_moves = 0
    
    board_history = []  # Add at start of function
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch_idx, (board_tensors, global_features, targets, evaluations) in enumerate(progress_bar):
        try:
            board_tensors = board_tensors.to(device, non_blocking=True)
            global_features = global_features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            evaluations = evaluations.to(device, non_blocking=True)
            
            # Ensure evaluations has the right shape [batch_size, 1]
            evaluations = evaluations.view(-1, 1)
            
            # Add current board state to history (using first board in batch as reference)
            board_history.append(board_tensors[0].cpu().numpy().tobytes())
            if len(board_history) > 4:  # Keep only last 4 positions
                board_history.pop(0)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                move_logits, position_eval = model(board_tensors, global_features)
                
                # Ensure position_eval has the right shape [batch_size, 1]
                position_eval = position_eval.view(-1, 1)
                
                move_loss = criterion(move_logits, targets)
                eval_loss = F.mse_loss(position_eval, evaluations)
                repetition_penalty = calculate_repetition_penalty(move_logits, board_history)
                loss = move_loss + 0.1 * eval_loss + repetition_penalty
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            pred_moves = move_logits.argmax(dim=1)
            correct_moves += (pred_moves == targets).sum().item()
            total_moves += targets.size(0)
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'move_acc': f'{100*correct_moves/total_moves:.2f}%',
                'eval_loss': f'{eval_loss.item():.4f}'
            })
            
        except Exception as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            continue
    
    return total_loss/len(train_loader), correct_moves/total_moves



def validate(model, val_loader, criterion, device):
    """Validate the model with mixed precision"""
    model.eval()
    total_loss = 0
    correct_moves = 0
    total_moves = 0
    
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for batch_idx, (board_tensors, global_features, targets, evaluations) in enumerate(val_loader):
                try:
                    board_tensors = board_tensors.to(device, non_blocking=True)
                    global_features = global_features.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    evaluations = evaluations.to(device, non_blocking=True).view(-1, 1)
                    
                    move_logits, position_eval = model(board_tensors, global_features)
                    position_eval = position_eval.view(-1, 1)
                    
                    move_loss = criterion(move_logits, targets)
                    eval_loss = F.mse_loss(position_eval, evaluations)
                    loss = move_loss + 0.1 * eval_loss
                    
                    total_loss += loss.item()
                    pred_moves = move_logits.argmax(dim=1)
                    correct_moves += (pred_moves == targets).sum().item()
                    total_moves += targets.size(0)
                    
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {str(e)}")
                    continue
                
                # Clear some memory
                if batch_idx % 100 == 0:
                    torch.cuda.empty_cache()
    
    if total_moves == 0:
        raise ValueError("No moves were processed during validation")
        
    return total_loss/len(val_loader), correct_moves/total_moves


def main():
    
    log_filename, timestamp = setup_logging()
    logging.info(f"Logging to: {log_filename}")

    # Enhanced configuration
    config = {
        'BATCH_SIZE': 128,         # Increased from 32 to process more data efficiently
        'EPOCHS': 25,              # Reduced from 30 since we have more data
        'BASE_LR': 0.0001,         # Slightly increased for better initial convergence
        'MIN_LR': 0.000003,        # Maintained 1/100 ratio of BASE_LR
        'WEIGHT_DECAY': 0.001,    # Reduced slightly due to larger dataset
        'NUM_WORKERS': 12,         # Increased for better data loading
        'PIN_MEMORY': True,
        'PREFETCH_FACTOR': 2,      # Reduced to prevent memory issues
        'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'GRADIENT_CLIP': .5,      # Increased slightly
        'MIN_DELTA': 0.0002        # Reduced threshold for model saving
    }

    
    logging.info("Training configuration:")
    for key, value in config.items():
        logging.info(f"{key}: {value}")


    # Load and prepare data
    logging.info("Loading chess games...")
    train_df, test_df = prepare_training_data("data.pgn", num_games=10000, cache_dir='cached_data')

    
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("No data loaded from PGN file")
    
    logging.info(f"Loaded {len(train_df)} training games and {len(test_df)} test games")
    
    # Create move mappings
    move_to_idx, idx_to_move = create_move_mapping()
    logging.info(f"Created mapping for {len(move_to_idx)} possible moves")
    
    # Create datasets with try-except
    try:
        train_dataset = ChessDataset(train_df, move_to_idx, cache_dir='cached_data')
        test_dataset = ChessDataset(test_df, move_to_idx, cache_dir='cached_data')
    except Exception as e:
        logging.error(f"Failed to create datasets: {str(e)}")
        raise
        
    # Validate dataset sizes
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError("Empty datasets created")
    
    logging.info(f"Created datasets with {len(train_dataset)} training positions and {len(test_dataset)} test positions")
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['BATCH_SIZE'],
        shuffle=True,
        num_workers=config['NUM_WORKERS'],
        pin_memory=config['PIN_MEMORY'],
        prefetch_factor=config['PREFETCH_FACTOR'],
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['BATCH_SIZE'],
        num_workers=config['NUM_WORKERS'],
        pin_memory=config['PIN_MEMORY'],
        prefetch_factor=config['PREFETCH_FACTOR'],
        persistent_workers=True
    )

    
    # Define loss function and optimizer with weight decay
    criterion = nn.CrossEntropyLoss()
    
    # Calculate number of training steps for scheduler

    # Create learning rate scheduler
    model = ChessModel().to(config['DEVICE'])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['BASE_LR'],
        weight_decay=config['WEIGHT_DECAY'],
        betas=(0.95, 0.999)
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=config['BASE_LR'],
        epochs=config['EPOCHS'],
        steps_per_epoch=len(train_loader),
        pct_start=0.075,          # Only spend 10% of time warming up
        div_factor=1.0,         # Start at max_lr
        final_div_factor=100,   # End at min_lr
        anneal_strategy='cos'
    )
    
    # Option 2: Alternative - CosineAnnealingLR
    # from torch.optim.lr_scheduler import CosineAnnealingLR
    # scheduler = CosineAnnealingLR(
    #     optimizer,
    #     T_max=config['EPOCHS'] * len(train_loader),
    #     eta_min=config['MIN_LR']
    # )

    scaler = GradScaler()
    
    
    best_accuracy = 0
    moving_avg_accuracy = []
    training_start_time = time.time()
    
    for epoch in range(config['EPOCHS']):
        epoch_metrics = {'epoch': epoch + 1}
        logging.info(f"\nEpoch {epoch+1}/{config['EPOCHS']}")
        
        try:
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, scheduler, criterion, scaler, config['DEVICE']
            )
            epoch_metrics.update({
                'train_loss': train_loss,
                'train_acc': train_acc
            })
            logging.info(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
            
            # Validate
            val_loss, val_acc = validate(model, test_loader, criterion, config['DEVICE'])
            epoch_metrics.update({
                'val_loss': val_loss,
                'val_acc': val_acc
            })
            logging.info(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
            
            # Update moving average
            moving_avg_accuracy.append(val_acc)
            if len(moving_avg_accuracy) > 5:
                moving_avg_accuracy.pop(0)
            
            # Learning rate
            current_lr = scheduler.get_last_lr()[0]
            epoch_metrics['learning_rate'] = current_lr
            logging.info(f"Learning rate: {current_lr:.6f}")
            
            # Save best model with improved criterion
            if val_acc > best_accuracy + config['MIN_DELTA']:
                best_accuracy = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_accuracy': best_accuracy,
                    'config': config,
                }, f'best_model_{timestamp}.pth')
                logging.info(f"Saved new best model with accuracy: {best_accuracy:.4f}")
            
        except Exception as e:
            logging.error(f"Error during epoch {epoch+1}: {str(e)}")
            import traceback
            traceback.print_exc()
            break

    
    training_end_time = time.time()
    training_duration = training_end_time - training_start_time
    logging.info(f"Training completed in {training_duration:.2f} seconds")




if __name__ == "__main__":
    main()
    