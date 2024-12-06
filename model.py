import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ChessEncoder(nn.Module):
    """
    Encodes chess position into a learned representation
    """
    def __init__(self, board_embedding_size=384):
        super().__init__()
        
        # Input features for each square:
        # - Piece type (6 pieces + empty = 7)
        # - Piece color (2)
        # - Additional features (castling rights, en passant, etc.)
        self.input_channels = 16
        
        # Convolutional layers to process the board
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        
        # Global features processing (castling rights, en passant, etc.)
        self.global_features = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        # Combine board and global features
        self.combine_features = nn.Sequential(
            nn.Linear(256 * 64 + 128, board_embedding_size),
            nn.ReLU(),
            nn.LayerNorm(board_embedding_size)
        )

    def forward(self, board_tensor, global_features):
        # Process board
        x = self.conv_layers(board_tensor)  # [batch, 256, 8, 8]
        x = x.view(x.size(0), -1)  # Flatten: [batch, 256 * 64]
        
        # Process global features
        g = self.global_features(global_features)
        
        # Combine
        combined = torch.cat([x, g], dim=1)
        return self.combine_features(combined)

class MovePredictor(nn.Module):
    """
    Predicts next move and position evaluation
    """
    def __init__(self, board_embedding_size=384, num_moves=1968):  # 1968 is typical max legal moves
        super().__init__()
        
        # Transformer layers for move sequence processing
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=board_embedding_size,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1
            ),
            num_layers=6
        )
        
        # Move prediction head
        self.move_predictor = nn.Sequential(
            nn.Linear(board_embedding_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, num_moves)
        )
        
        # Position evaluation head
        self.evaluation_predictor = nn.Sequential(
            nn.Linear(board_embedding_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
            nn.Tanh()  # Output between -1 and 1
        )

    def forward(self, board_embedding):
        # Process with transformer
        transformed = self.transformer(board_embedding.unsqueeze(0)).squeeze(0)
        
        # Predict moves and evaluation
        move_logits = self.move_predictor(transformed)
        evaluation = self.evaluation_predictor(transformed)
        
        return move_logits, evaluation

class ChessModel(nn.Module):
    """
    Complete chess model combining encoder and predictor
    """
    def __init__(self):
        super().__init__()
        self.encoder = ChessEncoder()
        self.predictor = MovePredictor()

    def forward(self, board_tensor, global_features):
        # Encode position
        board_embedding = self.encoder(board_tensor, global_features)
        
        # Predict moves and evaluation
        move_logits, evaluation = self.predictor(board_embedding)
        
        return move_logits, evaluation

def create_move_mapping():
    """
    Creates mapping between chess moves and indices
    Returns:
        dict: Mapping from moves to indices
        dict: Reverse mapping from indices to moves
    """
    # This is a simplified version - we'll need to expand this
    moves = []
    files = 'abcdefgh'
    ranks = '12345678'
    pieces = ['', 'N', 'B', 'R', 'Q']  # '' represents pawn moves
    
    # Generate all possible moves
    for f1 in files:
        for r1 in ranks:
            for f2 in files:
                for r2 in ranks:
                    for piece in pieces:
                        move = f"{piece}{f1}{r1}{f2}{r2}"
                        moves.append(move)
    
    # Create mappings
    move_to_idx = {move: idx for idx, move in enumerate(moves)}
    idx_to_move = {idx: move for idx, move in enumerate(moves)}
    
    return move_to_idx, idx_to_move