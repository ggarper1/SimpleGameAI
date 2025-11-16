import argparse
import sys
import asyncio
import websockets
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from model import Game_Model
import json

class Trainer:
    GAME_API_URI = "ws://localhost:8080/ws"
    NUM_EPOCHS = 100
    WINDOW_SIZE = 100

    def __init__(self, model, verbose, learning_rate=0.001):
        self.websocket = None

        self.verbose = verbose
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        self.player1_wins = []
        self.player2_wins = []

        self.losses = []
        self.win_rates = []

    def _parse_data(self, data:str):
        json_data = json.loads(data)
        player = json_data["player"]
        input = [player]

        for segment in json_data["map"][f"player{player}Segments"]:
            input.append(segment["p1"]["x"])
            input.append(segment["p1"]["y"])

            input.append(segment["p2"]["x"])
            input.append(segment["p2"]["y"])

        other_player = 2 if player==1 else 1
        for segment in json_data["map"][f"player{other_player}Segments"]:
            input.append(segment["p1"]["x"])
            input.append(segment["p1"]["y"])

            input.append(segment["p2"]["x"])
            input.append(segment["p2"]["y"])

        input.append(json_data["map"][f"player{player}King"]["x"])
        input.append(json_data["map"][f"player{player}King"]["y"])

        input.append(json_data["map"][f"player{other_player}King"]["x"])
        input.append(json_data["map"][f"player{other_player}King"]["y"])

        return torch.FloatTensor(input), player

    def _parse_output(self, output:torch.Tensor):
        moves = []
        for i in range(0, len(output), 3):
            moves.append({
                "Position":{
                    "x": output[i].item(),
                    "y": output[i+1].item()
                },
                "Angle":output[i+2].item()
            })

        return json.dumps({"pieces":moves})

    def get_pieces(self, data):
        input, player = self._parse_data(data)
        moves, value = self.model(input)
        return value, self._parse_output(moves), player

    def _parse_result(self, result:websockets.Data):
        return json.loads(result)

    def compute_loss(self, value, result, player):
        won = result["result"] == player

        target = torch.FloatTensor([[1.0]]) if won else torch.FloatTensor([[0.0]])

        loss = nn.functional.mse_loss(value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        print(f"Player {player}, won: {won}, result: {result['result']}")

        if player == 1:
            self.player1_wins.append(won)
        elif player == 2:
            self.player2_wins.append(won)

        return loss.item(), won

    async def train_episode(self):
        async with websockets.connect(self.GAME_API_URI) as self.websocket:
            try:
                data = await self.websocket.recv()

                value, pieces, player = self.get_pieces(data)

                await self.websocket.send(pieces)

                result = await self.websocket.recv()

                result = self._parse_result(result)

                loss, won = self.compute_loss(value, result, player)

                return loss, won, player

            except websockets.exceptions.ConnectionClosedOK:
                print("Connection wa closed.")

        return None, None, None

    async def train_verbose(self, num_epochs, window_size):
        print(f"Starting training for {num_epochs} episodes...")
        print("Game: Tactical positioning with line-of-sight and FOV mechanics\n")

        # Setup live plotting
        plt.ion()
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, :])  # Loss plot (full width)
        ax2 = fig.add_subplot(gs[1, :])  # Overall win rate (full width)
        ax3 = fig.add_subplot(gs[2, 0])  # Player 1 stats
        ax4 = fig.add_subplot(gs[2, 1])  # Player 2 stats

        recent_wins = []

        for episode in range(num_epochs):
            loss, won, player = await self.train_episode()

            self.losses.append(loss)
            recent_wins.append(won)

            # Keep only recent wins for win rate calculation
            if len(recent_wins) > window_size:
                recent_wins.pop(0)

            win_rate = sum(recent_wins) / len(recent_wins) if recent_wins else 0
            self.win_rates.append(win_rate)

            # Update plot every 10 episodes
            if episode % 10 == 0 and episode > 0:
                # Plot 1: Loss over time
                ax1.clear()
                ax1.plot(self.losses, 'b-', alpha=0.3, linewidth=0.5)
                if len(self.losses) > 50:
                    smoothed = np.convolve(self.losses, np.ones(50)/50, mode='valid')
                    ax1.plot(range(49, len(self.losses)), smoothed, 'r-', linewidth=2, label='Smoothed (50 ep)')
                ax1.set_xlabel('Episode')
                ax1.set_ylabel('Loss')
                ax1.set_title('Training Loss')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # Plot 2: Overall win rate
                ax2.clear()
                ax2.plot(self.win_rates, 'g-', linewidth=2, label='Win Rate')
                ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random (50%)')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Win Rate')
                ax2.set_title(f'Rolling Win Rate (window={window_size})')
                ax2.set_ylim((0.0, 1.0))
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                # Plot 3: Player 0 performance
                ax3.clear()
                if len(self.player1_wins) > 0:
                    p1_rate = [sum(self.player1_wins[max(0, i-window_size):i+1]) / 
                                min(i+1, window_size)
                              for i in range(len(self.player1_wins))]
                    ax3.plot(p1_rate, 'b-', linewidth=2)
                    ax3.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
                    ax3.set_title(f'Player 1 Win Rate\n(n={len(self.player1_wins)})')
                else:
                    ax3.set_title('Player 1 Win Rate\n(no games yet)')
                ax3.set_ylim((0, 1))
                ax3.grid(True, alpha=0.3)

                # Plot 4: Player 1 performance
                ax4.clear()
                if len(self.player2_wins) > 0:
                    p2_rate = [sum(self.player2_wins[max(0, i-window_size):i+1]) / 
                                min(i+1, window_size)
                              for i in range(len(self.player2_wins))]
                    ax4.plot(p2_rate, 'orange', linewidth=2)
                    ax4.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
                    ax4.set_title(f'Player 2 Win Rate\n(n={len(self.player2_wins)})')
                else:
                    ax4.set_title('Player 2 Win Rate\n(no games yet)')
                ax4.set_ylim((0, 1))
                ax4.grid(True, alpha=0.3)

                plt.pause(0.01)

            # Print progress
            if (episode + 1) % 100 == 0:
                p1_wr = sum(self.player1_wins) / len(self.player1_wins) if self.player1_wins else 0
                p2_wr = sum(self.player2_wins) / len(self.player2_wins) if self.player2_wins else 0
                print(f"Episode {episode + 1}/{num_epochs}")
                print(f"  Loss: {loss:.4f} | Overall WR: {win_rate:.2%}")
                print(f"  Player 1 WR: {p1_wr:.2%} ({len(self.player1_wins)} games)")
                print(f"  Player 2 WR: {p2_wr:.2%} ({len(self.player2_wins)} games)\n")

        plt.ioff()
        plt.show()
        print("\n" + "="*50)
        print("Training complete!")
        print("="*50)

        # Final statistics
        final_wr = sum(recent_wins) / len(recent_wins) if recent_wins else 0
        p1_final = sum(self.player1_wins) / len(self.player1_wins) if self.player1_wins else 0
        p2_final = sum(self.player2_wins) / len(self.player2_wins) if self.player2_wins else 0

        print(f"\nFinal Statistics:")
        print(f"  Overall Win Rate: {final_wr:.2%}")
        print(f"  Player 1 Win Rate: {p1_final:.2%} ({len(self.player1_wins)} games)")
        print(f"  Player 2 Win Rate: {p2_final:.2%} ({len(self.player2_wins)} games)")

    async def train_non_verbose(self, num_epochs, window_size):
        recent_wins = []

        for _ in range(num_epochs):
            loss, won, _ = await self.train_episode()

            self.losses.append(loss)
            recent_wins.append(1 if won else 0)

            # Keep only recent wins for win rate calculation
            if len(recent_wins) > window_size:
                recent_wins.pop(0)

            win_rate = sum(recent_wins) / len(recent_wins) if recent_wins else 0
            self.win_rates.append(win_rate)

    async def train(self, num_epochs, window_size):
        if self.verbose:
            await self.train_verbose(num_epochs, window_size)
        else:
            await self.train_non_verbose(num_epochs, window_size)

    def save_model(self, path='game_ai_model.pth'):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses,
            'win_rates': self.win_rates,
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path='game_ai_model.pth'):
        """Load trained model"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.losses = checkpoint.get('losses', [])
        self.win_rates = checkpoint.get('win_rates', [])
        print(f"Model loaded from {path}")

async def main():
    parser = argparse.ArgumentParser(description="Process a string and check for a verbose flag.")

    parser.add_argument(
        'path', 
        type=str, 
        help='Provide a string for model saving path.'
    )

    parser.add_argument(
        '-v', 
        '--verbose', 
        action='store_true', 
        help='Set the verbose flag.'
    )

    args = parser.parse_args()
    verbose = args.verbose
    print(f"Verbose: {verbose}")

    model = Game_Model()

    trainer = Trainer(model, verbose, learning_rate=0.001)

    await trainer.train(num_epochs=10000, window_size=100)

    # path = args.path
    # trainer.save_model(f'tactical_game_ai_{path}.pth')

if __name__ == "__main__":
    asyncio.run(main())
