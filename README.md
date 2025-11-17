# Simple Game AI
In this project I try to create a ML model that learns how to play a game I'v created by playing against itself.

## Game Explanation
This game is meant to be played in a phone. The phone's screen is split in to two halfs: a top half and a bottom half. Each half belongs to a player, and it's where that player will place
their pieces. Additionally, each half has a bunch of random walls generated in it and the players' king piece. The objective for each player is to place their pieces so thay can attack
the opponent's king piece and defend theirs. Each piece can "kill" a oponent's piece if it has direct line of sight with it. More thorough explanation coming soon.

## 📒 Training Logs
### 17 Novemeber, 2025
#### Current State
I've created a model with a value an policy head. The intention behind the value head is for it to be able to predict if the model will win a game (by outputing 1) or loose it (outputing 0).
This method hasn't worked in my case. In this game piece positions have to be in a certain boundary, for example, `x` and `y` coordinates must be between 0 and 1. If
these values aren't in this interval the training process treats it like a loss. Therefore, what's happening is the model just predicts some random values for `x` and `y` coordinates
(which aren't in the interval) and the value head learns to always predict a loss. This way it makes the loss go to zero: the model is just learning to give invalid values for pieces
and always predict losses.

I guess for games where one of the playes will win and another will loose this method would work, for games where a tie is possible this method alone will often fail.

#### Changes
I will add a loss to the policy get it to output correct values. That way, no incorrect piece positions will be sent. There is still posibility for ties, however it is much more likely that
at least one player will have to win or loose.

#### Predictions
I think this method will still not work. The reason why I believe this is because most times the match result will be a tie, and therefore the value head will always just learn to predict
a tie. I don't expect there to be enough cases where a player wins to influence the model. Additionally, I'm not sure this method trains a model to win at the game. I think it will train it
to predict a reliable result, and I think the most reliable will be a tie.

