import torch

from fireplace import cards
from fireplace.game import Game
from fireplace.player import Player
from fireplace.deck import Deck
from fireplace.exceptions import GameOver
from hearthstone.enums import CardClass, CardType

from agent import HearthStoneGod
from game_model import GameModel

cards.db.initialize('zhCN')

if __name__ == "__main__":
    model_data = torch.load('./models/model1/7_0.91.pth', map_location='cpu')
    game_config = model_data['game_config']

    game_model = GameModel(game_config)

    deck1 = [c[1] for c in game_config.card_config.cards_list[1:] * 4]
    player1 = HearthStoneGod(
                'Player1', deck1, CardClass.MAGE.default_hero, game_config, manual=True)

    deck2 = [c[1] for c in game_config.card_config.cards_list[1:] * 4]
    player2 = HearthStoneGod(
                'Player2', deck2, CardClass.MAGE.default_hero, game_config, game_model=game_model)

    game = Game(players=(player1, player2))
    game.start()
    
    player1.mulligan(skip=True)
    player2.mulligan(skip=True)

    try:
        while True:
            player = game.current_player
            player.play_turn(game)
            
    except GameOver:
        if game.player2.hero.dead:
            print('You Win!')
        else:
            print('You Lose!')
