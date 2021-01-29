
from fireplace import cards
from fireplace.game import Game
from hearthstone.enums import CardClass, CardType


def get_collection(card_class: CardClass, exclude=[]):
    collection = []
    for card in cards.db.keys():
        if card in exclude:
            continue
        cls = cards.db[card]
        if not cls.collectible:
            continue
        if cls.type == CardType.HERO:
            # Heroes are collectible...
            continue
        if cls.card_class and cls.card_class not in [card_class, CardClass.NEUTRAL]:
            # Play with more possibilities
            continue
        collection.append(cls)

    collection_dict = {}

    for card in collection:
        collection_dict[card.name] = card

    return collection, collection_dict


def print_game(game, player_name):
    if player_name == game.player1.name:
        player = game.player1
        opponent = game.player2
    else:
        player = game.player2
        opponent = game.player1

    print('============================================= ' + str(game.turn) + ' ==============================================')
    print('health', opponent.hero.health, 'mana', opponent.mana)
    print(f'{len(opponent.hand)} cards')
    print(opponent.hand)
    print(opponent.field)
    print('----------------------------------------------------------------------------------------------')
    print(player.field)
    print('health', player.hero.health, 'mana', player.mana)
    print(f'{len(player.hand)} cards')
    print(player.hand)
    print('====================================== availableactions ======================================')


def play_games():
    pass