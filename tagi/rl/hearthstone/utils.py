import random
from fireplace import cards
from termcolor import colored
from hearthstone.enums import CardClass, CardType


def manual_play_turn(agent):
    obs = agent.serialize_current_turn()
    print_game(agent.game, agent.name)
    action_mask = obs['action_mask']
    available_actions = []

    for i, mask in enumerate(action_mask):
        if mask < 1:
            continue

        if 0 <= i < 10:      # cards
            card = agent.hand[i]
            targets = None
            if card.requires_target():
                targets = card.targets
            available_actions.append(('card', card, targets, i))
        elif 10 <= i <= 17:  # characters
            character = agent.characters[i-10]
            available_actions.append(('attack', agent.characters[i-10], character.targets, i))
        elif i == 18:        # heropower
            targets = None
            if agent.hero.power.requires_target():
                targets = agent.hero.power.targets
            available_actions.append(('power', agent.hero.power, targets, i))
        elif i == 19:        # end turn
            available_actions.append(('end_turn', None))

    for i, action in enumerate(available_actions):
        print(i, action[0], action[1])
    print('--------------------------------')

    action_index = int(input(f'Input your action [0-{len(available_actions)-1}]: '))
    action = available_actions[action_index]
    print(action)
    if action[0] == 'card':
        target = None
        if action[2] is not None:
            print('targets', action[2])
            target_index = int(input(f'Please select a target [0-{len(action[2])-1}]: '))
            target = action[2][target_index]
        action[1].play(target=target)
    elif action[0] == 'attack':
        print('targets', action[2])
        target_index = int(input(f'Please select a target [0-{len(action[2])-1}]: '))
        action[1].attack(action[2][target_index])
    elif action[0] == 'power':
        if action[2] is not None:
            print('targets', action[2])
            target_index = int(input(f'Please select a target [0-{len(action[2])-1}]: '))
            action[1].use(action[2][target_index])
        else:
            action[1].use()
    elif action[0] == 'end_turn':
        agent.game.end_turn()
        print(colored('===================================================================' +
                      '===========================', 'red'))
    return agent.game


def random_play_turn(agent):
    while True:
        heropower = agent.hero.power
        if heropower.is_usable() and random.random() < 0.1:
            if heropower.requires_target():
                heropower.use(target=random.choice(heropower.targets))
            else:
                heropower.use()
            continue

        # iterate over our hand and play whatever is playable
        for card in agent.hand:
            if card.is_playable() and random.random() < 0.5:
                target = None
                if card.must_choose_one:
                    card = random.choice(card.choose_cards)
                if card.requires_target():
                    target = random.choice(card.targets)
                # print("Playing %r on %r" % (card, target))
                card.play(target=target)

                if agent.choice:
                    choice = random.choice(agent.choice.cards)
                    print("Choosing card %r" % choice)
                    agent.choice.choose(choice)

                continue

        # Randomly attack with whatever can attack
        for character in agent.characters:
            if character.can_attack():
                character.attack(random.choice(character.targets))

        break

    agent.game.end_turn()
    return agent.game


def get_character_name(character):
    if character is not None:
        return character.controller.name + '-' + str(character)


def print_card(agent, card, target):
    print(colored('%s used card <%s> on %s' % (agent.name, card, get_character_name(target)), 'red'))


def print_attack(agent, character, target):
    print(colored('%s attack %s' % (get_character_name(target), get_character_name(target)), 'red'))


def print_heropower(agent, heropower, target):
    print(colored('%s use %s on %s' % (agent, heropower, get_character_name(target)), 'red'))


def print_game(game, player_name):
    if player_name == game.players[0].name:
        player = game.players[0]
        opponent = game.players[1]
    else:
        player = game.players[1]
        opponent = game.players[0]

    print(colored('============================================= ' + str(game.turn) +
                  ' ==============================================', 'blue'))
    print('health', opponent.hero.health, 'mana', opponent.mana)
    print(f'{len(opponent.hand)} cards')
    print(colored('HAND', 'red'), opponent.hand)
    print(colored('FIELD', 'red'), opponent.field)
    print('----------------------------------------------------------------------------------------------')
    print(colored('FIELD', 'blue'), player.field)
    print(colored('HAND', 'blue'), player.hand)
    print(f'{len(player.hand)} cards')
    print('health', player.hero.health, 'mana', player.mana)
    print('====================================== available actions =====================================')


# utils
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


def play_games():
    pass