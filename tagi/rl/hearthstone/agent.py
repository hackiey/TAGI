import random
import torch
import torch.nn.functional as F
import numpy as np

from fireplace.card import Card
from fireplace.player import Player

import utils
from config import CardType
from config import CardConfig, HandConfig, HeroConfig, MinionConfig, GameConfig


class CardModel(torch.nn.Module):
    def __init__(self, card_config: CardConfig):
        super(CardModel, self).__init__()

        self.card_config = card_config

        # build card dict
        self.cards_dict = card_config.cards_dict

        # build model
        self.card_embedding = torch.nn.Embedding(card_config.cards_num, card_config.card_embedding_size)
        self.card_type_embedding = torch.nn.Embedding(card_config.card_type_num, card_config.card_type_embedding_size)

        if card_config.with_property:
            self.linear1 = torch.nn.Linear(
                card_config.card_embedding_size + card_config.card_property_size, card_config.hidden_size)
        else:
            self.linear1 = torch.nn.Linear(card_config.card_embedding_size, card_config.hidden_size)

        self.linear2 = torch.nn.Linear(card_config.hidden_size, card_config.state_size)

    def card_properties(self, cards):
        pass

    def forward(self, cards_id):
        cards_embedding = self.card_embedding(cards_id)
        if self.card_config.with_property:
            cards_properties = self.card_properties(cards)
            # card_embedding = torch.cat(cards_embedding, cards_properties)
            
        hidden = self.linear1(cards_embedding)
        hidden = torch.sigmoid(hidden)
        state = self.linear2(hidden)
        state = torch.sigmoid(state)
        return state


class HandModel(torch.nn.Module):
    def __init__(self, card_config: CardConfig, hand_config: HandConfig):
        super(HandModel, self).__init__()

        self.card_config = card_config
        self.hand_config = hand_config       

        self.card_model = CardModel(card_config)
        
        self.linear1 = torch.nn.Linear(hand_config.max_hand_cards * card_config.state_size, hand_config.hidden_size)
        self.linear2 = torch.nn.Linear(hand_config.hidden_size, hand_config.state_size)

    def forward(self, hand_obs):
        cards_embedding = self.card_model(hand_obs)
        cards_embedding = torch.flatten(cards_embedding, start_dim = 1)

        hidden = self.linear1(cards_embedding)
        hidden = torch.sigmoid(hidden)
        state = self.linear2(hidden)
        state = torch.sigmoid(state)
        return state


class HeroModel(torch.nn.Module):
    def __init__(self, hero_config: HeroConfig):
        super(HeroModel, self).__init__()
        self.hero_config = hero_config

    def forward(self, hero_obs):

        return hero_obs


class MinionModel(torch.nn.Module):
    def __init__(self, minion_config: MinionConfig):
        super(MinionModel, self).__init__()
        self.minion_config = minion_config
    
    def forward(self, minion_obs):
        return minion_obs


class GameModel(torch.nn.Module):
    def __init__(self, game_config: GameConfig):
        super(GameModel, self).__init__()
        self.game_config = game_config
        self.card_config = game_config.card_config
        self.hand_config = game_config.hand_config
        self.hero_config = game_config.hero_config
        self.minion_config = game_config.minion_config

        self.hand_model = HandModel(game_config.card_config, game_config.hand_config)
        self.hero_model = HeroModel(game_config.hero_config)
        self.minion_model = MinionModel(game_config.minion_config)

        self.action_hidden = torch.nn.Linear(
            self.hand_config.state_size + self.hero_config.state_size * 2 + 
                self.minion_config.state_size * self.minion_config.max_minions * 2, 
            self.game_config.action_hidden_size)
        self.action = torch.nn.Linear(self.game_config.action_hidden_size, self.game_config.action_size)

        self.targets_hidden = torch.nn.Linear(
            self.hand_config.state_size + self.hero_config.state_size * 2 + 
                self.minion_config.state_size * self.minion_config.max_minions * 2 + 
                self.game_config.action_size, 
            self.game_config.targets_hidden_size)
        self.targets = torch.nn.Linear(self.game_config.targets_hidden_size, self.game_config.targets_size)
    
    def get_action(self, hand_obs, hero_obs, current_minions_obs, opponent_obs, opponent_minions_obs, action_mask):
        hand_state = self.hand_model(hand_obs)
        hero_state = self.hero_model(hero_obs)

        opponent_state = self.hero_model(opponent_obs)

        current_minions_state = self.minion_model(current_minions_obs)
        opponent_minions_state = self.minion_model(opponent_minions_obs)

        game_state = torch.cat([hand_state, 
                    hero_state, current_minions_state, 
                    opponent_state, opponent_minions_state], dim=1)

        # action
        action_logits = self.action_hidden(game_state)
        action_logits = torch.sigmoid(action_logits)
        action_logits = self.action(action_logits)

        return action_logits, game_state

    def get_target(self, action_logits, game_state, targets_mask):
        targets_logits = self.targets_hidden(torch.cat([game_state, action_logits], dim = -1))
        targets_logits = torch.sigmoid(targets_logits)
        targets_logits = self.targets(targets_logits)

        return targets_logits

class HearthStoneGod(Player):
    def __init__(self, name, deck, hero, game_config: GameConfig, game_model=None, manual=False):
        super(HearthStoneGod, self).__init__(name, deck, hero)
        self.name = name

        self.game_config = game_config
        self.card_config = game_config.card_config
        self.hand_config = game_config.hand_config
        self.hero_config = game_config.hero_config
        self.minion_config = game_config.minion_config

        if game_model == 'random':
            self.game_model = None
        else:
            self.game_model = game_model
        self.manual = manual

        # self.random = random
        self.replay = []

    def get_character_by_entity_id(self, entity_id):
        characters_dict = {}
        for character in self.game.characters:
            characters_dict[character.entity_id] = character

        return characters_dict[entity_id]

    def calculate_targets_mask(self, targets_id, targets):
        targets_mask = np.array([0 for target_id in targets_id] + [0])
        for target in targets:
            for i in range(len(targets_id)):
                if target.entity_id == targets_id[i]:
                    targets_mask[i] = 1

        return targets_mask

    def calculate_targets(self, targets_id, available_targets, action_logits, game_state):
        targets_mask = self.calculate_targets_mask(targets_id, available_targets)
        targets_logits = self.game_model.get_target(action_logits, game_state, targets_mask)
        targets_logits = F.softmax(targets_logits - 1e30 * (1-targets_mask), dim=-1)
        target_index = int(targets_logits.argmax().item())
        target = self.get_character_by_entity_id(targets_id[target_index])

        return target

    def serialize_player(self, player):
        hero_obs = np.array([
            player.hero.atk / self.hero_config.max_attack,
            player.hero.health / self.hero_config.max_health, 
            player.mana / self.hero_config.max_mana])

        return hero_obs

    def serialize_minion(self, minion):
        minion_obs = np.array([minion.atk/self.minion_config.max_attack, 
                        minion.health/self.minion_config.max_health, 
                        minion.can_attack()])
        return minion_obs

    def serialize_current_turn(self):
        # hand obs
        padding = np.zeros(self.hand_config.max_hand_cards - len(self.hand)).astype(np.int64)
        hand_obs = np.array([self.card_config.cards_dict[card.data.id] for card in self.hand])
        hand_obs = np.concatenate([hand_obs, padding])

        # characters obs
        if self.game.player1.name == self.name:
            current_player = self.game.player1
            opponent_player = self.game.player2
        else:
            current_player = self.game.player2
            opponent_player = self.game.player1

        # hero obs
        hero_obs = self.serialize_player(current_player)
        opponent_obs = self.serialize_player(opponent_player)

        # minion obs
        current_minions_obs = np.array([self.serialize_minion(minion) for minion in current_player.field])
        padding = np.zeros((self.minion_config.max_minions-len(current_player.field), self.minion_config.state_size))
        if current_minions_obs.shape[0] == 0:
            current_minions_obs = padding
        else:
            current_minions_obs = np.concatenate([current_minions_obs, padding])
        current_minions_obs = current_minions_obs.flatten()

        opponent_minions_obs = np.array([self.serialize_minion(minion) for minion in opponent_player.field])
        padding = np.zeros((self.minion_config.max_minions-len(opponent_player.field), self.minion_config.state_size))
        if opponent_minions_obs.shape[0] == 0:
            opponent_minions_obs = padding
        else:
            opponent_minions_obs = np.concatenate([opponent_minions_obs, padding])
        opponent_minions_obs = opponent_minions_obs.flatten()

        characters_obs = np.concatenate([hero_obs, current_minions_obs, 
                                            opponent_obs, opponent_minions_obs])

        # targets id
        targets_id = [current_player.hero.entity_id]
        # current player
        targets_id += ([minion.entity_id for minion in current_player.characters[1:]] + 
                    [0 for _ in range(self.minion_config.max_minions - len(current_player.characters[1:]))])
        # opponent player
        targets_id += [opponent_player.hero.entity_id]
        targets_id += ([minion.entity_id for minion in opponent_player.characters[1:]] + 
                    [0 for _ in range(self.minion_config.max_minions - len(opponent_player.characters[1:]))])

        # card playable
        card_mask = np.array([card.is_playable() for card in self.hand])
        card_mask = np.concatenate([card_mask, np.zeros(self.hand_config.max_hand_cards - len(self.hand))])
        # characters attack
        attack_mask = np.array([character.can_attack() for character in self.characters])
        attack_mask = np.concatenate([attack_mask, np.zeros(self.game_config.can_attack_num - len(self.characters))])
        # heropower usable
        heropower_mask = np.array([self.hero.power.is_usable()])
        # end turn
        end_turn = np.array([1.0])
        action_mask = np.concatenate([card_mask, attack_mask, heropower_mask, end_turn])

        return {
            'hand': hand_obs.astype(np.int64), 
            'hero': hero_obs.astype(np.float32),
            'opponent': opponent_obs.astype(np.float32),
            'characters': {
                'current': current_minions_obs.astype(np.float32), 
                'opponent': opponent_minions_obs.astype(np.float32)}, 
            'targets_id': targets_id,
            'action_mask': action_mask.astype(np.float32),
            'targets_mask': None
        }

    def mulligan(self, skip=False):
        if skip:
            mull_count = 0
        else:
            mull_count = random.randint(0, len(self.choice.cards))
        
        cards_to_mulligan = random.sample(self.choice.cards, mull_count)
        self.choice.choose(*cards_to_mulligan)

    def play_turn(self, game):

        if self.manual:
            self.manual_play(game)
            return

        if self.game_model:
            with torch.no_grad():
                while True:
                    obs = self.serialize_current_turn()
                    # hand_obs, hero_obs, opponent_obs, current_minions_obs, opponent_minions_obs
                    action_logits, game_state = self.game_model.get_action(
                        torch.LongTensor(obs['hand']).unsqueeze(0),
                        torch.FloatTensor(obs['hero']).unsqueeze(0),
                        torch.FloatTensor(obs['characters']['current']).unsqueeze(0),
                        torch.FloatTensor(obs['opponent']).unsqueeze(0),
                        torch.FloatTensor(obs['characters']['opponent']).unsqueeze(0),
                        torch.FloatTensor(obs['action_mask']).unsqueeze(0))

                    action_logits = F.softmax(action_logits - 1e30 * (1 - obs['action_mask']), dim=-1)

                    action = int(action_logits.argmax(-1).item())
                    targets_id = obs['targets_id']
                    target, target_index = None, self.game_config.targets_size-1
                    targets_mask = np.array([0 for _ in targets_id]+[1])

                    turn_end = False
                    if 0 <= action < 10:
                        # cards
                        card = self.hand[action]
                        if card.requires_target():
                            # targets
                            target = self.calculate_targets(targets_id, card.targets, action_logits, game_state)
                            # targets_mask = self.calculate_targets_mask(targets_id, card.targets)
                            # targets_logits = self.game_model.get_target(action_logits, game_state, targets_mask)
                            # targets_logits = F.softmax(targets_logits - 1e30 * (1-targets_mask), dim=-1)
                            # target_index = int(targets_logits.argmax().item())
                            # target = self.get_character_by_entity_id(targets_id[target_index])

                        card.play(target = target)

                    elif 10 <= action <= 17:
                        character = self.characters[action - 10]
                        # character attack
                        # targets
                        target = self.calculate_targets(targets_id, character.targets, action_logits, game_state)
                        # targets_mask = self.calculate_targets_mask(targets_id, character.targets)
                        # targets_logits = self.game_model.get_target(action_logits, game_state, targets_mask)
                        # targets_logits = F.softmax(targets_logits - 1e30 * (1-targets_mask), dim=-1)
                        # target_index = int(targets_logits.argmax().item())
                        # target = self.get_character_by_entity_id(targets_id[target_index])

                        character.attack(target)

                    elif action == 18:
                        # heropower
                        if self.hero.power.requires_target():
                            # targets
                            target = self.calculate_targets(targets_id, self.hero.power.targets, action_logits, game_state)
                            # targets_mask = self.calculate_targets_mask(targets_id, self.hero.power.targets)
                            # targets_logits = self.game_model.get_target(action_logits, game_state, targets_mask)
                            # targets_logits = F.softmax(targets_logits - 1e30 * (1-targets_mask), dim=-1)
                            # target_index = int(targets_logits.argmax().item())
                            # target = self.get_character_by_entity_id(targets_id[target_index])

                            self.hero.power.use(target=target)
                        else:
                            self.hero.power.use()

                    elif action == 19: 
                        # end turn
                        game.end_turn()
                        turn_end = True

                    obs['targets_mask'] = targets_mask
                    self.replay.append((obs, action, target_index))

                    if turn_end:
                        return game          
        else:
            while True:
                heropower = self.hero.power
                if heropower.is_usable() and random.random() < 0.1:
                    if heropower.requires_target():
                        heropower.use(target=random.choice(heropower.targets))
                    else:
                        heropower.use()
                    continue

                # iterate over our hand and play whatever is playable
                for card in self.hand:
                    if card.is_playable() and random.random() < 0.5:
                        target = None
                        if card.must_choose_one:
                            card = random.choice(card.choose_cards)
                        if card.requires_target():
                            target = random.choice(card.targets)
                        # print("Playing %r on %r" % (card, target))
                        card.play(target=target)

                        if self.choice:
                            choice = random.choice(self.choice.cards)
                            print("Choosing card %r" % (choice))
                            self.choice.choose(choice)

                        continue

                # Randomly attack with whatever can attack
                for character in self.characters:
                    if character.can_attack():
                        character.attack(random.choice(character.targets))

                break

            game.end_turn()
            return game

    def manual_play(self, game):
        obs = self.serialize_current_turn()
        utils.print_game(game, self.name)
        action_mask = obs['action_mask']
        available_actions = []

        for i, mask in enumerate(action_mask):
            if mask < 1:
                continue
            
            if 0 <= i < 10:          # cards
                card = self.hand[i]
                targets = None
                if card.requires_target():
                    targets = card.targets
                available_actions.append(('card', card, targets, i))
            elif 10 <= i <= 17:      # characters
                character = self.characters[i-10]
                available_actions.append(('attack', self.characters[i-10], character.targets, i))
            elif i == 18:       # heropower
                targets = None
                if self.hero.power.requires_target():
                    targets = self.hero.power.targets
                available_actions.append(('power', self.hero.power, targets, i))
            elif i == 19:
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
            game.end_turn()

        return game
            