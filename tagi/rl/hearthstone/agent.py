import random
import torch
import numpy as np

from fireplace.player import Player

import utils
from config import GameConfig


class HearthStoneGod(Player):
    def __init__(self, name, deck, hero, game_config: GameConfig, game_model=None, mode='random',
                 device=torch.device('cpu')):
        super(HearthStoneGod, self).__init__(name, deck, hero)
        self.name = name

        self.game_config = game_config
        self.card_config = game_config.card_config
        self.hand_config = game_config.hand_config
        self.hero_config = game_config.hero_config
        self.minion_config = game_config.minion_config

        self.device = device

        self.mode = mode
        assert self.mode in ['manual', 'random', 'model_training', 'model_play']
        self.game_model = game_model

        self.replay = []

    def get_character_by_entity_id(self, entity_id):
        characters_dict = {}
        for character in self.game.characters:
            characters_dict[character.entity_id] = character

        return characters_dict[entity_id]

    def calculate_targets_mask(self, targets_id, targets):
        targets_mask = np.array([0 for _ in targets_id] + [0])
        for target in targets:
            for i in range(len(targets_id)):
                if target.entity_id == targets_id[i]:
                    targets_mask[i] = 1

        targets_mask = torch.from_numpy(targets_mask).to(self.device)
        return targets_mask

    def calculate_targets(self, targets_id, available_targets, action_logits, game_state):
        targets_mask = self.calculate_targets_mask(targets_id, available_targets)
        targets_policy = self.game_model.get_target(action_logits, game_state, targets_mask)
        if self.mode == 'model_training':
            target_index = int(targets_policy.sample().item())
        else:
            target_index = int(targets_policy.probs.argmax(-1).item())
        target = self.get_character_by_entity_id(targets_id[target_index])

        return target, target_index, targets_mask

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
        if self.game.players[0].name == self.name:
            current_player = self.game.players[0]
            opponent_player = self.game.players[1]
        else:
            current_player = self.game.players[1]
            opponent_player = self.game.players[0]

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

        if self.mode == 'manual':
            utils.manual_play_turn(self)
        elif self.mode == 'random':
            utils.random_play_turn(self)
        elif self.mode.startswith('model'):
            with torch.no_grad():
                while True:
                    obs = self.serialize_current_turn()
                    action_policy, action_logits, game_state = self.game_model.get_action(
                        torch.LongTensor(obs['hand']).unsqueeze(0).to(self.device),
                        torch.FloatTensor(obs['hero']).unsqueeze(0).to(self.device),
                        torch.FloatTensor(obs['characters']['current']).unsqueeze(0).to(self.device),
                        torch.FloatTensor(obs['opponent']).unsqueeze(0).to(self.device),
                        torch.FloatTensor(obs['characters']['opponent']).unsqueeze(0).to(self.device),
                        torch.FloatTensor(obs['action_mask']).unsqueeze(0).to(self.device))

                    if self.mode == 'model_training':
                        action = int(action_policy.sample().item())
                    else:
                        action = int(action_logits.argmax(-1).item())

                    targets_id = obs['targets_id']
                    target, target_index = None, self.game_config.targets_size-1
                    targets_mask = torch.from_numpy(np.array([0 for _ in targets_id]+[1])).to(self.device)

                    turn_end = False
                    if 0 <= action < 10:       # cards
                        card = self.hand[action]
                        if card.requires_target():
                            target, target_index, targets_mask = self.calculate_targets(targets_id, card.targets, action_logits, game_state)

                        card.play(target=target)
                        if self.mode == 'model_play':
                            utils.print_card(self, card, target)

                    elif 10 <= action <= 17:   # character attack
                        character = self.characters[action - 10]
                        target, target_index, targets_mask = self.calculate_targets(targets_id, character.targets, action_logits, game_state)

                        character.attack(target)
                        if self.mode == 'model_play':
                            utils.print_attack(self, character, target)

                    elif action == 18:         # heropower
                        if self.hero.power.requires_target():
                            target, target_index, targets_mask = self.calculate_targets(targets_id, self.hero.power.targets, action_logits, game_state)

                            self.hero.power.use(target=target)
                        else:
                            self.hero.power.use()
                        if self.mode == 'model_play':
                            utils.print_heropower(self, self.hero.power, target)

                    elif action == 19:         # end turn
                        game.end_turn()
                        turn_end = True

                    obs['targets_mask'] = targets_mask
                    self.replay.append((obs, action, target_index))

                    if turn_end:
                        return game
