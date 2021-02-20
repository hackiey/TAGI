import torch

from torch.distributions.categorical import Categorical
from config import CardConfig, HandConfig, HeroConfig, MinionConfig, GameConfig


class CardModel(torch.nn.Module):
    def __init__(self, card_config: CardConfig):
        super(CardModel, self).__init__()

        self.card_config = card_config
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
        # if self.card_config.with_property:
        #     cards_properties = self.card_properties(cards)
        #     card_embedding = torch.cat(cards_embedding, cards_properties)

        hidden = self.linear1(cards_embedding)
        hidden = torch.relu(hidden)
        state = self.linear2(hidden)
        state = torch.tanh(state)
        return state


class HandModel(torch.nn.Module):
    def __init__(self, card_config: CardConfig, hand_config: HandConfig, card_model: CardModel):
        super(HandModel, self).__init__()

        self.card_config = card_config
        self.hand_config = hand_config
        self.card_model = card_model

        self.linear1 = torch.nn.Linear(hand_config.max_hand_cards * card_config.state_size, hand_config.hidden_size)
        self.linear2 = torch.nn.Linear(hand_config.hidden_size, hand_config.state_size)

    def forward(self, hand_obs):
        cards_embedding = self.card_model(hand_obs)
        cards_embedding = torch.flatten(cards_embedding, start_dim=1)

        hidden = self.linear1(cards_embedding)
        hidden = torch.relu(hidden)
        state = self.linear2(hidden)
        state = torch.tanh(state)
        return state


class HeroModel(torch.nn.Module):
    def __init__(self, hero_config: HeroConfig):
        super(HeroModel, self).__init__()
        self.hero_config = hero_config

        self.linear1 = torch.nn.Linear(self.hero_config.state_size, self.hero_config.state_size)

    def forward(self, hero_obs):
        hero_state = self.linear1(hero_obs)
        hero_state = torch.tanh(hero_state)
        return hero_state


class MinionModel(torch.nn.Module):
    def __init__(self, minion_config: MinionConfig):
        super(MinionModel, self).__init__()
        self.minion_config = minion_config

        max_minions = self.minion_config.max_minions
        state_size = self.minion_config.state_size
        self.linear1 = torch.nn.Linear(max_minions * state_size, max_minions * state_size)

    def forward(self, minion_obs):
        minion_state = self.linear1(minion_obs)
        minion_state = torch.tanh(minion_state)
        return minion_state


class GameModel(torch.nn.Module):
    def __init__(self, game_config: GameConfig):
        super(GameModel, self).__init__()
        self.game_config = game_config
        self.card_config = game_config.card_config
        self.hand_config = game_config.hand_config
        self.hero_config = game_config.hero_config
        self.minion_config = game_config.minion_config

        self.card_model = CardModel(self.card_config)
        self.hand_model = HandModel(self.card_config, self.hand_config, self.card_model)
        self.hero_model = HeroModel(game_config.hero_config)
        self.minion_model = MinionModel(self.minion_config)

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

    def get_action(self, hand_obs, hero_obs, current_minions_obs, opponent_obs, opponent_minions_obs, action_mask=None):
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
        action_logits = torch.tanh(action_logits)
        action_logits = self.action(action_logits)

        action_logits = action_logits - 1e30 * (1 - action_mask)
        action_policy = Categorical(logits=action_logits)

        return action_policy, action_logits, game_state

    def get_target(self, action_logits, game_state, targets_mask):

        targets_logits = self.targets_hidden(torch.cat([game_state, action_logits], dim=-1))
        targets_logits = torch.tanh(targets_logits)
        targets_logits = self.targets(targets_logits)

        targets_logits = targets_logits - 1e30 * (1 - targets_mask)
        targets_policy = Categorical(logits=targets_logits)

        return targets_policy
