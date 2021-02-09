import torch

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
        # if self.card_config.with_property:
        #     cards_properties = self.card_properties(cards)
        #     card_embedding = torch.cat(cards_embedding, cards_properties)

        hidden = self.linear1(cards_embedding)
        hidden = torch.sigmoid(hidden)
        state = self.linear2(hidden)
        state = torch.sigmoid(state)
        return state


class HandModel(torch.nn.Module):
    def __init__(self, card_config: CardConfig, hand_config: HandConfig, card_model: CardModel):
        super(HandModel, self).__init__()

        self.card_config = card_config
        self.hand_config = hand_config

        # self.card_model = CardModel(card_config)
        self.card_model = card_model

        self.linear1 = torch.nn.Linear(hand_config.max_hand_cards * card_config.state_size, hand_config.hidden_size)
        self.linear2 = torch.nn.Linear(hand_config.hidden_size, hand_config.state_size)

    def forward(self, hand_obs):
        cards_embedding = self.card_model(hand_obs)
        cards_embedding = torch.flatten(cards_embedding, start_dim = 1)

        hidden = self.linear1(cards_embedding)
        hidden = torch.sigmoid(hidden)
        state = self.linear2(hidden)
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

    def get_action(self, hand_obs, hero_obs, current_minions_obs, opponent_obs, opponent_minions_obs):
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

    def get_target(self, action_logits, game_state, card=None, hero=None, minion=None):

        # if 0 <= action < 10:
        #     card_id = self.card_model.cards_dict[card.name]
        #     card_state = self.card_model(card_id)
        #     logits = self.card_model(card_state)
        # elif 10 <= action <= 17:
        #
        # elif action == 18:
        #     logits =
        # elif action == 19:
        #     logits = torch.zeros(self.game_config.action_state_size)

        targets_logits = self.targets_hidden(torch.cat([game_state, action_logits], dim=-1))
        targets_logits = torch.sigmoid(targets_logits)
        targets_logits = self.targets(targets_logits)

        return targets_logits
