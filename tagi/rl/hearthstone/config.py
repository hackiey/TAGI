from enum import IntEnum


class CardType(IntEnum): 
    SPELL = 0,
    MINION = 1


class CardConfig:
    def __init__(self, 
            cards_list: list,
            card_embedding_size: int, 
            card_type_num: int, card_type_embedding_size: int,
            hidden_size: int, state_size: int,
            max_cost: int, max_attack: int, max_health: int, 
            card_property_size: int, with_property: bool):

        self.cards_list = cards_list
        self.cards_dict = {'padding': 0}
        for card_id in cards_list:
            # card_id: (card_name, card_id)
            self.cards_dict[card_id[1]] = len(self.cards_dict)

        self.cards_num = len(self.cards_dict)
        self.card_embedding_size = card_embedding_size

        self.card_type_num = card_type_num
        self.card_type_embedding_size = card_type_embedding_size

        self.hidden_size = hidden_size
        self.state_size = state_size
        
        self.max_cost = max_cost
        self.max_attack = max_attack
        self.max_health = max_health
        self.card_property_size = card_property_size
        self.with_property = with_property


class HandConfig:
    def __init__(self, max_hand_cards: int, max_deck_cards: int,
                    hand_embedding_size: int, hidden_size: int, state_size: int):
        
        self.max_hand_cards = max_hand_cards
        self.max_deck_cards = max_deck_cards

        self.hand_embedding_size = hand_embedding_size
        self.hidden_size = hidden_size
        self.state_size = state_size


class HeroConfig:
    def __init__(self, max_attack: int, max_health: int, max_mana: int, state_size: int): 
        self.max_attack = max_attack
        self.max_health = max_health
        self.max_mana = max_mana
        self.state_size = state_size


class MinionConfig:
    def __init__(self, max_attack: int, max_health: int):
        self.max_attack = max_attack
        self.max_health = max_health
        self.max_minions = 7
        self.state_size = 3 # [atk: 0, health: 1, can_attack: 2]


class GameConfig:
    def __init__(self, card_config: CardConfig, hand_config: HandConfig, 
                        hero_config: HeroConfig, minion_config: MinionConfig,
                        action_hidden_size: int, action_state_size: int,
                        targets_hidden_size: int):
        self.card_config = card_config
        self.hand_config = hand_config
        self.hero_config = hero_config
        self.minion_config = minion_config
        
        self.can_attack_num = 8

        self.action_hidden_size = action_hidden_size
        self.action_size = 20 # [card: 0-9, characcters_attack: 10-17, heropower: 18, end_turn: 19]
        self.action_state_size = action_state_size

        self.targets_hidden_size = targets_hidden_size
        self.targets_size = 17 # [:0-15]


class ModelConfig:
    def __init__(self, model_name: str, seed: int, 
                epoch: int, round_num: int, 
                learning_rate: float, batch_size: int, optim: str):
        self.model_name = model_name
        self.seed = seed
        self.epoch = epoch
        self.round_num = round_num
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optim = optim
