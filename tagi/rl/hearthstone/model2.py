import torch

from fireplace import cards
from hearthstone.enums import CardClass, CardType

from utils import get_collection
from config import CardConfig, HandConfig, HeroConfig, MinionConfig, GameConfig, ModelConfig
from trainer import Trainer
# from agent import GameModel

# initialize cards db
cards.db.initialize('zhCN')

# initialize cards
collection, collection_dict = get_collection(CardClass.MAGE)

# config
cards_list = ['变形术', '火球术', '水元素', '寒冰箭', '巫师学徒']
cards_list = [(card_name, collection_dict[card_name].id) for card_name in cards_list]
cards_list = [('幸运币', 'GAME_005')] + cards_list

card_config = CardConfig(
    cards_list = cards_list,
    card_embedding_size = 32,
    card_type_num = 2,
    card_type_embedding_size = 2,
    hidden_size = 32,
    state_size = 32,
    max_cost = 10,
    max_attack = 15,
    max_health = 15,
    card_property_size = 5,
    with_property = False)

hand_config = HandConfig(
    max_hand_cards = 10, 
    max_deck_cards = 10, 
    hand_embedding_size = 256,
    hidden_size = 128,
    state_size = 64)

hero_config = HeroConfig(
    max_attack = 10,
    max_health = 30, 
    max_mana = 10, 
    state_size = 3)

minion_config = MinionConfig(
    max_attack = 10, 
    max_health = 10)

game_config = GameConfig(
    card_config, hand_config, hero_config, minion_config, 
    targets_hidden_size = 64, action_hidden_size = 64)

model_config = ModelConfig(
    model_name='model2', seed=520, 
    epoch=100, round_num=100, learning_rate=0.001, batch_size=64, optim='Adam')

player1_model_data = torch.load('./models/model1/69_0.94.pth')
player1_model = GameModel(player1_model_data['game_config'])

player2_model_data = torch.load('./models/model1/69_0.94.pth')
player2_model = GameModel(player2_model_data['game_config'])

trainer = Trainer(model_config, game_config, 
            player1_model=player1_model, player2_model=player2_model, model_dir='models')

trainer.train()
