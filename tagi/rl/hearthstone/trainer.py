import os
import time
import torch
import random
import torch.nn.functional as F
import numpy as np

from visdom import Visdom
from hearthstone.enums import CardClass
from fireplace.game import Game
from fireplace.exceptions import GameOver

from agent import HearthStoneGod
from game_model import GameModel


class HearthStoneDataset(torch.utils.data.Dataset):
    def __init__(self, game_data):
        self.game_data = []
        for data in game_data:
            self.game_data.extend([(d[0], d[1], d[2], data[1]) for d in data[0]])

    def __len__(self):
        return len(self.game_data)

    def __getitem__(self, index):
        obs, action, target, reward = self.game_data[index]

        hand = obs['hand']

        hero = obs['hero']
        current_minions = obs['characters']['current']

        opponent = obs['opponent']
        opponent_minions = obs['characters']['opponent']

        action_mask = obs['action_mask']
        targets_mask = obs['targets_mask']

        return hand, hero, current_minions, opponent, opponent_minions, action_mask, targets_mask, action, target, reward


class Trainer:
    def __init__(self, model_config, game_config, player1_model=None, player2_model=None, model_dir='.'):
        self.model_config = model_config
        self.game_config = game_config
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        if player1_model is not None:
            self.player1_model = player1_model
        else:
            self.player1_model = GameModel(game_config)
        self.player1_model.to(self.device)
        
        if player2_model is not None:
            self.player2_model = player2_model
            self.player2_model.to(self.device)
        else:
            self.player2_model = 'random'

        self.model_dir = model_dir
        self.vis = Visdom()
        trace = dict(x=[1, 2, 3], y=[4, 5, 6], mode="markers+lines", type='custom',
                     marker={'color': 'red', 'symbol': 104, 'size': "10"},
                     text=["one", "two", "three"], name='1st Trace')
        layout = dict(title="First Plot", xaxis={'title': 'x1'}, yaxis={'title': 'x2'})

        self.vis._send({'data': [trace], 'layout': layout, 'win': 'mywin'})

    def collect_game_data(self):
        win_num = 0
        game_data = []
        start_time = time.time()

        for game_round in range(self.model_config.round_num):
            # initialize game
            deck1 = [c[1] for c in self.game_config.card_config.cards_list[1:] * 4]
            player1 = HearthStoneGod('Player1', deck1, CardClass.MAGE.default_hero,
                                        self.game_config, game_model=self.player1_model, device=self.device)

            deck2 = [c[1] for c in self.game_config.card_config.cards_list[1:] * 4]
            player2 = HearthStoneGod('Player2', deck2, CardClass.MAGE.default_hero,
                                        self.game_config, game_model=self.player2_model, device=self.device)

            game = Game(players=(player1, player2))
            game.start()

            # play game
            # mulligan
            player1.mulligan(skip=True)
            player2.mulligan(skip=True)

            try:
                while True:
                    player = game.current_player
                    player.play_turn(game)
                    
            except GameOver:
                if player2.hero.dead:
                    win_num += 1
                    game_data.append((player1.replay, 1))
                else:
                    game_data.append((player1.replay, -1))
                # print(game_round, "Game completed normally.")

        end_time = time.time()
        win_rate = win_num / self.model_config.round_num

        return game_data, win_rate

    def train(self):
        # set seed
        random.seed(self.model_config.seed)
        torch.manual_seed(self.model_config.seed)
        np.random.seed(self.model_config.seed)

        device = self.device

        if self.model_config.optim == 'Adam':
            optimizer = torch.optim.Adam(self.player1_model.parameters(), self.model_config.learning_rate)
        elif self.model_config.optim == 'SGD': 
            optimizer = torch.optim.SGD(self.player1_model.parameters(), self.model_config.learning_rate)
        else:
            raise NotImplementedError(self.model_config.optim + " is not implemented!")

        best_win_rate = 0
        for epoch in range(self.model_config.epoch):
            game_data, win_rate = self.collect_game_data()

            if win_rate > best_win_rate or epoch % 10 == 0:
                self.save(epoch, win_rate)
                best_win_rate = win_rate
                
            print(f'epoch {epoch} win rate:', win_rate)

            dataset = HearthStoneDataset(game_data)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.model_config.batch_size, shuffle=True)
            
            for step, data in enumerate(dataloader):
                optimizer.zero_grad()

                hand, hero, current_minions, opponent, opponent_minions, \
                        action_mask, targets_mask, action, target, reward = data

                hand, hero, current_minions = hand.to(device), hero.to(device), current_minions.to(device)
                opponent, opponent_minions = opponent.to(device), opponent_minions.to(device)
                action_mask, targets_mask = action_mask.to(device), targets_mask.to(device)
                action, target, reward = action.to(device), target.to(device), reward.to(device)

                # actions
                action_policy, action_logits, game_state = self.player1_model.get_action(
                    hand, hero, current_minions, opponent, opponent_minions, action_mask)

                action_loss = - (reward * action_policy.log_prob(action)).mean()
                # _action_mask = F.one_hot(action, self.game_config.action_size).squeeze(1)
                # log_probs = torch.sum(_action_mask * F.log_softmax(action_logits, dim=-1), 1)
                # action_loss = - torch.mean(reward * log_probs)

                # targets
                targets_policy = self.player1_model.get_target(action_logits, game_state, targets_mask)
                target_loss = - (reward * targets_policy.log_prob(target)).mean()
                # _targets_mask = F.one_hot(target, self.game_config.targets_size).squeeze(1)
                # log_probs = torch.sum(_targets_mask * F.log_softmax(targets_logits, dim=-1), 1)
                # target_loss = - torch.mean(reward * log_probs)

                loss = action_loss + target_loss

                if step % 100 == 0:
                    print(loss.item())
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.player1_model.parameters(), 20, norm_type=2)
                optimizer.step()
    
    def save(self, epoch, win_rate):
        save_data = {
            'game_config': self.game_config,
            'model_config': self.model_config, 
            'state_dict': self.player1_model.state_dict(), 
            'epoch': epoch,
            'win_rate': win_rate
        }

        save_dir = os.path.join(self.model_dir, self.model_config.model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_path = os.path.join(save_dir, f'{epoch}_{win_rate}.pth')
        torch.save(save_data, save_path)
