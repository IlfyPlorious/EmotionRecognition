import itertools
import os
import sys
from datetime import date
from threading import Thread
from time import sleep
import matplotlib.pyplot as plt

from util import ioUtil as iou
import torch

torch.cuda.empty_cache()

"""

----------------------------------------- TRAINER FOR SPECTROGRAMS --------------------------------------------------

"""


class TrainerSpectrogram:
    def __init__(self, model, train_dataloader, eval_dataloader, criterion, optimizer, scheduler, loss_fn, config):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_metric = 0.0
        self.loss_fn = loss_fn
        self.log_file = open(os.path.join(config['save_file_path'], str(date.today()) + '.txt'), 'a+')
        self.epoch_loss_data = []
        self.loading_testing = True

    def animate(self):
        for c in itertools.cycle(['|', '/', '-', '\\']):
            if not self.loading_testing:
                break
            sys.stdout.write('\rTesting ' + c)
            sys.stdout.flush()
            sleep(0.3)

    def train_loop(self):
        size = len(self.train_dataloader.dataset)

        if self.config['resume_training'] is True:
            checkpoint = torch.load(
                os.path.join(self.config['exp_path'], self.config['exp_name_spec'], 'latest_checkpoint.pkl'),
                map_location=self.config['device'])
            self.model.load_state_dict(checkpoint['model_weights'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        # enumerate ads a counter for each element in for loop ( like an i index )
        # this index corresponds to the batch being processed
        # enumerate returns ( index, obj ), and here object is ( spectrogram, label )
        for batch, (spectrogram_batch, emotion_prediction_batch) in enumerate(self.train_dataloader, start=1):
            # Compute prediction and loss

            spectrogram_batch = spectrogram_batch.cuda()
            emotion_prediction_batch = emotion_prediction_batch.cuda()

            prediction, _ = self.model(spectrogram_batch)
            prediction = prediction.cuda()
            loss = self.loss_fn(prediction, emotion_prediction_batch)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if torch.isnan(torch.tensor(loss)):
                print('found a nan')

            spectrogram_per_batch = self.config['batch_size']
            loss, current = loss.item(), batch * spectrogram_per_batch

            if batch % 5 == 0:
                print(f"batch: {batch}  loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                self.log_file.write(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]\n")

    def test_loop(self):
        size = len(self.eval_dataloader.dataset)
        not_nan_loss = 0
        test_loss, correct = 0, 0

        print('\n----- Starting test loop -----\n')
        self.log_file.write('\n----- Starting test loop -----\n')

        # starting loading animation
        self.loading_testing = True
        t = Thread(target=self.animate)
        t.start()

        with torch.no_grad():
            for batch, (spectrogram_batch, emotion_prediction_batch) in enumerate(self.eval_dataloader, 0):
                spectrogram_batch = spectrogram_batch.cuda()
                emotion_prediction_batch = emotion_prediction_batch.cuda()

                prediction, _ = self.model(spectrogram_batch)
                prediction = prediction.cuda()
                loss = self.loss_fn(prediction, emotion_prediction_batch)

                if not torch.isnan(torch.tensor(loss.item())):
                    test_loss += loss.item()
                    not_nan_loss += 1

                correct += (prediction.argmax(axis=1) == emotion_prediction_batch.argmax(axis=1)).type(
                    torch.float).sum().item()

        test_loss /= not_nan_loss
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        self.log_file.write(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        self.loading_testing = False
        print()

        return correct, test_loss


    def save_net_state(self, epoch, latest=False, best=False):
        if latest is True:
            path_to_save = os.path.join(self.config['exp_path'], self.config['exp_name_spec'], f'latest_checkpoint.pkl')
            to_save = {
                'epoch': epoch,
                'model_weights': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }
            torch.save(to_save, path_to_save)
        elif best is True:
            path_to_save = os.path.join(self.config['exp_path'], self.config['exp_name_spec'], f'best_model.pkl')
            to_save = {
                'epoch': epoch,
                'stats': self.best_metric,
                'model_weights': self.model.state_dict()
            }
            torch.save(to_save, path_to_save)
        else:
            path_to_save = os.path.join(self.config['exp_path'], self.config['exp_name_spec'], f'latest_checkpoint.pkl')
            to_save = {
                'epoch': epoch,
                'model_weights': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }
            torch.save(to_save, path_to_save)
            path_to_save = os.path.join(self.config['exp_path'], self.config['exp_name_spec'],
                                        f'model_epoch_{epoch}.pkl')
            torch.save(self.model, path_to_save)

    def run(self):
        self.log_file.write(f'\n\nRunning new training session...\nLogs from {date.today()}\n\n')

        for t in range(self.config['train_epochs']):
            print(f"Epoch {t + 1}\n-------------------------------")
            self.log_file.write(f"Epoch {t + 1}\n-------------------------------\n")
            self.train_loop()
            self.save_net_state(epoch=t + 1)
            accuracy, loss = self.test_loop()
            self.epoch_loss_data.append(loss)
            self.scheduler.step()

        print(f"Loss data: {sum(self.epoch_loss_data) / len(self.epoch_loss_data)}")
        self.log_file.write(f"Loss data: {sum(self.epoch_loss_data) / len(self.epoch_loss_data)}\n")
        print("Done!")
        self.log_file.write("Done!\n")
        self.log_file.close()
