import itertools
import os
import sys
from datetime import date
from threading import Thread
from time import sleep

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

            prediction = self.model(spectrogram_batch).cuda()
            loss = self.loss_fn(prediction, emotion_prediction_batch)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            spectrogram_per_batch = self.config['batch_size']
            loss, current = loss.item(), batch * spectrogram_per_batch
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

                # best_window_batch = []
                # best_window_predictions = []
                #
                # if self.config['windowing']:
                #     for i in range(0, len(spectrogram_batch)):
                #         windows = iou.spectrogram_windowing(spectrogram_batch[i]).cuda()
                #         emotion_label = emotion_prediction_batch[i]
                #         window_predictions = self.model(windows)
                #         correct_label_index = torch.argmax(emotion_label)
                #         correct_pred_column = window_predictions[:, correct_label_index.item()]
                #         best_window = torch.argmax(correct_pred_column)
                #         best_window_batch.append(best_window.item())
                #         best_window_predictions.append(window_predictions[best_window.item()])
                #
                #         self.log_file.write(
                #             f'\n\nBest window for {iou.get_labels[correct_label_index.item()]} is window {best_window.item()}/5'
                #             f'\nwith prediction: {window_predictions[best_window.item()]}\n')

                prediction = self.model(spectrogram_batch).cuda()

                # if self.config['windowing']:
                #     print(f'\nPredictions with best window {best_window_batch} with {best_window_predictions}'
                #           f'\nPrediction on full spectrogram: {prediction}\n---end-----\n\n\n')
                #     self.log_file.write(
                #         f'\nPredictions with best window {best_window_batch} with {best_window_predictions}'
                #         f'\nPrediction on full spectrogram: {prediction}\n---end-----\n\n\n')

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

    # def test_model(self, model):
    #     size = len(self.eval_dataloader.dataset)
    #     num_batches = len(self.eval_dataloader)
    #     test_loss, correct = 0, 0
    #
    #     print('\n----- Starting test loop -----\n')
    #
    #     with torch.no_grad():
    #         for batch, (spectrogram_batch, emotion_prediction_batch) in enumerate(self.eval_dataloader, 0):
    #             spectrogram_batch = spectrogram_batch.cuda()
    #             emotion_prediction_batch = emotion_prediction_batch.cuda()
    #
    #             best_window_batch = []
    #             best_window_predictions = []
    #
    #             if self.config['windowing']:
    #                 for i in range(0, len(spectrogram_batch)):
    #                     windows = iou.spectrogram_windowing(spectrogram_batch[i]).cuda()
    #                     emotion_label = emotion_prediction_batch[i]
    #                     window_predictions = model(windows)
    #                     correct_label_index = torch.argmax(emotion_label)
    #                     correct_pred_column = window_predictions[:, correct_label_index.item()]
    #                     best_window = torch.argmax(correct_pred_column)
    #                     best_window_batch.append(best_window.item())
    #                     best_window_predictions.append(window_predictions[best_window.item()])
    #
    #                     self.log_file.write(
    #                         f'\n\nBest window for {iou.get_labels[correct_label_index.item()]} is window {best_window.item()}/5'
    #                         f'\nwith prediction: {window_predictions[best_window.item()]}\n')
    #
    #             prediction = model(spectrogram_batch).cuda()
    #
    #             if self.config['windowing']:
    #                 print(f'\nPredictions with best window {best_window_batch} with {best_window_predictions}'
    #                       f'\nPrediction on full spectrogram: {prediction}\n---end-----\n\n\n')
    #                 self.log_file.write(
    #                     f'\nPredictions with best window {best_window_batch} with {best_window_predictions}'
    #                     f'\nPrediction on full spectrogram: {prediction}\n---end-----\n\n\n')
    #
    #             loss = self.loss_fn(prediction, emotion_prediction_batch)
    #             test_loss += loss.item()
    #             correct += (prediction.argmax(axis=1) == emotion_prediction_batch.argmax(axis=1)).type(
    #                 torch.float).sum().item()
    #
    #     test_loss /= num_batches
    #     correct /= size
    #     print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    #     self.log_file.write(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    #
    #     return correct, test_loss

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


"""

----------------------------------------- TRAINER FOR VIDEO --------------------------------------------------

"""


class TrainerFrames:
    def __init__(self, model, train_dataloader, eval_dataloader, criterion, optimizer, loss_fn, config):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.best_metric = 0.0
        self.loss_fn = loss_fn
        self.log_file = open(os.path.join(config['save_file_path'], str(date.today()) + '.txt'), 'a+')
        self.epoch_loss_data = []

    def train_loop(self):
        size = len(self.train_dataloader.dataset)

        if self.config['resume_training'] is True:
            checkpoint = torch.load(
                os.path.join(self.config['exp_path'], self.config['exp_name_frame'], 'latest_checkpoint.pkl'),
                map_location=self.config['device'])
            self.model.load_state_dict(checkpoint['model_weights'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        for batch, (input, emotion_prediction) in enumerate(self.train_dataloader, 0):
            # Compute prediction and loss
            input = input.cuda()
            emotion_prediction = emotion_prediction.cuda()
            pred = self.model(input).cuda()
            loss = self.loss_fn(pred, emotion_prediction)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # if batch % 2 == 0:
            #     loss, current = loss.item(), batch * len(X)
            #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            x_length = len(input)
            loss, current = loss.item(), batch * x_length
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            self.log_file.write(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]\n")

    def test_loop(self):
        size = len(self.eval_dataloader.dataset)
        num_batches = len(self.eval_dataloader.dataset) // self.eval_dataloader.batch_size
        test_loss, correct = 0, 0
        with torch.no_grad():
            for batch, (input, emotion_prediction) in enumerate(self.eval_dataloader, 0):
                input = input.cuda()
                emotion_prediction = emotion_prediction.cuda()
                pred = self.model(input).cuda()
                loss = self.loss_fn(pred, emotion_prediction)
                test_loss += loss.item()
                correct += (pred.argmax(axis=1) == emotion_prediction.argmax(axis=1)).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        self.log_file.write(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        return correct, test_loss

    def save_net_state(self, epoch, latest=False, best=False):
        if latest is True:
            path_to_save = os.path.join(self.config['exp_path'], self.config['exp_name_frame'],
                                        f'latest_checkpoint.pkl')
            to_save = {
                'epoch': epoch,
                'model_weights': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }
            torch.save(to_save, path_to_save)
        elif best is True:
            path_to_save = os.path.join(self.config['exp_path'], self.config['exp_name_frame'], f'best_model.pkl')
            to_save = {
                'epoch': epoch,
                'stats': self.best_metric,
                'model_weights': self.model.state_dict()
            }
            torch.save(to_save, path_to_save)
        else:
            path_to_save = os.path.join(self.config['exp_path'], self.config['exp_name_frame'],
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

        print(f"Loss data: {sum(self.epoch_loss_data) / len(self.epoch_loss_data)}")
        self.log_file.write(f"Loss data: {sum(self.epoch_loss_data) / len(self.epoch_loss_data)}\n")
        print("Done!")
        self.log_file.write("Done!\n")
        self.log_file.close()
