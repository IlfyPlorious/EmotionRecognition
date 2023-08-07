import json

from torch import cuda
from torch import nn
from torch import optim

from trainers import trainer as tr
from trainers.brain_trainer import BrainTrainer
from data import data_manager_spectrogram
from data.data_manager import DataManager
from networks_files import networks, res_net
from networks_files.brain import Brain
from util.ioUtil import get_audio_files, save_spectrogram_data, write_pretrained_model_features_for_video, \
    initialize_model

config = json.load(open('config.json'))

device = "cuda" if cuda.is_available() else "cpu"
print(f"Using {device} device")

# ## Preprocesare ##
#
# files = get_audio_files(limit=7000)
# for file in files:
#     save_spectrogram_data(file)
#
# write_pretrained_model_features_for_video(initialize_model(num_classes=6, feature_extract=True))


def run_training():
    model = networks.SpectrogramBrain(block=res_net.BasicBlock, layers=[2, 2, 3, 2], num_classes=6).to(device)
    train_dataloader, eval_dataloader = data_manager_spectrogram.DataManagerSpectrograms(
        config).get_train_eval_dataloaders_spectrograms()
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.LinearLR(optimizer=optimizer, total_iters=config['train_epochs'] * 0.1)

    trainer = tr.TrainerSpectrogram(model=model, train_dataloader=train_dataloader,
                                    eval_dataloader=eval_dataloader,
                                    loss_fn=nn.CrossEntropyLoss(), optimizer=optimizer,
                                    scheduler=scheduler,
                                    config=config)

    trainer.run()


# run_training()

def run_brain_training():
    # model = Brain().to(device)
    model = Brain().to(device)
    train_dataloader, eval_dataloader = DataManager(config=config).get_train_eval_dataloaders_audiovideo()
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.LinearLR(optimizer=optimizer, total_iters=config['train_epochs'] * 0.1)

    trainer = BrainTrainer(model=model, train_dataloader=train_dataloader,
                           eval_dataloader=eval_dataloader,
                           loss_fn=nn.CrossEntropyLoss(), optimizer=optimizer,
                           scheduler=scheduler,
                           config=config)

    trainer.run()


run_brain_training()
