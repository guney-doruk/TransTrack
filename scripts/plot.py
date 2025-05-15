import json
from matplotlib import pyplot as plt

BASE_PATH = "/cta/users/grad4/master/TransTrack/output/finetune/mots_train_no_ignore_from_cocopersonv2_100thepoch_halftrain_halfval"

class File:
    def __init__(self) -> None:
        pass

    def parse_txt(path: str, loss_type: str):
        epochs, train_loss = [], []
        with open(path, 'r') as file:
            for line in file:
                try: 
                    data = json.loads(line)
                    if 'train_loss' in data and 'epoch' in data:
                        epochs.append(data['epoch'])
                        train_loss.append(data[loss_type])
                    else:
                        print('Train Loss & Epoch does not exist...')
                except json.JSONDecodeError:
                    continue
        
        return epochs, train_loss


class Graph:
    def __init__(self) -> None:
        pass

    def save_loss_image(epochs: list, loss: list, name: str):
        plt.plot(epochs, loss)
        plt.xlabel('Epoch')
        plt.ylabel('Train Loss')
        plt.title('Train Loss Graph')
        plt.savefig(BASE_PATH + f'/loss_plots/train_loss{name}.png')

if __name__ == '__main__':
    input_path = BASE_PATH + '/log.txt'
    epochs, train_loss = File.parse_txt(path=input_path, loss_type="train_loss")
    Graph.save_loss_image(epochs=epochs, loss=train_loss, name= "")
    