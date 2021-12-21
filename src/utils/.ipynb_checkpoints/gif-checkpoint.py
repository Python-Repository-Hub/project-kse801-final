import os
import tqdm
import imageio


def gif_generator(load_path, save_path):
    ''' Generate gif from series figures
    Args:
        load_path: <str> folder contain figures in png format
        save_path: <str> generated gif saved place
    '''
    with imageio.get_writer(save_path, mode='I') as writer:
        for root, dirs, files in os.walk(load_path):
            for file in tqdm.tqdm(files):
                image = imageio.imread(os.path.join(root, file), '.png')
                writer.append_data(image)