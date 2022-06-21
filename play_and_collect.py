import argparse
import os
import matplotlib.pyplot as plt
import torch
import cv2
from argparse import ArgumentParser
import os
from time import sleep
import vizdoom as vzd

parser = argparse.ArgumentParser(description='play and collect data')
parser.add_argument('--image-dir', default='images', help='')

args = parser.parse_args(args=[])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    isExist = os.path.exists(args.image_dir)
    if not isExist:
        os.makedirs(args.image_dir + "/1")

    game = vzd.DoomGame()

    game.load_config("scenarios/health_gathering.cfg")
    game.add_game_args("+freelook 1")
    game.set_render_hud(False)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.SPECTATOR)
    game.init()
    episodes = 10
    for i in range(episodes):
        print("Episode #" + str(i + 1))
        frame = 0
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            game.advance_action()
            last_action = game.get_last_action()
            reward = game.get_last_reward()
            path = os.path.join(args.image_dir, f"{i}_{frame}.jpg")
            #plt.imshow(img.transpose(1,2,0))
            #plt.show()
            img = img.transpose(1, 2, 0)[:, :, ::-1]
            cv2.imshow("Original", img)
            cv2.waitKey(1)
            img = cv2.resize(img, (80, 60), interpolation = cv2.INTER_AREA)
            cv2.imwrite(path, img)
            print("State " + str(state.number))
            print("Game variables: ", state.game_variables)
            print("Action:", last_action)
            print("Reward:", reward)
            print("=====================")
            frame += 1

        print("Total reward:", game.get_total_reward())
        sleep(2.0)

    game.close()