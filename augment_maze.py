import imageio
from PIL import Image
import maze_utils
import numpy as np


def save_output_image(i: int, output: list, dir: str):
    img_mode = 'RGB'
    s = ""
    img_out = []
    for elm in output:
        s = "".join(elm)
        img_out.append(list(map(int, s.replace("#", "1").replace(" ", "0"))))
    arr = np.asarray(img_out)
    arr[arr == 0] = 0
    arr[arr == 1] = 255

    # add 2 rows, 2 columns to achieve 32x32 images
    size = len(output)
    z = np.zeros((size, 1))
    z[z == 0] = 255
    arr = np.append(arr, z, axis=1)
    w = np.zeros((1, size + 1 ))
    w[w == 0] = 255
    arr = np.append(arr, w, axis=0)
    
    im = Image.fromarray(arr)
    if im.mode != img_mode:
        im = im.convert(img_mode)
    imageio.imsave(dir + str(i) + ".png", im)


def main():

    n_samples = 1000
    n = 1
    p = 0.5
    size = 9

    train_dir = "imgs\\train\\"
    test_dir = "imgs\\test\\"

    for i in range(n_samples):
        grid = np.random.binomial(n, p, size=(size, size))
        grid = maze_utils.preprocess_grid(grid, size)
        output = maze_utils.carve_maze(grid, size)
        save_output_image(i, output, train_dir)


if __name__ == '__main__':
    main()