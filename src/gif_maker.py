import glob
import imageio



anim_file = './images_from_seed_images/generation.gif'


with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('./images_from_seed_images/image_at_epoch_*.png')
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)
