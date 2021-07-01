import glob
import imageio



GAN_anim_file = './GAN/images_from_seed_images/generation.gif'
GAN_image_files = './GAN/images_from_seed_images/image_at_epoch_*.png'

cluster_anim_file = './Kmeans/clustering_example_images/complicated_clustering.gif'
cluster_image_files = './Kmeans/clustering_example_images/c_image_at_it_*.png'



with imageio.get_writer(cluster_anim_file, mode='I') as writer:
    filenames = glob.glob(cluster_image_files)
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)
