import sif_parser
import matplotlib.pyplot as plt

# Open the SIF file
# The first index is the frame and the second, third index is the pixel
def openSIF(filename):
    data = sif_parser.xr_open(filename)
    # Access the data array
    image_data = data.values
    return image_data

# Plot the frames as a movie with a colorbar
def plot_frames(image_data, scale = []):
    fig, ax = plt.subplots()
    cbar = None
    # Plot the image
    for i, img in enumerate(image_data):
        ax.clear()
        cax = ax.imshow(img, cmap="gray", vmin = scale[0], vmax = scale[1])
        if cbar is None:
            cbar = fig.colorbar(cax, ax=ax)
        else:
            cbar.update_normal(cax)
        ax.set_title(f"Image {i}")
        plt.pause(0.01)
        plt.waitforbuttonpress()


if __name__ == "__main__":
    filename = '/Users/tommtommbom/Desktop/qdrepos/andor_data/2025_01_22/beads/1_free&tether&stuck.sif'
    image_data = openSIF(filename)
    plot_frames(image_data, [0,10000])
    
