from torchvision.utils import save_image
import matplotlib.pyplot as plt
from seq_imagenet100 import Imagenet100Sal

# Istanzi il dataset (usa un subset ridotto per debug)
ds = Imagenet100Sal(root="/path/imagenet100", train=True)

# Prendi un esempio
(img, sal_map), target = ds[0]

print("Image shape:", img.shape)         # torch.Size([3, H, W])
print("Saliency shape:", sal_map.shape)  # torch.Size([1, H', W'])
print("Target:", target)

# Mostra sia immagine che mappa
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(img.permute(1,2,0).numpy())
plt.title("RGB Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(sal_map.squeeze().numpy(), cmap='hot')
plt.title("Saliency Map")
plt.axis("off")

plt.savefig("example_image_saliency.png")
