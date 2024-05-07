import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max_pool_1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool_2(x)
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def preprocessImage(imagePath):
    image = Image.open(imagePath)
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)


def train(data, model):
    model.train()

    learning_rate = 0.01
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    num_epochs = 5
    p = 1
    with open("loss.txt", "wt") as f:
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, sample in enumerate(data, 0):
                optimizer.zero_grad()
                # print(sample[0])
                # print(sample[1])
                inputs = sample[0]
                # img = np.reshape(inputs, (1, 1, 28, 28)) / 255
                # img = torch.from_numpy(img)
                # img = img.type(torch.FloatTensor)
                labels = sample[1]

                output = model(inputs)
                loss = criterion(output, labels)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 500 == 499:  # print every 500 mini-batches
                    print('[%d, %d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 500))
                    s = "{0} {1}\n".format(p, running_loss / 500)
                    f.write(s)
                    p += 1
                    running_loss = 0.0

    torch.save(model.state_dict(), './model.pth')

def validation(data, model):
    model.eval()
    print("Validating...")
    show_image = False

    size = len(data)
    num_incorrect = 0
    i = 0
    for sample in data:
        images, labels = sample
        img = transforms.functional.to_pil_image(images[0][0], mode='L')
        img.save("img_{}.png".format(i), "png")
        output = model(images)
        predicted = torch.max(output.data, 1)
        if labels[0] != predicted[1].item():
            num_incorrect += 1
            if show_image:
                s = "Real: {0}\t Predicted: {1}".format(labels[0], predicted[1].item())
                print(s)
                my_imshow(torchvision.utils.make_grid(images))
        i += 1
    print("Validation Error: {0} %".format(100.0 * num_incorrect / size))

def my_imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def sliding_window(model, image, patch_size=(28, 28)):
    """
    Implement a sliding window to recognize numbers in any location in a given image.
    We do not expect numbers to be rotated, so this is much simplified.
    """
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(patch_size),  # Resize to the desired patch size
        transforms.ToTensor()
    ])

    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    height, width = image.size
    patch_height, patch_width = patch_size

    # Iterate over the image with a sliding window
    for y in range(0, height - patch_height + 1, patch_height):
        for x in range(0, width - patch_width + 1, patch_width):
            # Extract a patch from the image
            patch = image.crop((x, y, x + patch_width, y + patch_height))

            # Apply transformation to the patch and pass through the model
            with torch.no_grad():
                patch_tensor = transform(patch).unsqueeze(0)
                output = model(patch_tensor)
                _, predicted = torch.max(output, 1)
                predicted_class = predicted.item()

                # If a number is detected (class != 0), draw a rectangle around the patch
                if predicted_class != 0:
                    draw = ImageDraw.Draw(image)
                    draw.rectangle([x, y, x + patch_width, y + patch_height], outline="red")

    # Display the original image with highlighted regions
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def main():
    transform = torchvision.transforms.Compose([
         torchvision.transforms.Grayscale(),
         torchvision.transforms.Resize(28),
         torchvision.transforms.ToTensor()
    ])

    batch_size_train = 16

    # image = Image.open("data/MNIST/numbers.png")
    # print(image.size)

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform))

    # trainfolder = datasets.ImageFolder("train", transform)
    # train_loader = torch.utils.data.DataLoader(trainfolder, batch_size=batch_size_train, shuffle=True)

    # create instance of a model
    model = LeNet()

    # train new model
    train(train_loader, model)

    # use existing model
    model.load_state_dict(torch.load('./model.pth'))

    validation(test_loader, model)

    # uncoment to run sliding window
    img = Image.load('numbers.png')
    sliding_window(model, img, (28, 28))

if __name__ == '__main__':
    main()