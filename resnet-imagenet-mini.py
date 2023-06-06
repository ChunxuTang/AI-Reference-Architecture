# https://moiseevigor.github.io/software/2022/12/18/one-pager-training-resnet-on-imagenet/
import torch
import torchvision
import torchvision.transforms as transforms
import time
from torch.utils.data.dataloader import default_collate

print(f"Start time: {time.perf_counter()}")

# Set device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device}")

# Set hyperparameters
num_epochs = 5
batch_size = 64
learning_rate = 0.001

# Initialize transformations for data augmentation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=45),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Initialize pytorch profiler
prof = torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/mnist'))


# https://discuss.pytorch.org/t/questions-about-dataloader-and-dataset/806/4
def my_collate(batch):
    batch = filter (lambda x:x is not None, batch)
    return default_collate(batch)

class MyImageFolder(torchvision.datasets.ImageFolder):
    __init__ = torchvision.datasets.ImageFolder.__init__
    def __getitem__(self, index):
        try: 
            return super(MyImageFolder, self).__getitem__(index)
        except Exception as e:
            print(e)

train_dataset = torchvision.datasets.ImageFolder(
    root='/mnt/alluxio/fuse/imagenet-mini/train',
    transform=transform
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Load the ResNet18 model
model = torchvision.models.resnet18(pretrained=True)
print(f"Current time after loading the model: {time.perf_counter()}")

# Parallelize training across multiple GPUs
model = torch.nn.DataParallel(model)

# Set the model to run on the device
model = model.to(device)
print(f"Current time after loading the model to device: {time.perf_counter()}")

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

start_time = time.perf_counter()
# Train the model...
prof.start()
for epoch in range(num_epochs):
    batch_start = time.perf_counter()
    for inputs, labels in train_loader:
        # Move input and label tensors to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        batch_end = time.perf_counter()
        print(f"Loaded input and labels to the device in {batch_end - batch_start:0.4f} seconds")

        # Zero out the optimization
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        batch_start = time.perf_counter()

    # Print the loss for every epoch
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f} at the timestamp {time.perf_counter()}')
    prof.step()

print(f'Finished Training, Loss: {loss.item():.4f}')

end_time = time.perf_counter()
print(f"Training time in {end_time - start_time:0.4f} seconds")

prof.stop()
torch.save(model.state_dict(), "resnet-imagenet-model.pth")
print("Saved PyTorch Model State to resnet-imagenet-model.pth")
