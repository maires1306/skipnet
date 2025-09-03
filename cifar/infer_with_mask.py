# infer_with_mask.py
import torch
import torchvision
import torchvision.transforms as transforms
from models import cifar10_resnet_38_withmask

def evaluate(model, loader, device, mask):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, mask=mask)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset CIFAR-10
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    # Carregar modelo base com máscara
    model = cifar10_resnet_38_withmask().to(device)
    ckpt = torch.load("save_checkpoints/cifar10_resnet_38/model_best.pth.tar", map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict)
    model.eval()

    num_blocks = 18
    masks_to_test = [
        [1] * num_blocks,                        # baseline
        [0] * num_blocks,                        # tudo pulado
        [1 if i % 2 == 0 else 0 for i in range(num_blocks)],  # alternando
        [1] * (num_blocks//2) + [0] * (num_blocks//2),        # só primeira metade
        [0] * (num_blocks//2) + [1] * (num_blocks//2),        # só segunda metade
    ]

    for idx, mask in enumerate(masks_to_test):
        acc = evaluate(model, testloader, device, mask)
        print(f"Mask {idx} ({mask}): Acc = {acc:.2f}%")
