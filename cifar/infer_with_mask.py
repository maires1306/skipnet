import torch
import torchvision
import torchvision.transforms as transforms
from models import cifar10_resnet_38_withmask

# ============
# CONFIGURAÇÃO
# ============
CKPT_PATH = "save_checkpoints/cifar10_resnet_38/model_best.pth.tar"
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============
# DATASET
# ============
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(root="./data", train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)

# ============
# MODELO
# ============
model = cifar10_resnet_38_withmask().to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["state_dict"])
model.eval()

# ============
# FUNÇÃO DE AVALIAÇÃO
# ============
def evaluate_mask(mask, name=""):
    correct, total = 0, 0
    with torch.no_grad():
        for images, targets in testloader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            outputs = model(images, mask=mask)
            preds = outputs.argmax(1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    acc = 100.0 * correct / total
    print(f"[{name}] Acurácia: {acc:.2f}% ({correct}/{total})")
    return acc

# ============
# MÁSCARAS DE TESTE
# ============
masks = {
    "baseline_all_exec": [1]*18,                     # executa todos
    "all_skip": [0]*18,                              # pula todos (só conv1 + fc)
    "half_first": [1]*9 + [0]*9,                     # executa metade inicial
    "half_last": [0]*9 + [1]*9,                      # executa metade final
    "pairs_1100": [1,1,0,0]*4 + [1,1],               # padrão em duplas
    "alternating": [1,0]*9                           # executa a cada 2 blocos
}

# ============
# LOOP
# ============
for name, mask in masks.items():
    evaluate_mask(mask, name=name)
