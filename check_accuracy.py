# check_accuracy.py
import torch
from data_loader import get_data_loaders
from victim import build_victim_model, evaluate_model, DEVICE

loaders = get_data_loaders(batch_size=64, data_dir="./data/cifar10")

for arch, ckpt in [
    ("resnet50",        "models/victim_resnet50.pth"),
    ("efficientnet_b0", "models/victim_efficientnet.pth"),
]:
    model = build_victim_model(architecture=arch, pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()

    train_acc = evaluate_model(model, loaders["train"])
    test_acc  = evaluate_model(model, loaders["test"])

    print(f"\n{'='*40}")
    print(f"  {arch}")
    print(f"  Train accuracy : {train_acc:.2f}%")
    print(f"  Test  accuracy : {test_acc:.2f}%")
    print(f"{'='*40}")
