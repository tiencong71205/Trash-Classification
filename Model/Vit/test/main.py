import torch
from PIL import Image
from torchvision import transforms

# ==== Load TorchScript model ====
model = torch.jit.load("trash_15k_3_cls_torchscript_fixed.pt", map_location="cpu")
model.eval()

print("[INFO] Loaded TorchScript model")

# ==== Label map ====
LABELS = ["Glass", "Paper", "Plastic"]

# ==== Tiền xử lý ảnh ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ==== Dự đoán ====
def classify_image(image_path):
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1).item()
        conf = probs[0][pred].item()

    label = LABELS[pred]
    print(f"✅ {image_path}: {label} ({conf:.2f})")

# ==== Test ====
if __name__ == "__main__":
    classify_image("test.jpg")
