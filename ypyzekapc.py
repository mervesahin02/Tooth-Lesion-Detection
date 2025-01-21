import sys
import uuid
import zipfile
import torch
from torchvision.transforms import functional as F
import cv2
import numpy as np
import json
import os
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def preprocess_image(image):
    """
    Görüntüyü ön işleme: Gürültü giderme, histogram eşitleme ve yumuşatma.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    equalized = cv2.equalizeHist(denoised)
    processed_image = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
    return processed_image


def load_model(model_path, num_classes=3):
    backbone = resnet_fpn_backbone(backbone_name='resnet101', weights="IMAGENET1K_V1")
    model = MaskRCNN(backbone=backbone, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


def save_predictions_to_json(output_path, tooth_boxes, tooth_scores, lesion_boxes, lesion_scores, image_width, image_height):
    """
    Tahmin sonuçlarını JSON formatında kaydeder.
    """
    output_data = {
        "version": "5.5.0",
        "flags": {},
        "shapes": [],
        "imageWidth": image_width,
        "imageHeight": image_height
    }

    # Dişler için JSON formatı
    for i, (box, score) in enumerate(zip(tooth_boxes, tooth_scores)):
        x_min, y_min, x_max, y_max = box
        points = [
            [float(x_min), float(y_max)],
            [float(x_max), float(y_max)],
            [float(x_max), float(y_min)],
            [float(x_min), float(y_min)]
        ]
        shape = {
            "label": f"dis_{i+1}",
            "points": points,
            "score": float(score),
            "shape_type": "polygon",
            "flags": {}
        }
        output_data["shapes"].append(shape)

    # Lezyonlar için JSON formatı
    for i, (box, score) in enumerate(zip(lesion_boxes, lesion_scores)):
        x_min, y_min, x_max, y_max = box
        points = [
            [float(x_min), float(y_max)],
            [float(x_max), float(y_max)],
            [float(x_max), float(y_min)],
            [float(x_min), float(y_min)]
        ]
        shape = {
            "label": f"lezyon_{i+1}",
            "points": points,
            "score": float(score),
            "shape_type": "polygon",
            "flags": {}
        }
        output_data["shapes"].append(shape)

    with open(output_path, 'w') as json_file:
        json.dump(output_data, json_file, indent=4)
    print(f"Tahmin sonuçları {output_path} dosyasına kaydedildi.")


def StudentFunction(image_path, model_path, original_png_path, seg_png_path, output_json_path):
    """
    Modeli yükler, tahmin yapar, JSON dosyasını oluşturur ve görüntüleri kaydeder.
    """
    # Cihaz seçimi
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Modeli yükle
    model = load_model(model_path, num_classes=3)
    model.to(device)

    # Görüntüyü yükle
    img = cv2.imread(image_path)
    image_height, image_width = img.shape[:2]

    # Orijinal görüntüyü kaydet
    cv2.imwrite(original_png_path, img)

    # Preprocessed görüntüyü oluştur ve kaydet
    processed_img = preprocess_image(img)
    cv2.imwrite(seg_png_path, processed_img)

    # Görüntüyü tensora çevir ve liste içine al
    img_tensor = F.to_tensor(processed_img).to(device)
    img_tensor = [img_tensor]  # Model bir liste bekliyor

    # Tahminleri al
    with torch.no_grad():
        predictions = model(img_tensor)

    # Tahmin sonuçları
    predictions = predictions[0]  # Model bir liste döndürüyor, ilk eleman tahminlerdir
    boxes = predictions["boxes"].cpu().numpy()
    labels = predictions["labels"].cpu().numpy()
    scores = predictions["scores"].cpu().numpy()

    # Filtreleme
    high_conf_indices = scores > 0.5
    boxes = boxes[high_conf_indices]
    labels = labels[high_conf_indices]
    scores = scores[high_conf_indices]

    # Diş ve lezyonları ayır
    tooth_boxes = boxes[labels == 1]
    tooth_scores = scores[labels == 1]
    lesion_boxes = boxes[labels == 2]
    lesion_scores = scores[labels == 2]

    # JSON'a kaydet
    save_predictions_to_json(output_json_path, tooth_boxes, tooth_scores, lesion_boxes, lesion_scores, image_width, image_height)

    return seg_png_path, original_png_path, output_json_path



def CreateUniqueFolder():
    output_dir =r"C:\Users\sydme\Desktop\ypyzekacikti"
    os.makedirs(output_dir, exist_ok=True)
    unique_id = str(uuid.uuid4())[:8]
    outputImgPngPath = os.path.join(output_dir, f"outputImg_{unique_id}.png")
    inputImgOrginalPath = os.path.join(output_dir, f"originalImg_{unique_id}.png")
    outputJsonPath = os.path.join(output_dir, f"outputJson_{unique_id}.json")
    return outputImgPngPath, inputImgOrginalPath, outputJsonPath


def Process(inputs, outputImgPngPath, inputImgOrginalPath, outputJsonPath):
    seg_png_path, original_png_path, output_json_path = StudentFunction(
        inputs["filePath"], inputs["modelPath"], inputImgOrginalPath, outputImgPngPath, outputJsonPath
    )

    zip_path = os.path.join(os.path.dirname(outputImgPngPath), f"outputZip_{uuid.uuid4().hex[:8]}.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(outputImgPngPath, os.path.basename(outputImgPngPath))
        zipf.write(inputImgOrginalPath, os.path.basename(inputImgOrginalPath))
        zipf.write(outputJsonPath, os.path.basename(outputJsonPath))

    Outputs = {
        "inputImgPath": inputImgOrginalPath,
        "outputImgPngPath": outputImgPngPath,
        "outputJsonPath": outputJsonPath,
        "outputZipPath": zip_path,
        "outputImgNrrdPath": [],
        "outputModelvtpPath": []
    }
    return Outputs


def main(inputs):
    outputImgPngPath, inputImgOrginalPath, outputJsonPath = CreateUniqueFolder()
    outputs = Process(inputs, outputImgPngPath, inputImgOrginalPath, outputJsonPath)
    return outputs


if __name__ == "__main__":
    filePath = r"C:\Users\sydme\Desktop\Lezyon.img\dis (1).JPG"
    modelPath = r"C:\Users\sydme\Desktop\dis_ve_lezyon_model2.pth"

    inputs = {
        "filePath": filePath,
        "modelPath": modelPath
    }

    outputs = main(inputs)
    print(outputs)
