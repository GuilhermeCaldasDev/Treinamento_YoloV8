from ultralytics import YOLO

def main():
    print("Iniciando Treino.....")

    model = YOLO("yolov8n.pt")

    model.train(data='custom_dataset.yaml', epochs=15, batch=16, workers=8, pretrained= True, resume= False, single_cls=False, box=7.5, cls=0.5, val=True, degrees=0.3, hsv_s=0.3, hsv_v=0.3, scale=0.5, fliplr=0.5)

    print("Finalizando Treino...")


if __name__ == '__main__':
    main()