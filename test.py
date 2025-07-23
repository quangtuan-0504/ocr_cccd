from paddleocr import DocImgOrientationClassification, TextImageUnwarping




model = DocImgOrientationClassification(
    model_name= "PP-LCNet_x1_0_doc_ori",
    model_dir= "model/imgorient/PP-LCNet_x1_0_doc_ori",
    device = "gpu"
)
print(model)

# model = TextImageUnwarping(
#     model_name= "UVDoc",
#     model_dir= "model/UVDoc",
# )

print('######################## vietocr')
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import yaml
with open("model/vietocr/base.yml", encoding = "utf-8") as f:
    base_cf = yaml.safe_load(f)
with open("model/vietocr/vgg-seq2seq.yml", encoding = "utf-8") as f:
    cus_cf = yaml.safe_load(f)
base_cf.update(cus_cf)

vietocr_model = Predictor(Cfg(base_cf))

print(vietocr_model)


print('################### YOLO')
from ultralytics import YOLO
yolo_model = YOLO('model/obb_card/best_n.pt')
print(yolo_model)