from paddleocr import DocImgOrientationClassification, TextImageUnwarping




model = DocImgOrientationClassification(
    model_name= "PP-LCNet_x1_0_doc_ori",
    model_dir= "model/PP-LCNet_x1_0_doc_ori",
)

# model = TextImageUnwarping(
#     model_name= "UVDoc",
#     model_dir= "model/UVDoc",
# )