import onnxruntime as ort
available_providers = ort.get_available_providers()
print(available_providers)