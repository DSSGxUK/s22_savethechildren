# Convert MobNet pretrained model to suitable form for
# transfer learning:
# Base idea is to
# (i) add trainable Conv layer at start, to compress
# down to 3 channels as acceptable by pretrained
# architecture, and
# (ii) replace FC layer at end with alt trainable FC
# layer, that outputs to desired num classes (base
# assumed 5, for quintiles)
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

print("MobNet orig:")
print(mobilenet_v2())

print("MobNet transforms:")
print(MobileNet_V2_Weights.DEFAULT.transforms())

# ResNet50 example
# from torchvision.io import read_image
# from torchvision.models import resnet50, ResNet50_Weights

# img = read_image("test/assets/encode_jpeg/grace_hopper_517x606.jpg")

# # Step 1: Initialize model with the best available weights
# weights = ResNet50_Weights.DEFAULT
# model = resnet50(weights=weights)
# model.eval()

# # Step 2: Initialize the inference transforms
# preprocess = weights.transforms()

# # Step 3: Apply inference preprocessing transforms
# batch = preprocess(img).unsqueeze(0)

# # Step 4: Use the model and print the predicted category
# prediction = model(batch).squeeze(0).softmax(0)
# class_id = prediction.argmax().item()
# score = prediction[class_id].item()
# category_name = weights.meta["categories"][class_id]
# print(f"{category_name}: {100 * score:.1f}%")

# ...
