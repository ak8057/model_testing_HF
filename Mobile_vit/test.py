from gradio_client import Client, handle_file
import time
import json
import os

client = Client("abhaykumar8057/Mobile_vit")

IMAGE_PATH = "pic1.webp"

def assess_model():

    file_size = os.path.getsize(IMAGE_PATH) / 1024

    print("\n===== MODEL TEST START =====")

    print(f"Image size: {file_size:.2f} KB")

    start = time.perf_counter()

    result = client.predict(
        image=handle_file(IMAGE_PATH),
        api_name="/predict"
    )

    end = time.perf_counter()

    latency = end - start

    print(f"\nLatency: {latency*1000:.2f} ms")

    print("\nRaw result:")
    print(json.dumps(result, indent=4))

    # Extract prediction properly
    predicted_class = result["label"]

    confidence = None

    # Find confidence of predicted class
    for item in result["confidences"]:
        if item["label"] == predicted_class:
            confidence = item["confidence"]
            break

    throughput = 1 / latency

    print(f"\nPredicted Class: {predicted_class}")

    print(f"Confidence: {confidence:.4f}")

    print(f"Throughput: {throughput:.2f} requests/sec")

    print("\nAll confidences:")

    for item in result["confidences"]:
        print(f"{item['label']}: {item['confidence']:.4f}")

    print("\n===== TEST COMPLETE =====")


assess_model()
