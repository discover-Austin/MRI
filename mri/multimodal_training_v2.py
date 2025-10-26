import numpy as np
from mri_master import MRIMasterSystem
from production_config import production_config
from multimodal_dataset import get_multimodal_dataset
from mri_integration import MRIDataPreprocessor
from scipy.ndimage import zoom

def main():
    print("Starting multimodal training v2...")

    # Initialize the system with the production configuration
    system = MRIMasterSystem(config=production_config)
    preprocessor = MRIDataPreprocessor()

    # Get the dataset
    dataset = get_multimodal_dataset()

    # Encode and associate the patterns
    print("Encoding and associating patterns...")
    pattern_pairs = []
    for item in dataset:
        # Resize image to match field size
        resized_image = zoom(item["image"], [d/s for d, s in zip(production_config.field_size, item["image"].shape)], order=1)
        image_pattern = resized_image
        text_pattern = preprocessor.encode_text_to_field(item["text"], field_size=production_config.field_size)
        pattern_pairs.append((image_pattern, text_pattern))

    iterations = 1000
    checkpoint_interval = 100
    for i in range(0, iterations, checkpoint_interval):
        print(f"Running training iterations {i} to {i + checkpoint_interval}...")
        system.associate(pattern_pairs, iterations=checkpoint_interval)
        metrics = system.get_metrics()
        print(f"  Checkpoint metrics: Avg. Resonance = {metrics['avg_resonance']:.4f}, Field Energy = {metrics['field_statistics']['energy']:.4f}")

    print("Association training complete.")

    # Verification
    print("\nVerifying associations...")

    # Test 1: Predict text from image
    test_image = dataset[0]["image"]
    resized_test_image = zoom(test_image, [d/s for d, s in zip(production_config.field_size, test_image.shape)], order=1)
    predicted_text_pattern = system.predict(resized_test_image)
    
    correct_text_pattern = preprocessor.encode_text_to_field(dataset[0]["text"], field_size=production_config.field_size)
    similarity = system.measure_similarity(correct_text_pattern)
    print(f"Test 1: Similarity of predicted text from image to correct text: {similarity:.4f}")

    # Test 2: Predict image from text
    test_text = dataset[1]["text"]
    test_text_pattern = preprocessor.encode_text_to_field(test_text, field_size=production_config.field_size)
    predicted_image_pattern = system.predict(test_text_pattern)

    correct_image_pattern = dataset[1]["image"]
    resized_correct_image = zoom(correct_image_pattern, [d/s for d, s in zip(production_config.field_size, correct_image_pattern.shape)], order=1)
    similarity = system.measure_similarity(resized_correct_image)
    print(f"Test 2: Similarity of predicted image from text to correct image: {similarity:.4f}")

if __name__ == "__main__":
    main()
