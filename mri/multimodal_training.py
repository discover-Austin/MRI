import numpy as np
from mri_master import MRIMasterSystem
from production_config import production_config
from multimodal_dataset import get_multimodal_dataset
from mri_integration import MRIDataPreprocessor

def main():
    print("Starting multimodal training...")

    # Initialize the system with the production configuration
    system = MRIMasterSystem(config=production_config)
    preprocessor = MRIDataPreprocessor()

    # Get the dataset
    dataset = get_multimodal_dataset()

    # Encode and associate the patterns
    print("Encoding and associating patterns...")
    pattern_pairs = []
    for item in dataset:
        image_pattern = preprocessor.encode_text_to_field(item["image_description"], field_size=production_config.field_size)
        text_pattern = preprocessor.encode_text_to_field(item["text"], field_size=production_config.field_size)
        pattern_pairs.append((image_pattern, text_pattern))

    system.associate(pattern_pairs, iterations=100)
    print("Association training complete.")

    # Verification
    print("\nVerifying associations...")

    # Test 1: Predict text from image
    test_image_description = dataset[0]["image_description"]
    test_image_pattern = preprocessor.encode_text_to_field(test_image_description, field_size=production_config.field_size)
    predicted_text_pattern = system.predict(test_image_pattern)
    
    # For this demonstration, we'll measure the similarity to the correct text pattern
    correct_text_pattern = preprocessor.encode_text_to_field(dataset[0]["text"], field_size=production_config.field_size)
    similarity = np.corrcoef(predicted_text_pattern.flatten().real, correct_text_pattern.flatten().real)[0, 1]
    print(f"Test 1: Similarity of predicted text from image to correct text: {similarity:.4f}")

    # Test 2: Predict image from text
    test_text = dataset[1]["text"]
    test_text_pattern = preprocessor.encode_text_to_field(test_text, field_size=production_config.field_size)
    predicted_image_pattern = system.predict(test_text_pattern)

    # For this demonstration, we'll measure the similarity to the correct image pattern
    correct_image_description = dataset[1]["image_description"]
    correct_image_pattern = preprocessor.encode_text_to_field(correct_image_description, field_size=production_config.field_size)
    similarity = np.corrcoef(predicted_image_pattern.flatten().real, correct_image_pattern.flatten().real)[0, 1]
    print(f"Test 2: Similarity of predicted image from text to correct image: {similarity:.4f}")

if __name__ == "__main__":
    main()
