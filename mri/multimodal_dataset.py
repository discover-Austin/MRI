from mri_production_complete import create_test_patterns

def get_multimodal_dataset():
    patterns = create_test_patterns(size=(128, 128))
    dataset = [
        {
            "image": patterns["square"],
            "image_description": "A geometric shape with four equal sides and four right angles.",
            "text": "A simple square shape in the center of the image."
        },
        {
            "image": patterns["circle"],
            "image_description": "A round shape with no corners or edges.",
            "text": "A perfect circle in the middle of the frame."
        },
        {
            "image": patterns["stripes"],
            "image_description": "A pattern of parallel lines.",
            "text": "A series of horizontal stripes, creating a lined pattern."
        },
        {
            "image": patterns["diagonal"],
            "image_description": "A line segment connecting two non-adjacent vertices of a polygon.",
            "text": "A straight line running from one corner to the other."
        },
        {
            "image": patterns["checkerboard"],
            "image_description": "A board with a pattern of alternating light and dark squares.",
            "text": "A classic checkerboard pattern of alternating squares."
        }
    ]
    return dataset
