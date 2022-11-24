DATASET_SIZE = 5754

num_steps_to_batch_size = {1: 64, 5: 32, 10: 16, 20: 8, 50: 8, 100: 4}
batch_size_to_accumulate = {64: 1, 32: 2, 16: 4, 8: 8, 4: 16}

MIN_CONFIDENCE = 0.2
NUM_HAND_KEYPOINTS = 22
NUM_FACE_KEYPOINTS = 70
