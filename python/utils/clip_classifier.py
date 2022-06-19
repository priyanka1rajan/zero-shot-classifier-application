import torch
import clip
import numpy as np
from PIL import Image

from dataclasses import dataclass

@dataclass
class CLIP:
    """
    This class implements a CLIP-based classifier which processes a video frame and returns most-likely detected object
    along with object detection probabilities assigned to all the frames
    """
    # hard-coded threshold to make decision on if one of desired object has been detected.
    detection_threshold = 0.4

    def classifier(self, frame_arr, offsets, fixed_objects=['railway track', 'tree'],
                        moving_objects=['pedestrian', 'dog', 'cyclist']):

        # find the right device including GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)

        objects = fixed_objects + moving_objects

        text = clip.tokenize(objects).to(device)

        detected_object = None
        detected_object_prob = 0
        detected_object_frame_indx = 0
        frame_probs = []

        indx = 0
        for img in frame_arr:

            # extract area of interest to apply CLIP
            frame = img[offsets[0]:offsets[1], offsets[2]:offsets[3]]

            PIL_image = Image.fromarray(np.uint8(frame)).convert('RGB')
            # PIL_image = frame
            image = preprocess(PIL_image).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)

                logits_per_image, logits_per_text = model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            probs = list(probs[0])

            # update dictionary list with new set of probabilities
            probabilities_json = {}
            for x, y in zip(objects, probs):
                probabilities_json[x] = round(100 * y, 1)
            frame_probs.append(probabilities_json)

            probs_moving_objects = probs[len(fixed_objects):]

            # logic to decide if one of the desired object is detected
            if sum(probs_moving_objects) > self.detection_threshold:
                max_probs = max(probs_moving_objects)

                if max_probs > detected_object_prob:
                    max_index = probs_moving_objects.index(max_probs)
                    detected_object = moving_objects[max_index]
                    detected_object_frame_indx = indx
                    detected_object_prob = max_probs

            indx += 1

        return detected_object, detected_object_frame_indx, frame_probs
