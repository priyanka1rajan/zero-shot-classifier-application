# Real-time classification of moving objects using "CLIP"
I recently came across a zero-shot text-image pretrained model from OpenAI called Contrastive Language-Image Pre-Training aka CLIP. CLIP has been trained on massive amounts of text-image datasets and can be really useful to build applications where no training datasets are available (or you want something that works out-of-the-box). As the name suggests, CLIP works quite well on even on image classes that it hasn't seen before (unlike ResNet50 for example). Interetingly, you can also use CLIP framework to build a text-based query engine for images. You can find out more about CLIP [here](https://arxiv.org/abs/2103.00020) and [here](https://openai.com/blog/clip/).

In this project, we will use CLIP to build a zero-shot classification application for real-time monitoring and classification of moving objects. Analyzed data will be stored in a MySQL database and visualizaton rendered using Grafana. For details, read on...

## How CLIP works?
It's really very simple. Grab an image along with right set of object keywords to query it and CLIP will assign probabilities for each of those keywords (*all probabilities sum up to 1*). Here's CLIP-processed video snippet where each frame in the video clip is queried against keywords *'railway track'*, *'pedestrian'*, and *'dog'*. 

https://user-images.githubusercontent.com/68397302/171525580-8fb6e6fd-a4ec-44e6-8de6-b1c2ca7ba811.mov

And here's the code snippet for CLIP query:

```python
fixed_objects = ['railway track']
moving_object = ['pedestrian', 'dog']
test_image = "trail.png"

objects = fixed_objects + moving_objects

model, preprocess = clip.load("ViT-B/32", device="cuda")
text = clip.tokenize(objects).to(device)

image = preprocess(Image.open(test_image)).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

probabilities = list(probs[0])
print(probabilities)
```
### Some obervations:
* If we were to use only *'pedestrian'* and *'dog'* keywords to query the image, then CLIP will assign higher probablity to the keyword object which is present in the image. However, for scenarios where both are absent or present, roughly *0.5* probabilities each will be assigned to both the labels. Hence, it's always best to include another keyword object that's always present in order to distinguish all possible scenarios.
* It's very important to choose the right object keywords. For example, in order to detect *'dog'* and *'pedestrian'*, you need to pick up a 3rd object label as well that is always present. Obvious choice is *'railway track'* but you could also choose its synonymns such as *'railroad'* , *'rail'* etc. They all will  produce slightly differing outcomes. 
* As you an see from the CLIP-processed video clip, overall CLIP does quite a good job in accurately classifying the objects. But you still need to build some averaging logic that gives you output based on a collection of contiguous frames.  

![Detection probabilities](https://github.com/priyanka1rajan/CLIP-Application/blob/main/detection_probabilities.png?raw=true)


## Let's build a cool application! 
The idea is to analyze real-time camera feed (RTSP) to classify target objects using CLIP. And ofcourse, it's desirable to not pass all frames to CLIP as this will burden your CPU/GPUs. Instead, we will make use of OpenCV library to pass only selected frames where motion is detected (using frame difference approach to detect motion). The following code snippet shows how to compute max contour size associated with two subsequent frames. 


```python
    def get_max_contour_area(self, frame1, frame2):

        # Process image frame to get contours corresponding to perceived movements relative to frame1
        # You may have to play with the threshold value based on your usecase
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, self.gaussian_ksize, cv2.BORDER_DEFAULT)
        _, thresh = cv2.threshold(blur, self.image_threshold, self.image_maxval, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Compute the areas for each contour and return max contour size
        contour_areas = list(map(lambda x: cv2.contourArea(x), contours))
```

Note *'self.image_threshold'* is a key parameter that needs to be tuned according to given scenario. Following two clips show contours based on threhold values of 10 and 75 respectively. For our setup, we pick threshold = 75 as this leads to motion detection only when desired objects are in view.

**Image Thresholding = 75**

https://user-images.githubusercontent.com/68397302/171956035-498a1ae2-a3d8-4389-a39f-21d8b09510c1.mov

**Image Thresholding = 10**

https://user-images.githubusercontent.com/68397302/171956063-4ce14ceb-debd-4413-bdaa-074d45a4bfce.mov

### Putting it all together
The following figure shows implementation logic for this application. We first acquire streaming image and use OpenCV library to detect motion. Once motion is detected, we collect frames around motion and pass on the video snippet to CLIP module for processing. Given that CLIP processing time can range from 0.01 - 10 seconds based on your compute capability, we spun a new thread to process the video snippet. If a positive detection is made, results are logged into MySQL database. To make this more interesting we also log local weather report at that timestamp. 

![System](https://github.com/priyanka1rajan/CLIP-Application/blob/main/motion_classifier.png?raw=true)

And finally here's the Grafana dashboard. It has nice and easy to use interface to MySQL and embed images / video clips. Hope you find this useful. If any questions, please do reach out!

![Grafana Dashboard](https://github.com/priyanka1rajan/CLIP-Application/blob/main/Grafana%20Dashboard-1.png?raw=true)
