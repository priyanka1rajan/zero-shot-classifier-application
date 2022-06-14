import cv2
from PIL import Image
import time
from datetime import datetime
from dateutil import parser
import json

from utils.weather_service import WeatherService
from utils.postgres_io import PostgresIO
from utils.opencv_utils import OpencvUtils
from utils.clip_classifier import CLIP

def log_data(frame_arr, offsets, timestamp_arr, fps, width, height):
    """
    This function processes video segment to identify if a valid object has been detected. For positive detection,
    it logs the data including object probabilities, weather report, and image/video data. Data is stored in MySQL.
    """

    clip = CLIP()
    record_clip_flag = False
    base_location = '/usr/share/grafana/public/img/trail_pics/'
    base_grafana_location = '/public/img/trail_pics/'

    # check if it's day or dark. If it's dark, we will stored video clips for each positive detection along with images.
    who, indx, frame_probs = clip.classifier([frame_arr[0]], offsets, ['dark', 'day'])
    if frame_probs[0]['dark'] > frame_probs[0]['day']:
        record_clip_flag = True

    record_clip_flag = False

    # Process all frames via CLIP classifier to ascertain any positive detection. We could have passed just the middle
    # frame but that may lead of false positives as documentation on Readme shows.
    who, indx, frame_probs = clip.classifier(frame_arr, offsets)

    if who is not None:
        print(who, ' detected at ', timestamp_arr[indx])

        timestamp_to_use = int(parser.parse(timestamp_arr[indx]).timestamp())

        opencv_utils = OpencvUtils()
        # save the frame
        bg_color = (0, 0, 0)  # white
        grafana_frame_url = base_grafana_location + str(timestamp_to_use) + '.png'
        frame_location = base_location + str(timestamp_to_use) + '.png'
        frame = frame_arr[indx]
        opencv_utils.draw_label(frame, timestamp_arr[indx], (10, 50), bg_color)

        # testing..
        text = str(frame_probs[0])
        text = text.replace('\'', '')
        text = text.replace('{', '')
        text = text.replace('}', '')
        text = 'Likelihood (%): ' + text
        opencv_utils.draw_label(frame, text, (10, 100), bg_color)

        cv2.imwrite(frame_location, frame)

        if 0:
            PIL_image = Image.open(frame_location)
            display(PIL_image)

        # save the clip
        if record_clip_flag:
            grafana_clip_url = base_grafana_location + str(timestamp_to_use) + '.mp4'
            clip_location = base_location + str(timestamp_to_use) + '.mp4'

            fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
            out = cv2.VideoWriter(clip_location, fourcc, fps, (width, height))

            k = 0
            # Embed timestamp on all frames
            for frame, timestamp in zip(frame_arr, timestamp_arr):
                cv2.waitKey(1)
                text = timestamp
                opencv_utils.draw_label(frame, text, (int(0.45 * width), 50), bg_color)
                text = json.dumps(str(frame_probs[k]))
                opencv_utils.draw_label(frame, text, (0, 0), bg_color)
                out.write(frame)
                k += 1

            out.release()
        else:
            grafana_clip_url = None

        # log all data to postgres
        day = datetime.today().strftime('%A');

        obj = WeatherService()
        temperature, humidity, conditions = obj.get_weather_report()

        postgres = PostgresIO()
        if 0:
            postgres.update_table(timestamp_arr[indx], day, temperature, humidity, conditions, who, frame_probs[indx],
                              grafana_frame_url, grafana_clip_url)


def main():
    """
    This function implements the main loop where we gather live camera feed and apply OpenCV logic to ascertain if a
    motion has been detected. If there is valid motion, we pass the video segment with motion to log_data() to check
    if any desired object has been detected via CLIP classifier. Positive object detections lead to SQL table updates.
    """

    # rtsp_url = 'rtsp://192.168.86.150/axis-media/media.3gp'#73.15.66.192:23570/axis-media/media.amp'

    # Open rtsp stream (this could be offline video stream as well)
    rtsp_url = '../rtsp/output5.avi'
    vcap = cv2.VideoCapture(rtsp_url)
    fps = vcap.get(cv2.CAP_PROP_FPS)
    width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("fps = ", fps, " height = ", height, " width = ", width)

    # sanity check ...
    if fps > 10:
        print('some issue with fps decoding...')
        exit_flag = True

    # find out desired offsets to check for objects - this could be the entire frame based on the specific scenario
    h1_offset = int(0.25 * height)
    h2_offset = int(1 * height)

    w1_offset = int(0.3 * width)
    w2_offset = int(0.7 * width)

    offsets = [h1_offset, h2_offset, w1_offset, w2_offset]
    print(h1_offset, h2_offset, w1_offset, w2_offset)

    ret, frame1_orig = vcap.read()
    frame1 = frame1_orig[offsets[0]:offsets[1], offsets[2]:offsets[3]]

    sliding_frame_arr = []
    sliding_frame_timestamp = []
    sliding_frame_arr_duration = int(4 * fps)  # frames corresponding to "X" seconds

    record_clip_arr = []
    record_clip_timestamp = []
    record_clip_duration_before_event_max = int(1.5 * fps)  # num frames corresponding to 1 sec
    record_clip_duration_after_event_max = int(1.5 * fps)  # num frames corresponding to 1 sec
    max_frames_recorded_clip = int(3 * fps)

    event_timestamp = []
    exit_flag = False
    motion_flag = False
    curr_time = time.time()
    frame_count_post_event = None

    cnt = -1

    opencv_utils = OpencvUtils()
    # Loop over all the frames (forever in case of live rtsp stream)
    while exit_flag is False:

        # cv2.waitKey(1)
        cnt += 1
        if cnt % 5 == 0:
            print('time in secs is ', cnt / 5)

        ret, frame2_orig = vcap.read()

        if ret is False:
            print('Cannot read frames, exiting...')
            exit_flag = True
            break

        frame2 = frame2_orig[h1_offset:h2_offset, w1_offset:w2_offset]

        # update the buffer
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

        if len(sliding_frame_arr) < sliding_frame_arr_duration:
            sliding_frame_arr.append(frame2_orig)
            sliding_frame_timestamp.append(timestamp)
        else:
            sliding_frame_arr = sliding_frame_arr[1:] + [frame2_orig]
            sliding_frame_timestamp = sliding_frame_timestamp[1:] + [timestamp]

        # find contours to detect motion
        max_contour_area = opencv_utils.get_max_contour_area(frame1, frame2)

        # check if the motion is valid
        if max_contour_area < 2000:
            if motion_flag is True:  # implies motion period has ended
                motion_flag = False
                frame_count_post_event = 0

        else:
            if motion_flag is False:
                print(max_contour_area)
                if frame_count_post_event is not None:
                    print("finshing recording the previous event as a new one has been detected..")
                    print('clip len: ', len(record_clip_arr))

                    log_data(record_clip_arr, offsets, record_clip_timestamp, fps, width, height)
                    record_clip_arr = []
                    record_clip_timestamp = []
                    frame_count_post_event = None

                motion_flag = True
                record_clip_arr = [frame2_orig]
                record_clip_timestamp = [timestamp]
            else:
                if len(record_clip_arr) < max_frames_recorded_clip:
                    record_clip_arr.append(frame2_orig)
                    record_clip_timestamp.append(timestamp)
                else:
                    # print('finishing recording as current event is too long..')
                    log_data(record_clip_arr, offsets, record_clip_timestamp, fps, width, height)
                    print('clip len: ', len(record_clip_arr))

                    record_clip_arr = []
                    record_clip_timestamp = []
                    frame_count_post_event = None
                    motion_flag = False

        # there is positive motion, gather frames for further processing at log_data()
        if frame_count_post_event is not None:
            if frame_count_post_event < record_clip_duration_after_event_max:
                record_clip_arr.append(frame2_orig)
                record_clip_timestamp.append(timestamp)
                frame_count_post_event += 1
            else:
                print('clip len: ', len(record_clip_arr))
                log_data(record_clip_arr, offsets, record_clip_timestamp, fps, width, height)
                record_clip_arr = []
                record_clip_timestamp = []
                frame_count_post_event = None

        frame1 = frame2

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
