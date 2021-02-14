import cv2
import time


def run_video(video_url, callback, frame_interval=1, delay=30, 
              display=True, win_name=None, win_size=None):

    cap = cv2.VideoCapture(video_url)

    if display:
        win_name = win_name or 'video_display'
        win_size = win_size if win_size is not None else (1024, 768)
        cv2.namedWindow(win_name, 0)
        cv2.resizeWindow(win_name, win_size[0], win_size[1])

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % frame_interval != 0:
            continue
        
        try:
            callback(frame)
        except:
            pass

        if display:
            cv2.imshow(win_name, frame)

            k = cv2.waitKey(delay)
            
            if k == ord('q'):
                break
            elif k == ord('p'):
                waiting = True
            elif k == ord('c'):
                waiting = False
        else:
            time.sleep(0.001)
