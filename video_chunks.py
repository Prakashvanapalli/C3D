""" Given a video and time, Divide the video into chucks based on time and save it to a folder. Leave the end chuck if it is less than time (sec)

Author @Prakash

Resources: https://unix.stackexchange.com/questions/1670/how-can-i-use-ffmpeg-to-split-mpeg-video-into-10-minute-chunks
"""

import cv2
from tqdm import tqdm
import os


# vid_loc = "../../CPG_Video_analytics/All_Vacuuming/all_annotated/Boston_Elizabeth_G_LivingRoom_20150418162435/37_Boston_Elizabeth_G_LivingRoom_20150418162435.mp4"
# save_loc = "video_sample"
# save_format = ".avi"
# required_time = 10

def video_chuncks(vid_loc, save_loc, save_format=".avi", required_time=10, verbose=False):
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    cap = cv2.VideoCapture(vid_loc)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if verbose:
        print("frames: {} height: {} width: {}".format(frames, width, height))

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_time_in_sec = int(frames/fps)
    if verbose:
        print("frames_per_sec: ", fps)
        print("total time in sec: ", total_time_in_sec)

    #capturing all the frames
    vid = []

    while True:
        ret, img = cap.read()
        if not ret:
            break
        vid.append(img)
    print("video_length", len(vid))

    #sec
    total_frames_to_capture_required_time = int(fps * required_time)
    if verbose:
        print("total_frames_to_capture_required_time:", total_frames_to_capture_required_time)

    ## Now dividing the video into 10 sec chunks ..
    for i in range(int(frames/total_frames_to_capture_required_time)+1):
        start_frame = i * total_frames_to_capture_required_time
        end_frame = (i+1) * total_frames_to_capture_required_time
        if end_frame-start_frame < total_frames_to_capture_required_time:
            capture = vid[start_frame:]
        else:
            capture = vid[start_frame:end_frame]
        ## Capture the video
        video_save_loc = save_loc+vid_loc.rsplit("/")[-1].rsplit(".")[0]+"_"+str(i)+save_format
        out = cv2.VideoWriter(video_save_loc, cv2.VideoWriter_fourcc('M','J','P','G') , fps, (width, height))
        for frame in capture:
            out.write(frame)
        out.release()
        if verbose:
            print("video saved at", video_save_loc)


def video_chunks_ffpmeg(vid_loc, save_loc, seconds=10):
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    video_save_loc = save_loc+vid_loc.rsplit("/")[-1].rsplit(".")[0]+"_"
    k = "ffmpeg -i {} -c copy -map 0 -segment_time {} -f segment -reset_timestamps 1 {}%03d.mp4".format(vid_loc, seconds, video_save_loc)
    os.system(k)


if __name__ == '__main__':
    for num, i in enumerate(tqdm(open("txt_files/cpg_others.lst", "r"))):
        classs = i.split()[1]
        loc = i.split()[0]
        #video_chuncks(loc, "CPG_VIDEO_CHUNKS/", save_format=".avi", required_time=10, verbose=False)
        video_chunks_ffpmeg(loc, "CPG_VIDEO_OTHERS_CHUNKS/")
