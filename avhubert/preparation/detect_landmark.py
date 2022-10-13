# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys,os,pickle,math
import cv2,dlib,time
import numpy as np
from tqdm import tqdm

def load_video(path):
    videogen = skvideo.io.vread(path)
    frames = np.array([frame for frame in videogen])
    return frames

def detect_landmark(image, detector, cnn_detector, predictor):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        rects = detector(gray, 1)
        if len(rects) == 0:
            rects = cnn_detector(gray)
            rects = [d.rect for d in rects]
        coords = None
        for (_, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            coords = np.zeros((68, 2), dtype=np.int32)
            for i in range(0, 68):
                coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

def detect_face_landmarks(face_predictor_path, cnn_detector_path, root_dir, landmark_dir, flist_fn, nshard):
    from concurrent.futures import ThreadPoolExecutor
    def callback(future):
        finished[future.result()] = 1
        print(f'currently finished:{sum(finished)}')
    pool = ThreadPoolExecutor(max_workers=64)
    input_dir = root_dir #
    output_dir = landmark_dir #
    fids = [ln.strip() for ln in open(flist_fn).readlines()]
    num_per_shard = math.ceil(len(fids)/nshard)
    finished = [0] * nshard
    for rank in range(nshard):
        start_id, end_id = num_per_shard*rank, num_per_shard*(rank+1)
        fids_per_shard = fids[start_id: end_id]
        future = pool.submit(detect_face_landmarks_per_shard, face_predictor_path, cnn_detector_path, input_dir, output_dir, fids_per_shard, rank)
        future.add_done_callback(callback)
        print(f"task {rank} submitted :{len(fids_per_shard)} files")
    pool.shutdown()
    unfinished_idx = [i for i, x in enumerate(finished) if x == 0]
    print(*unfinished_idx)
    if len(unfinished_idx) == 0:
        print(f'ALL {sum(finished)} SHARDS ACCOMPLISHED.')
    

def detect_face_landmarks_per_shard(face_predictor_path, cnn_detector_path, input_dir, output_dir, fids_per_shard, rank):
    detector = dlib.get_frontal_face_detector()
    cnn_detector = dlib.cnn_face_detection_model_v1(cnn_detector_path)
    predictor = dlib.shape_predictor(face_predictor_path)
    
    
    for fid in tqdm(fids_per_shard):
        output_fn = os.path.join(output_dir, fid+'.pkl')
        video_path = os.path.join(input_dir, fid+'.mp4')
        frames = load_video(video_path)
        landmarks = []
        for frame in frames:
            landmark = detect_landmark(frame, detector, cnn_detector, predictor)
            landmarks.append(landmark)
        os.makedirs(os.path.dirname(output_fn), exist_ok=True)
        pickle.dump(landmarks, open(output_fn, 'wb'))
    return rank


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='detecting facial landmarks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root', type=str, help='root dir')
    parser.add_argument('--landmark', type=str, help='landmark dir')
    parser.add_argument('--manifest', type=str, help='a list of filenames')
    parser.add_argument('--cnn_detector', type=str, default='mmod_human_face_detector.dat', help='path to cnn detector (download and unzip from: http://dlib.net/files/mmod_human_face_detector.dat.bz2)')
    parser.add_argument('--face_predictor', type=str, default='shape_predictor_68_face_landmarks.dat', help='path to face predictor (download and unzip from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)')
    parser.add_argument('--nshard', type=int, default=1, help='number of shards')
    parser.add_argument('--ffmpeg', type=str, help='ffmpeg path')
    args = parser.parse_args()
    import skvideo
    skvideo.setFFmpegPath(os.path.dirname(args.ffmpeg))
    print(skvideo.getFFmpegPath())
    import skvideo.io
    detect_face_landmarks(args.face_predictor, args.cnn_detector, args.root, args.landmark, args.manifest, args.nshard)
