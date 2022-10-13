# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys, os, glob, subprocess, shutil, math
from datetime import timedelta
import tempfile
from collections import OrderedDict
from pydub import AudioSegment
from tqdm import tqdm

def read_csv(csv_file, delimit=','):
    lns = open(csv_file, 'r').readlines()
    keys = lns[0].strip().split(delimit)
    df = {key: [] for key in keys}
    for ln in lns[1:]:
        ln = ln.strip().split(delimit)
        for j, key in enumerate(keys):
            df[key].append(ln[j])
    return df

def make_short_manifest(pretrain_dir, output_fn):
    """
    Target: txt files in pretrain dir
    Output: csv file with columns id, text, start, end
    Description:
    Merge words into longer texts of pretrain(if gap between words <= min_interval = 0.4),
    and make a csv file that specifies fid, text, start, end for each text.
    Note that the boundaries between merged words(texts) are average-smoothed based on marginal words on both ends.
    """
    subdirs = os.listdir(pretrain_dir)
    min_interval = 0.4
    max_duration = 15
    df = {'fid': [], 'sent': [], 'start': [], 'end': []}
    for subdir in tqdm(subdirs):
        txt_fns = glob.glob(os.path.join(pretrain_dir, subdir+'/*txt'))
        for txt_fn in txt_fns:
            fid = os.path.relpath(txt_fn, pretrain_dir)[:-4]
            lns = open(txt_fn).readlines()
            raw_text = lns[0].strip().split(':')[-1].strip()
            conf = lns[1].strip().split(':')[-1].strip()  # unused
            word_intervals = []
            """
            find which line starts the column heads WORD, START, END, ASDSCORE in dataset.
            """
            for i_line, ln in enumerate(lns):
                if ln[:4] == 'WORD':
                    start_index = i_line
                    break
            """
            Keep record of useful information along the lines for every word.
            """
            for ln in lns[start_index+1:]:
                word, start, end, score = ln.strip().split()
                word_intervals.append([word, float(start), float(end)])
            """
            If the last word ends after max_duration = 15, return the whole text in this text file.
            """
            if word_intervals[-1][-1] < max_duration:
                df['fid'].append(fid)
                df['sent'].append(raw_text)
                df['start'].append(0)
                df['end'].append(-1)
                continue
            """
            If not, merge words into longer text if possible.
            list of words(word_intervals) -> list of merged words or text(sents, abbv for sentences)
            """
            sents, cur_sent = [], []
            for i_word, (word, start, end) in enumerate(word_intervals):
                if i_word == 0:
                    cur_sent.append([word, start, end])
                else:
                    assert start >= cur_sent[-1][-1], f"{fid} , {word}, start-{start}, prev-{cur_sent[-1][-1]}"
                    if start - cur_sent[-1][-1] > min_interval:
                        sents.append(cur_sent)
                        cur_sent = [[word, start, end]]
                    else:
                        cur_sent.append([word, start, end])
            if len(cur_sent) > 0:
                sents.append(cur_sent)
            """
            Boudary averaging for sents.
            This leaves df a record of merged words with smoothed boundary.
            """
            for i_sent, sent in enumerate(sents):
                df['fid'].append(fid+'_'+str(i_sent))
                sent_words = ' '.join([x[0] for x in sent])
                if i_sent == 0:
                    sent_start = 0
                else:
                    sent_start = (sent[0][1] + sents[i_sent-1][-1][2])/2
                if i_sent == len(sents)-1:
                    sent_end = -1
                else:
                    sent_end = (sent[-1][2] + sents[i_sent+1][0][1])/2
                df['sent'].append(sent_words)
                df['start'].append(sent_start)
                df['end'].append(sent_end)
    durations = [y-x for x, y in zip(df['start'], df['end'])]
    num_long = len(list(filter(lambda x: x > 15, durations)))
    print(f"Percentage of >15 second: {100*num_long/len(durations)}%")
    num_long = len(list(filter(lambda x: x > 20, durations)))
    print(f"Percentage of >20 second: {100*num_long/len(durations)}%")
    """
    csv format:
    fid = sample id + index number in sents.
    (so multiple fids correspond to one id only, as fids were derived from sample id.)
    start = smoothed start time
    end = smoothed end time
    """
    with open(output_fn, 'w') as fo:
        fo.write('id,text,start,end\n')
        for i in range(len(df['fid'])):
            fo.write(','.join([df['fid'][i], df['sent'][i], '%.3f' % (df['start'][i]), '%.3f' % (df['end'][i])])+'\n')
    return

def get_ffmpeg_path(ffmpeg):
    return 'ffmpeg' if ffmpeg is None else ffmpeg

def get_raw2fid(csv_fn):
    df = read_csv(csv_fn)
    raw2fid = OrderedDict()
    """
    raw2fid:
    size:the number of utterances in LRS3(118516)
    keys:(sample id, aka original(raw) fids)
    values:(list of (fid, start, end) corresponding to sample id)
    """
    for fid, start, end in zip(df['id'], df['start'], df['end']):
        if '_' in fid:
            """
            raw_fid = original folder which fid belongs to.
            eg:00001_02(fid)->00001(raw_fid).
            """
            raw_fid = '_'.join(fid.split('_')[:-1])
        else:
            raw_fid = fid
        if raw_fid in raw2fid:
            raw2fid[raw_fid].append([fid, start, end])
        else:
            raw2fid[raw_fid] = [[fid, start, end]]
    return raw2fid

def trim_video_frame(csv_fn, raw_dir, output_dir, ffmpeg, nshard):
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor()
    ffmpeg = get_ffmpeg_path(ffmpeg)
    raw2fid = get_raw2fid(csv_fn)
    num_per_shard = math.ceil(len(raw2fid.keys())/nshard)
    for rank in range(nshard):
        start_id, end_id = num_per_shard*rank, num_per_shard*(rank+1)
        fid_info_shard = list(raw2fid.items())[start_id: end_id]
        """
        fid_info_shard contains (key, value) pair in raw2fid in a list
        """
        print(f"Total videos in current shard: {len(fid_info_shard)}/{len(raw2fid.keys())}")
        future = pool.submit(trim_video_frame_per_block, fid_info_shard, raw_dir, output_dir, ffmpeg, rank)
        future.add_done_callback(lambda future: print(future.result()))
        print(f'task {rank} submitted.')
    pool.shutdown()
        
    

def trim_video_frame_per_block(fid_info_shard, raw_dir, output_dir, ffmpeg, rank):
    decimal, fps = 9, 25
    i_raw = -1
    
    for raw_fid, fid_info in tqdm(fid_info_shard):
        i_raw += 1
        """
        original video path
        """
        raw_path = os.path.join(raw_dir, raw_fid+'.mp4')
        tmp_dir = tempfile.mkdtemp()
        cmd = ffmpeg + " -i " + raw_path + " " + tmp_dir + '/%0' + str(decimal) + 'd.png -loglevel quiet'
        subprocess.call(cmd, shell=True)
        num_frames = len(glob.glob(tmp_dir+'/*png'))
        for fid, start_sec, end_sec in fid_info:
            sub_dir = os.path.join(tmp_dir, fid)
            os.makedirs(sub_dir, exist_ok=True)
            start_sec, end_sec = float(start_sec), float(end_sec)
            if end_sec == -1:
                end_sec = 24*3600
            start_frame_id, end_frame_id = int(start_sec*fps), min(int(end_sec*fps), num_frames)
            imnames = [tmp_dir+'/'+str(x+1).zfill(decimal)+'.png' for x in range(start_frame_id, end_frame_id)]
            for ix, imname in enumerate(imnames):
                shutil.copyfile(imname, sub_dir+'/'+str(ix).zfill(decimal)+'.png')
            output_path = os.path.join(output_dir, fid+'.mp4')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cmd = [ffmpeg, "-i", sub_dir+'/%0'+str(decimal)+'d.png', "-y", "-crf", "20", output_path, "-loglevel", "warning"]

            pipe = subprocess.call(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT) # subprocess.PIPE
        shutil.rmtree(tmp_dir)
    return f'shard {rank} finished.'

def trim_audio(csv_fn, raw_dir, output_dir, ffmpeg, rank, nshard):
    ffmpeg = get_ffmpeg_path(ffmpeg)
    df = read_csv(csv_fn)
    raw2fid = OrderedDict()
    for fid, start, end in zip(df['id'], df['start'], df['end']):
        if '_' in fid:
            raw_fid = '_'.join(fid.split('_')[:-1])
        else:
            raw_fid = fid
        if raw_fid in raw2fid:
            raw2fid[raw_fid].append([fid, start, end])
        else:
            raw2fid[raw_fid] = [[fid, start, end]]
    i_raw = -1
    num_per_shard = math.ceil(len(raw2fid.keys())/nshard)
    start_id, end_id = num_per_shard*rank, num_per_shard*(rank+1)
    fid_info_shard = list(raw2fid.items())[start_id: end_id]
    print(f"Total audios in current shard: {len(fid_info_shard)}/{len(raw2fid.keys())}")
    for raw_fid, fid_info in tqdm(fid_info_shard):
        i_raw += 1
        tmp_dir = tempfile.mkdtemp()
        wav_path = os.path.join(tmp_dir, 'tmp.wav')
        cmd = ffmpeg + " -i " + os.path.join(raw_dir, raw_fid+'.mp4') + " -f wav -vn -y " + wav_path + ' -loglevel quiet'
        subprocess.call(cmd, shell=True)
        raw_audio = AudioSegment.from_wav(wav_path)
        for fid, start_sec, end_sec in fid_info:
            start_sec, end_sec = float(start_sec), float(end_sec)
            if end_sec == -1:
                end_sec = 24*3600
            t1, t2 = int(start_sec*1000), int(end_sec*1000)
            new_audio = raw_audio[t1: t2]
            output_path = os.path.join(output_dir, fid+'.wav')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            new_audio.export(output_path, format="wav")
        shutil.rmtree(tmp_dir)
    return

def trim_pretrain(root_dir, out_dir, ffmpeg, rank, nshard, step=1):
    pretrain_dir = os.path.join(root_dir, 'pretrain')
    print(f"Trim original videos in pretrain")
    csv_fn = os.path.join(out_dir, 'short-pretrain.csv')
    if step == 1:
        print(f"Step 1. Make csv file {csv_fn}")
        make_short_manifest(pretrain_dir, csv_fn)
    else:
        print(f"Step 2. Trim video and audio")
        output_video_dir, output_audio_dir = os.path.join(out_dir, 'short-pretrain'), os.path.join(out_dir, 'audio/short-pretrain/')
        os.makedirs(output_video_dir, exist_ok=True)
        os.makedirs(output_audio_dir, exist_ok=True)
        trim_video_frame(csv_fn, pretrain_dir, output_video_dir, ffmpeg, nshard)
        trim_audio(csv_fn, pretrain_dir, output_audio_dir, ffmpeg, rank, nshard)
    return

def prep_wav(lrs3_root, output, ffmpeg, nshard):
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor()
    output = f"{output}/audio/"
    ffmpeg = get_ffmpeg_path(ffmpeg)
    video_fns = glob.glob(lrs3_root + '/trainval/*/*mp4') + glob.glob(lrs3_root + '/test/*/*mp4')
    video_fns = sorted(video_fns)  # sort is applied as a strategy to make the processing result reproducable.
    num_per_shard = math.ceil(len(video_fns)/nshard)
    for rank in range(nshard):
        start_id, end_id = num_per_shard*rank, num_per_shard*(rank+1)
        video_fns_into_shard = video_fns[start_id: end_id]
        print(f"{len(video_fns_into_shard)} videos")
        future = pool.submit(prep_wav_per_block, video_fns_into_shard, output, ffmpeg, rank)
        future.add_done_callback(lambda future: print(future.result()))
        print(f'task {rank} submitted.')
    pool.shutdown()
        

def prep_wav_per_block(video_fns_into_shard, output_dir, ffmpeg, rank):
    for video_fn in tqdm(video_fns_into_shard):
        """
        Iterating over mp4 files in trainval and test,
        and convert mp4 files to audio(.wav) files in output dir.
        Note that unlike step 1 and 2 which were applied on pretrain set,
        trainval and test sets will not undergo step 1 and 2
        because they are generally shorter and more stable in length.
        
        base_name: trainval(test)/*/*mp4
        output_dir: {output}/audio
        audio_fn: {output}/audio/trainval(test)/*/*wav
        """
        base_name = leave_last_two_slashes(video_fn)
        audio_fn = os.path.join(output_dir, base_name.replace('mp4', 'wav'))
        os.makedirs(os.path.dirname(audio_fn), exist_ok=True)
        cmd = ffmpeg + " -i " + video_fn + " -f wav -vn -y " + audio_fn + ' -loglevel warning'
        subprocess.call(cmd, shell=True)
    return f'shard {rank} finished.'

def leave_last_two_slashes(x):
    """
    trim absolute filename to produce str like:
    /pretrain/*/*.mp4
    /test/*/*.mp4
    ...
    """
    return '/'.join(x.split('/')[-3:])

def get_file_label(lrs3_root, output_dir):
    """
    merge text results for pretrain, test and trainval.
    produce file.list for videos and label.list for labels.
    """
    video_ids_total, labels_total = [], []
    for split in ['trainval', 'test']:
        subdirs = os.listdir(os.path.join(lrs3_root, split))
        for subdir in tqdm(subdirs):
            video_fns = glob.glob(os.path.join(lrs3_root, split, subdir, '*mp4'))
            video_ids = [leave_last_two_slashes(x)[:-4] for x in video_fns]  # ['(trainval or test)/GlpGBSBpxOE/50010', ...]
            for video_id in video_ids:
                txt_fn = os.path.join(lrs3_root, video_id+'.txt') 
                label = open(txt_fn).readlines()[0].split(':')[1].strip()
                labels_total.append(label)
                video_ids_total.append(video_id)
    pretrain_csv = os.path.join(output_dir, 'short-pretrain.csv')
    df = read_csv(pretrain_csv)
    for video_id, label in zip(df['id'], df['text']):
        video_ids_total.append(os.path.join('short-pretrain', video_id))  # ['short-pretrain/GlpGBSBpxOE/00006_1', ...]
        labels_total.append(label)
    video_id_fn, label_fn = os.path.join(output_dir, 'file.list'), os.path.join(output_dir, 'label.list')
    print(video_id_fn, label_fn)
    with open(video_id_fn, 'w') as fo:
        fo.write('\n'.join(video_ids_total)+'\n')
    with open(label_fn, 'w') as fo:
        fo.write('\n'.join(labels_total)+'\n')
    return

if __name__ == '__main__':
    """
    step 1 should be quick and smooth, producing a csv file to be used for step 2 only.
    step 2 relies on the csv file produced from step 1 and is most time-consuming.
    step 3 is independent of any step and can therefore be executed anytime.
    step 4 relies on the csv file produced from step 1 only and can be executed independently of step 2 and 3.
    """
    import argparse
    parser = argparse.ArgumentParser(description='LRS3 preprocess pretrain dir', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lrs3', type=str, help='lrs3 root dir')
    parser.add_argument('--output', type=str, help='output dir')
    parser.add_argument('--ffmpeg', type=str, help='path to ffmpeg')
    parser.add_argument('--rank', type=int, help='rank id', default=0)
    parser.add_argument('--nshard', type=int, help='number of shards', default=1)
    parser.add_argument('--step', type=int, help='Steps (1: split labels, 2: trim video/audio, 3: prep audio for trainval/test, 4: get labels and file list)')
    args = parser.parse_args()
    if args.step <= 2:
        trim_pretrain(args.lrs3, args.output, args.ffmpeg, args.rank, args.nshard, step=args.step)
    elif args.step == 3:
        print(f"Extracting audio for trainval/test")
        prep_wav(args.lrs3, args.output, args.ffmpeg, args.nshard)
    elif args.step == 4:
        get_file_label(args.lrs3, args.output)
