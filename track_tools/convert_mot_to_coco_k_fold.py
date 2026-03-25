"""
https://github.com/xingyizhou/CenterTrack
Modified by Rufeng Zhang
"""
import os
import numpy as np
import json
import cv2


DATA_PATH = '/cta/users/grad4/master/datasets/mot17_copy'
OUT_PATH = os.path.join(DATA_PATH, 'annotations')

# ─── Hangi işlemlerin çalışacağını kontrol eden hyperparametreler ───
CREATE_HALF_SPLITS = False      # train_half / val_half oluşturulsun mu?
CREATE_FULL_SPLITS = False      # train / test (full) oluşturulsun mu?
CREATE_CUSTOM_SPLITS = True     # Özel video bazlı train/val ayrımı yapılsın mı?

# Validation olarak ayrılacak video adları (seq klasör adlarıyla eşleşmeli)
# Örnek: ['MOT17-02-FRCNN', 'MOT17-04-FRCNN']
CUSTOM_VAL_VIDEOS = [
    'MOT17-02-DPM',
    'MOT17-02-FRCNN',
    'MOT17-02-SDP'
]

CUSTOM_TRAIN_SPLIT_NAME = 'train_MOT17_02_excluded'   # Çıktı JSON dosyasının adı
CUSTOM_VAL_SPLIT_NAME   = 'val_MOT17_02'    # Çıktı JSON dosyasının adı

CREATE_SPLITTED_ANN = True
CREATE_SPLITTED_DET = True


def process_split(split_name, data_path, seqs, out_path,
                  image_range_fn=None,
                  create_ann=False, create_det=False):
    """
    split_name      : JSON dosyasına yazılacak split adı (sadece loglama için)
    data_path       : Sekans klasörlerinin bulunduğu üst dizin
    seqs            : İşlenecek sekans (video) adları listesi
    out_path        : Çıktı JSON dosyasının tam yolu
    image_range_fn  : (num_images) -> [start, end] döndüren fonksiyon (None=tümü)
    create_ann/det  : Splitted gt/det dosyaları oluşturulsun mu?
    """
    out = {
        'images': [], 'annotations': [], 'videos': [],
        'categories': [{'id': 1, 'name': 'pedestrian'}]
    }
    image_cnt = 0
    ann_cnt   = 0
    video_cnt = 0

    for seq in sorted(seqs):
        if '.DS_Store' in seq:
            continue

        video_cnt += 1
        out['videos'].append({'id': video_cnt, 'file_name': seq})

        seq_path = os.path.join(data_path, seq)
        img_path = os.path.join(seq_path, 'img1')
        ann_path = os.path.join(seq_path, 'gt/gt.txt')

        images     = os.listdir(img_path)
        num_images = len([im for im in images if 'jpg' in im])

        image_range = image_range_fn(num_images) if image_range_fn else [0, num_images - 1]

        # ── Görüntü kayıtları ──────────────────────────────────────────
        for i in range(num_images):
            if i < image_range[0] or i > image_range[1]:
                continue
            img = cv2.imread(os.path.join(data_path,
                             '{}/img1/{:06d}.jpg'.format(seq, i + 1)))
            height, width = img.shape[:2]
            image_info = {
                'file_name':     '{}/img1/{:06d}.jpg'.format(seq, i + 1),
                'id':            image_cnt + i + 1,
                'frame_id':      i + 1 - image_range[0],
                'prev_image_id': image_cnt + i if i > 0 else -1,
                'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                'video_id':      video_cnt,
                'height':        height,
                'width':         width,
            }
            out['images'].append(image_info)

        print('{}: {} images'.format(seq, num_images))

        # ── Annotasyon kayıtları ───────────────────────────────────────
        det_path = os.path.join(seq_path, 'det/det.txt')
        anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
        dets = np.loadtxt(det_path, dtype=np.float32, delimiter=',')

        if create_ann and image_range_fn:
            anns_out = np.array([anns[i] for i in range(anns.shape[0])
                                 if image_range[0] <= int(anns[i][0]) - 1 <= image_range[1]],
                                dtype=np.float32)
            anns_out[:, 0] -= image_range[0]
            gt_out = os.path.join(seq_path, 'gt/gt_{}.txt'.format(split_name))
            with open(gt_out, 'w') as fout:
                for o in anns_out:
                    fout.write('{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n'.format(
                        int(o[0]), int(o[1]), int(o[2]), int(o[3]),
                        int(o[4]), int(o[5]), int(o[6]), int(o[7]), o[8]))

        if create_det and image_range_fn:
            dets_out = np.array([dets[i] for i in range(dets.shape[0])
                                 if image_range[0] <= int(dets[i][0]) - 1 <= image_range[1]],
                                dtype=np.float32)
            dets_out[:, 0] -= image_range[0]
            det_out = os.path.join(seq_path, 'det/det_{}.txt'.format(split_name))
            with open(det_out, 'w') as dout:
                for o in dets_out:
                    dout.write('{:d},{:d},{:.1f},{:.1f},{:.1f},{:.1f},{:.6f}\n'.format(
                        int(o[0]), int(o[1]), float(o[2]), float(o[3]),
                        float(o[4]), float(o[5]), float(o[6])))

        print('{} ann images'.format(int(anns[:, 0].max())))

        for i in range(anns.shape[0]):
            frame_id = int(anns[i][0])
            if frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]:
                continue
            track_id = int(anns[i][1])
            ann_cnt += 1

            if not ('15' in DATA_PATH):
                if not (float(anns[i][8]) >= 0.25):
                    continue
                if not (int(anns[i][6]) == 1):
                    continue
                if int(anns[i][7]) in [3, 4, 5, 6, 9, 10, 11]:
                    continue
                if int(anns[i][7]) in [2, 7, 8, 12]:
                    category_id = -1
                else:
                    category_id = 1
            else:
                category_id = 1

            ann = {
                'id':          ann_cnt,
                'category_id': category_id,
                'image_id':    image_cnt + frame_id,
                'track_id':    track_id,
                'bbox':        anns[i][2:6].tolist(),
                'conf':        float(anns[i][6]),
                'iscrowd':     0,
                'area':        float(anns[i][4] * anns[i][5]),
            }
            out['annotations'].append(ann)

        image_cnt += num_images

    print('Loaded {} → {} images, {} annotations'.format(
        split_name, len(out['images']), len(out['annotations'])))
    json.dump(out, open(out_path, 'w'))


if __name__ == '__main__':
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    train_data_path = os.path.join(DATA_PATH, 'train')
    test_data_path  = os.path.join(DATA_PATH, 'test')

    # ── 1) Eski full train / test splitleri ───────────────────────────
    if CREATE_FULL_SPLITS:
        for split, data_path in [('train', train_data_path), ('test', test_data_path)]:
            seqs = [s for s in os.listdir(data_path) if '.DS_Store' not in s]
            if split != 'test':
                seqs = [s for s in seqs if 'FRCNN' in s]
            process_split(
                split_name=split,
                data_path=data_path,
                seqs=seqs,
                out_path=os.path.join(OUT_PATH, '{}.json'.format(split)),
                image_range_fn=None,
                create_ann=False,
                create_det=False,
            )

    # ── 2) Eski half splitleri ─────────────────────────────────────────
    if CREATE_HALF_SPLITS:
        for split in ['train_half', 'val_half']:
            seqs = [s for s in os.listdir(train_data_path)
                    if '.DS_Store' not in s and 'FRCNN' in s]
            range_fn = (lambda n: [0, n // 2]) if 'train' in split \
                       else (lambda n: [n // 2 + 1, n - 1])
            process_split(
                split_name=split,
                data_path=train_data_path,
                seqs=seqs,
                out_path=os.path.join(OUT_PATH, '{}.json'.format(split)),
                image_range_fn=range_fn,
                create_ann=CREATE_SPLITTED_ANN,
                create_det=CREATE_SPLITTED_DET,
            )

    # ── 3) Yeni özel video bazlı train/val ayrımı ─────────────────────
    if CREATE_CUSTOM_SPLITS:
        all_seqs = sorted([s for s in os.listdir(train_data_path)
                           if '.DS_Store' not in s and 'FRCNN' in s])

        val_seqs   = [s for s in all_seqs if s in CUSTOM_VAL_VIDEOS]
        train_seqs = [s for s in all_seqs if s not in CUSTOM_VAL_VIDEOS]

        print('\n=== Custom Split ===')
        print('Train seqs:', train_seqs)
        print('Val   seqs:', val_seqs)

        # Custom train — tüm kareler
        process_split(
            split_name=CUSTOM_TRAIN_SPLIT_NAME,
            data_path=train_data_path,
            seqs=train_seqs,
            out_path=os.path.join(OUT_PATH, '{}.json'.format(CUSTOM_TRAIN_SPLIT_NAME)),
            image_range_fn=None,
            create_ann=False,
            create_det=False,
        )

        # Custom val — tüm kareler
        process_split(
            split_name=CUSTOM_VAL_SPLIT_NAME,
            data_path=train_data_path,
            seqs=val_seqs,
            out_path=os.path.join(OUT_PATH, '{}.json'.format(CUSTOM_VAL_SPLIT_NAME)),
            image_range_fn=None,
            create_ann=False,
            create_det=False,
        )