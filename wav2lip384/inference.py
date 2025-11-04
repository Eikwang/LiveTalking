from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch
from models import Wav2Lip
import platform
from face_detection import FaceAlignment, LandmarksType
import dlib
import yaml
import argparse
import logging
from datetime import datetime
import copy
def setup_logging():
    os.makedirs('logs', exist_ok=True)
    log_filename = f'logs/wav2lip_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

config = load_config()

class Args:
    def __init__(self, config):
        self.face = config['face']
        self.audio = config['audio']
        self.outfile = config['outfile']
        self.checkpoint_path = config['checkpoint_path']
        self.static = config['static']
        self.fps = config['fps']
        self.pads = config['pads']
        self.face_det_batch_size = config['face_det_batch_size']
        self.wav2lip_batch_size = config['wav2lip_batch_size']
        self.resize_factor = config['resize_factor']
        self.crop = config['crop']
        self.box = config['box']
        self.nosmooth = config['nosmooth']
        self.img_size = config['img_size']
args = Args(config)

class LaplacianSmoother:
    def __init__(self, config):
        self.kernel_size = config.get('laplacian_kernel_size', 3)
        self.sigma = config.get('laplacian_sigma', 1.5)
        self.lambda_weight = config.get('laplacian_weight', 0.1)  # 调整平滑权重
        self.prev_frame = None
        self.temporal_weight = config.get('temporal_weight', 0.3)
        logger.info(f"初始化拉普拉斯平滑器: kernel_size={self.kernel_size}, sigma={self.sigma}, weight={self.lambda_weight}")

    def apply_laplacian_smoothing(self, frame):
        """
        对单帧应用拉普拉斯平滑
        """
        # 转换为浮点数进行处理
        frame_float = frame.astype(np.float32) / 255.0

        # 计算拉普拉斯算子
        laplacian = cv2.Laplacian(frame_float, cv2.CV_32F, ksize=self.kernel_size)

        # 应用高斯模糊以减少噪声
        smoothed = cv2.GaussianBlur(frame_float, (self.kernel_size, self.kernel_size), self.sigma)

        # 结合原始图像和平滑结果
        result = frame_float - self.lambda_weight * laplacian
        result = (1 - self.lambda_weight) * frame_float + self.lambda_weight * smoothed

        # 时间平滑
        if self.prev_frame is not None:
            result = (1 - self.temporal_weight) * result + self.temporal_weight * self.prev_frame

        self.prev_frame = result.copy()

        # 剪切到有效范围并转回uint8
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return result

    def smooth_sequence(self, frames):
        """
        对视频序列进行平滑处理
        """
        smoothed_frames = []
        for frame in frames:
            smoothed = self.apply_laplacian_smoothing(frame)
            smoothed_frames.append(smoothed)
        return smoothed_frames

    def reset(self):
        """
        重置平滑器状态
        """
        self.prev_frame = None

class FaceStabilizer:
    def __init__(self, config):
        self.temporal_window = config['temporal_window']
        self.alpha = config['alpha']
        self.margin = config['margin']
        self.prev_mask = None
        self.prev_size = None
        self.prev_boxes = []
        logger.info(f"初始化人脸稳定器: window={self.temporal_window}, alpha={self.alpha}, margin={self.margin}")

    def get_smoothened_boxes(self, boxes, T):
        logger.debug("开始平滑人脸边界框")
        weights = np.array([1 - abs(i - T // 2) / (T // 2) for i in range(T)])
        weights = weights / np.sum(weights)

        smoothed_boxes = boxes.copy()
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i: i + T]

            if len(window) < T:
                weights_subset = weights[:len(window)]
                weights_subset = weights_subset / np.sum(weights_subset)
            else:
                weights_subset = weights

            smoothed_boxes[i] = np.sum(window * weights_subset.reshape(-1, 1), axis=0)

        return smoothed_boxes

    def blend_frames(self, frame, predicted_frame, positions, predictor):
        predicted_frame_resized = cv2.resize(predicted_frame, (0, 0), fx=1, fy=1)
        height, width = predicted_frame_resized.shape[:2]
        face = dlib.rectangle(left=0, top=0, right=width, bottom=height)
        gray_img = cv2.cvtColor(predicted_frame_resized, cv2.COLOR_BGR2GRAY)

        try:
            landmarks = predictor(gray_img, face)
            face_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
            face_center = np.mean(face_points, axis=0)
            face_contour_points = face_points[0:17]

            shrunk_face_contour_points = [
                (int(x + (x - face_center[0]) * (
                            -self.margin / np.linalg.norm([x - face_center[0], y - face_center[1]]))),
                 int(y + (y - face_center[1]) * (
                             -self.margin / np.linalg.norm([x - face_center[0], y - face_center[1]]))))
                for x, y in face_contour_points
            ]

            mask = np.zeros_like(gray_img, dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(shrunk_face_contour_points)], 255)

            mask = cv2.GaussianBlur(mask, (31, 31), 15)
            mask = mask.astype(np.float32) / 255.0

            current_size = mask.shape
            if self.prev_mask is not None:
                if self.prev_mask.shape != current_size:
                    self.prev_mask = cv2.resize(self.prev_mask, (current_size[1], current_size[0]))
                mask = self.alpha * mask + (1 - self.alpha) * self.prev_mask

            x1, y1, x2, y2 = positions
            new_width, new_height = x2 - x1, y2 - y1
            resized_x1 = x1 + (new_width - width) // 2
            resized_y1 = y1 + (new_height - height) // 2
            resized_x2 = resized_x1 + width
            resized_y2 = resized_y1 + height

            frame_height, frame_width = frame.shape[:2]
            resized_x1 = max(0, resized_x1)
            resized_y1 = max(0, resized_y1)
            resized_x2 = min(frame_width, resized_x2)
            resized_y2 = min(frame_height, resized_y2)

            actual_width = resized_x2 - resized_x1
            actual_height = resized_y2 - resized_y1
            if actual_width != width or actual_height != height:
                predicted_frame_resized = cv2.resize(predicted_frame_resized, (actual_width, actual_height))
                mask = cv2.resize(mask, (actual_width, actual_height))

            for i in range(3):
                frame_part = frame[resized_y1:resized_y2, resized_x1:resized_x2, i].astype(float)
                pred_part = predicted_frame_resized[:, :, i].astype(float)

                mean_frame = np.mean(frame_part)
                mean_pred = np.mean(pred_part)
                std_frame = np.std(frame_part)
                std_pred = np.std(pred_part)

                if std_pred > 0:
                    pred_part = (pred_part - mean_pred) * (std_frame / std_pred) + mean_frame
                pred_part = np.clip(pred_part, 0, 255)

                frame_part = frame_part * (1 - mask) + pred_part * mask
                frame[resized_y1:resized_y2, resized_x1:resized_x2, i] = frame_part.astype(np.uint8)

            self.prev_mask = mask.copy()
            self.prev_size = current_size

        except Exception as e:
            logger.error(f"帧混合过程中出现错误: {str(e)}")
            return frame

        return frame

def face_detect(images):
    logger.info("开始人脸检测")
    detector = FaceAlignment(LandmarksType._2D, flip_input=False, device=f'cuda')
    batch_size = 2

    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1:
                logger.error("GPU内存不足，无法进行人脸检测")
                raise RuntimeError(
                    'Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            logger.warning(f'内存溢出错误恢复中; 新的batch size: {batch_size}')
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image)
            logger.error("未检测到人脸！请确保视频中所有帧都包含人脸。")
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not args.nosmooth:
        stabilizer = FaceStabilizer(config)
        boxes = stabilizer.get_smoothened_boxes(boxes, T=config['temporal_window'])
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    logger.info("人脸检测完成")
    del detector
    return results

def datagen(frames, mels):
    logger.info("开始数据生成")
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if args.box[0] == -1:
        if not args.static:
            logger.info("对所有帧进行人脸检测")
            face_det_results = face_detect(frames)
        else:
            logger.info("仅对第一帧进行人脸检测")
            face_det_results = face_detect([frames[0]])
    else:
        logger.info('使用指定的边界框替代人脸检测...')
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

    for i, m in enumerate(mels):
        idx = 0 if args.static else i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (args.img_size, args.img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, args.img_size // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, args.img_size // 2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info('使用 {} 进行推理'.format(device))

def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()


def main():
    # audio_smooth = True
    logger.info("开始处理视频合成任务")
    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')

    elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps

    else:
        logger.info(f"正在读取视频文件: {args.face}")
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        # 获取视频方向信息
        rotation = int(video_stream.get(cv2.CAP_PROP_ORIENTATION_META))
        logger.info(f"视频方向信息: {rotation}")

        logger.info('读取视频帧...')

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if args.resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1] // args.resize_factor, frame.shape[0] // args.resize_factor))

            # 根据视频实际方向自动处理
            if rotation == 180:
                frame = cv2.flip(frame, -1)  # 180度翻转
            elif rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif rotation == 270:
                frame = cv2.flip(frame,-1)

            y1, y2, x1, x2 = args.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]
            full_frames.append(frame)

    print("Number of frames available for inference: " + str(len(full_frames)))

    if not args.audio.endswith('.wav'):
        print('Extracting raw audio...')
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')
        subprocess.call(command, shell=True)
        args.audio = 'temp/temp.wav'

    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)
    # 可以添加一个缩放因子
    scaling_factor = 1.25 # 大于1会增大张口幅度
    mel = mel * scaling_factor
    print(mel.shape)
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80. / fps
    # gen_frame_num = int(len(mel[0]) / mel_idx_multiplier)
    i = 0
    T = 5
    while 1:
        start_idx = int((i - (T - 1) // 2) * mel_idx_multiplier)
        if start_idx < 0:
            start_idx = 0
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1

    print("Length of mel chunks: {}".format(len(mel_chunks)))

    full_frames = full_frames[:len(mel_chunks)]

    batch_size = args.wav2lip_batch_size
    gen = datagen(full_frames.copy(), mel_chunks)

    # 初始化平滑器
    stabilizer = FaceStabilizer(config)
    predictor = load_face_predictor(config['face_predictor_path'])
    laplacian_smoother = LaplacianSmoother(config)

    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,
                                                                total=int(np.ceil(float(len(mel_chunks)) / batch_size)))):
        if i == 0:
            model = load_model(args.checkpoint_path)
            print("Model loaded")

            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter('temp/result.avi',
                                cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            # 使用稳定器和拉普拉斯平滑
            f = stabilizer.blend_frames(f, p, (x1, y1, x2, y2), predictor)
            f = laplacian_smoother.apply_laplacian_smoothing(f)
            out.write(f)

    out.release()

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, 'temp/result.avi', args.outfile)
    subprocess.call(command, shell=platform.system() != 'Windows')

def load_face_predictor(path):
    return dlib.shape_predictor(path)

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='Inference code to lip-sync videos using Wav2Lip models')
        parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
        args_cmd = parser.parse_args()

        # parser.add_argument('--audio_smooth', default=True, action='store_true', help='smoothing audio embedding')
        # 加载配置文件
        config = load_config(args_cmd.config)

        # 创建必要的目录
        os.makedirs('temp', exist_ok=True)
        os.makedirs(os.path.dirname(config['outfile']), exist_ok=True)

        logger.info("开始执行主程序")
        main()
        logger.info("程序执行完成")
    except Exception as e:
        logger.error(f"程序执行过程中出现错误: {str(e)}", exc_info=True)
        raise