import os
import numpy as np
import cv2
import json
import time
import threading
import queue
from loguru import logger
from torchvision import transforms
from tqdm import tqdm
import torch
from scipy.interpolate import interp1d
import shutil
import subprocess
import mediapipe as mp

def geneHeadInfo(sampleRate, bits, sampleNum):
    import struct
    rHeadInfo = b'\x52\x49\x46\x46'
    fileLength = struct.pack('i', sampleNum + 36)
    rHeadInfo += fileLength
    rHeadInfo += b'\x57\x41\x56\x45\x66\x6D\x74\x20\x10\x00\x00\x00\x01\x00\x01\x00'
    rHeadInfo += struct.pack('i', sampleRate)
    rHeadInfo += struct.pack('i', int(sampleRate * bits / 8))
    rHeadInfo += b'\x02\x00'
    rHeadInfo += struct.pack('H', bits)
    rHeadInfo += b'\x64\x61\x74\x61'
    rHeadInfo += struct.pack('i', sampleNum)
    return rHeadInfo

class liteAvatar(object):
    def __init__(self,
                 data_dir=None,
                 language='ZH',
                 a2m_path=None,
                 num_threads=1,
                 use_bg_as_idle=False,
                 fps=30,
                 generate_offline=True,
                 use_gpu=False):
        logger.info('liteAvatar init start...')
        self.data_dir = data_dir
        self.fps = fps
        self.use_bg_as_idle = use_bg_as_idle
        self.use_gpu = use_gpu
        self.device = "cuda" if use_gpu else "cpu"
        s = time.time()
        from audio2mouth_cpu import Audio2Mouth
        self.audio2mouth = Audio2Mouth(use_gpu)
        logger.info(f'audio2mouth init over in {time.time() - s}s')
        self.p_list = [str(ii) for ii in range(32)]
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.load_data_thread = None
        logger.info('liteAvatar init over')
        self._generate_offline = generate_offline
        if generate_offline:
            self.load_dynamic_model(data_dir)
            self.threads_prep = []
            barrier_prep = threading.Barrier(num_threads, action=None, timeout=None)
            for i in range(num_threads):
                t = threading.Thread(target=self.face_gen_loop, args=(i, barrier_prep, self.input_queue, self.output_queue))
                self.threads_prep.append(t)
            for t in self.threads_prep:
                t.daemon = True
                t.start()

    def stop_algo(self):
        pass

    def load_dynamic_model(self, data_dir):
        logger.info("start to load dynamic data")
        start_time = time.time()
        self.encoder = torch.jit.load(f'{data_dir}/net_encode.pt').to(self.device)
        self.generator = torch.jit.load(f'{data_dir}/net_decode.pt').to(self.device)
        self.load_data_sync(data_dir=data_dir, bg_frame_cnt=150)
        self.load_data(data_dir=data_dir, bg_frame_cnt=150)
        self.ref_data_list = [0 for x in range(150)]
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        logger.info("load dynamic model in {:.3f}s", time.time() - start_time)

    def unload_dynamic_model(self):
        pass

    def enhance_image(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return enhanced

    def load_data_sync(self, data_dir, bg_frame_cnt=None):
        t = time.time()
        self.neutral_pose = np.load(f'{data_dir}/neutral_pose.npy')
        self.mouth_scale = None
        self.bg_data_list = []
        bg_video = cv2.VideoCapture(f'{data_dir}/bg_video.mp4')
        while True:
            ret, img = bg_video.read()
            self.bg_data_list.append(img)
            if ret is False:
                break
        self.bg_video_frame_count = len(self.bg_data_list) if bg_frame_cnt is None else min(bg_frame_cnt, len(self.bg_data_list))
        user_image_path = os.path.join(data_dir, 'image.png')
        if os.path.exists(user_image_path):
            image = cv2.imread(user_image_path)[:, :, :3]
            max_dim = 1024  # Resize to reduce memory usage
            if image.shape[0] > max_dim or image.shape[1] > max_dim:
                scale = max_dim / max(image.shape[:2])
                new_dims = (int(image.shape[1] * scale), int(image.shape[0] * scale))
                image = cv2.resize(image, new_dims, interpolation=cv2.INTER_AREA)
                logger.info(f"Resized image to {new_dims}")
            image = self.enhance_image(image)
            self.original_height, self.original_width = image.shape[:2]
            logger.info(f"Original image dimensions: height={self.original_height}, width={self.original_width}")
            try:
                y1, y2, x1, x2 = self.detect_mouth_box(image)
                self.y1, self.y2, self.x1, self.x2 = y1, y2, x1, x2
                logger.info(f"Auto-detected face box: y1={y1}, y2={y2}, x1={x1}, x2={x2}")
            except Exception as e:
                logger.error(f"Failed to detect mouth box: {e}")
                raise
        else:
            logger.warning("No image.png found in data_dir, assuming original dimensions as 1024x1024")
            self.original_height, self.original_width = 1024, 1024
            self.y1, self.y2, self.x1, self.x2 = 400, 600, 400, 600
        logger.info(f"Face box dimensions: y1={self.y1}, y2={self.y2}, x1={self.x1}, x2={self.x2}, height={self.y2-self.y1}, width={self.x2-self.x1}")
        self.merge_mask = self.create_elliptical_mask(self.y2 - self.y1, self.x2 - self.x1)
        logger.info(f"merge_mask shape: {self.merge_mask.shape}")
        self.frame_vid_list = []
        self.image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        logger.info("load data sync in {:.3f}s", time.time() - t)

    def create_elliptical_mask(self, height, width):
        mask = np.zeros((height, width, 1), dtype=np.float32)  # Single channel
        center = (width // 2, height // 2)
        axes = (int(width * 0.5), int(height * 0.4))  # Wider ellipse
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)  # Solid fill
        mask = cv2.GaussianBlur(mask, (25, 25), 0)  # Less blur = sharper cut
        mask = np.clip(mask * 1.5, 0, 1)  # Boost mask strength
        return mask

    def detect_mouth_box(self, image):
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                h, w = image.shape[:2]
                mouth_landmarks = [landmarks[i] for i in range(61, 88)]
                xs = [int(p.x * w) for p in mouth_landmarks]
                ys = [int(p.y * h) for p in mouth_landmarks]
                x1, x2 = max(0, min(xs) - 10), min(w, max(xs) + 10)
                y1, y2 = max(0, min(ys) - 5), min(h, max(ys) + 5)
                debug_img = image.copy()
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imwrite(os.path.join(self.data_dir, 'debug_mouth_box.jpg'), debug_img)
                return y1, y2, x1, x2
            else:
                raise ValueError("No face landmarks detected in image.")

    def load_data(self, data_dir, bg_frame_cnt=None):
        logger.info(f'loading data from {data_dir}')
        s = time.time()
        self.ref_img_list = []
        user_image_path = os.path.join(data_dir, 'image.png')
        if os.path.exists(user_image_path):
            logger.info(f'Using user-uploaded image: {user_image_path}')
            image = cv2.cvtColor(cv2.imread(user_image_path)[:, :, :3], cv2.COLOR_BGR2RGB)
            ref_img = self.image_transforms(np.uint8(image))
            encoder_input = ref_img.unsqueeze(0).float().to(self.device)
            x = self.encoder(encoder_input)
            logger.info(f"Encoder output: {type(x)}, content: {x}")
            if isinstance(x, torch.Tensor):
                tensor_list = [x.clone() for _ in range(4)]
                self.ref_img_list = [tensor_list for _ in range(bg_frame_cnt or 150)]
            elif isinstance(x, list) and len(x) == 4 and all(isinstance(t, torch.Tensor) for t in x):
                self.ref_img_list = [[t.clone() for t in x] for _ in range(bg_frame_cnt or 150)]
            else:
                logger.error(f"Unexpected encoder output: {type(x)}, content: {x}")
                raise ValueError("Encoder output must be a tensor or a list of 4 tensors")
        else:
            logger.warning(f'No user image found at {user_image_path}, using ref_frames')
            for ii in tqdm(range(bg_frame_cnt or 150)):
                img_file_path = os.path.join(data_dir, 'ref_frames', f'ref_{ii:05d}.jpg')
                if not os.path.exists(img_file_path):
                    logger.error(f'Reference frame not found: {img_file_path}')
                    continue
                image = cv2.cvtColor(cv2.imread(img_file_path)[:, :, :3], cv2.COLOR_BGR2RGB)
                ref_img = self.image_transforms(np.uint8(image))
                encoder_input = ref_img.unsqueeze(0).float().to(self.device)
                x = self.encoder(encoder_input)
                if isinstance(x, torch.Tensor):
                    tensor_list = [x.clone() for _ in range(4)]
                    self.ref_img_list.append(tensor_list)
                elif isinstance(x, list) and len(x) == 4 and all(isinstance(t, torch.Tensor) for t in x):
                    self.ref_img_list.append([t.clone() for t in x])
                else:
                    logger.error(f"Ref frame encoding failed at {ii}: {type(x)}")
        logger.info(f'load data over in {time.time() - s}s')

    def face_gen_loop(self, thread_id, barrier, in_queue, out_queue):
        logger.info(f"Starting face_gen_loop for thread {thread_id}")
        while True:
            try:
                data = in_queue.get(timeout=30)
                logger.info(f"Thread {thread_id} received data from input_queue")
            except queue.Empty:
                logger.warning(f"Thread {thread_id} input_queue timeout after 30s")
                break
            if data is None:
                logger.info(f"Thread {thread_id} received None, shutting down")
                in_queue.put(None)
                break
            s = time.time()
            param_res = data[0]
            bg_frame_id = data[1]
            global_frame_id = data[2]
            logger.info(f"Thread {thread_id} processing frame {global_frame_id}")
            with torch.no_grad():
                mouth_img = self.param2img(param_res, bg_frame_id)
                full_img, mouth_img = self.merge_mouth_to_bg(mouth_img, bg_frame_id, use_photo=True)
            logger.info(f"Thread {thread_id} processed global_frame_id: {global_frame_id} in {round(time.time() - s, 3)}s")
            out_queue.put((global_frame_id, full_img, mouth_img))
        barrier.wait()
        if thread_id == 0:
            logger.info("Thread 0 putting None to output_queue to signal completion")
            out_queue.put(None)

    def param2img(self, param_res, bg_frame_id, global_frame_id=0, is_idle=False):
        param_val = []
        for key in param_res:
            val = param_res[key]
            param_val.append(val)
        param_val = np.asarray(param_val)
        input_list = self.ref_img_list[bg_frame_id]
        logger.info(f"Generator input list length: {len(input_list)}, types: {[type(t) for t in input_list]}, shapes: {[t.shape for t in input_list]}")
        source_img = self.generator(input_list, torch.from_numpy(param_val).unsqueeze(0).float().to(self.device))
        source_img = source_img.detach().to("cpu")
        mouth_img_np = (source_img / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).numpy() * 255
        mouth_img_np = mouth_img_np.astype(np.uint8)
        cv2.imwrite(f'debug_mouth_{bg_frame_id}.jpg', mouth_img_np)
        return source_img

    def get_idle_param(self):
        bg_param = self.neutral_pose
        tmp_json = {}
        for ii in range(len(self.p_list)):
            tmp_json[str(ii)] = float(bg_param[ii])
        return tmp_json

    def merge_mouth_to_bg(self, mouth_image, bg_frame_id, use_photo=False):
        # Process mouth image (convert to BGR & resize)
        mouth_image = (mouth_image / 2 + 0.5).clamp(0, 1)
        mouth_image = mouth_image[0].permute(1, 2, 0) * 255
        mouth_image = mouth_image.numpy().astype(np.uint8)
        mouth_image = cv2.cvtColor(mouth_image, cv2.COLOR_RGB2BGR)
        
        # Resize to match face box
        target_height = self.y2 - self.y1
        target_width = self.x2 - self.x1
        mouth_image = cv2.resize(mouth_image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

        # Load background
        if use_photo and os.path.exists(os.path.join(self.data_dir, 'image.png')):
            full_img = cv2.imread(os.path.join(self.data_dir, 'image.png'))
        else:
            full_img = self.bg_data_list[bg_frame_id].copy()

        # Resize and expand mask to 3 channels
        merge_mask_resized = cv2.resize(self.merge_mask, (target_width, target_height))
        merge_mask_resized = np.repeat(merge_mask_resized, 3, axis=2)
        merge_mask_resized = (merge_mask_resized * 255).astype(np.uint8)

        # Apply blending
        y1, y2, x1, x2 = self.y1, self.y2, self.x1, self.x2
        full_img[y1:y2, x1:x2] = (
            full_img[y1:y2, x1:x2].astype(float) * (1 - merge_mask_resized / 255.0) +
            mouth_image.astype(float) * (merge_mask_resized / 255.0)
        ).astype(np.uint8)

        # Save debug images
        cv2.imwrite(f'debug_frame_{bg_frame_id}.jpg', full_img)
        cv2.imwrite(f'debug_mouth_region_{bg_frame_id}.jpg', mouth_image)
        return full_img, mouth_image

    def interp_param(self, param_res, fps=25):
        old_len = len(param_res)
        new_len = int(old_len / 30 * fps + 0.5)
        interp_list = {}
        for key in param_res[0]:
            tmp_list = []
            for ii in range(len(param_res)):
                tmp_list.append(param_res[ii][key])
            tmp_list = np.asarray(tmp_list)
            x = np.linspace(0, old_len - 1, num=old_len, endpoint=True)
            newx = np.linspace(0, old_len - 1, num=new_len, endpoint=True)
            f = interp1d(x, tmp_list)
            y = f(newx)
            interp_list[key] = y
        new_param_res = []
        for ii in range(new_len):
            tmp_json = {}
            for key in interp_list:
                tmp_json[key] = interp_list[key][ii]
            new_param_res.append(tmp_json)
        return new_param_res

    def padding_last(self, param_res, last_end=None):
        bg_param = self.neutral_pose
        if last_end is None:
            last_end = len(param_res)
        padding_cnt = 5
        final_end = max(last_end + 5, len(param_res))
        param_res = param_res[:last_end]
        padding_list = []
        for ii in range(last_end, final_end):
            tmp_json = {}
            for key in param_res[-1]:
                kk = ii - last_end
                scale = max((padding_cnt - kk - 1) / padding_cnt, 0.0)
                end_value = bg_param[int(key)]
                tmp_json[key] = (param_res[-1][key] - end_value) * scale + end_value
            padding_list.append(tmp_json)
        print('padding_cnt:', len(padding_list))
        param_res = param_res + padding_list
        return param_res

    def audio2param(self, audio_file_path, prefix_padding_size=0, is_complete=False, audio_status=-1):
        logger.warning("Skipping audio2param and using hardcoded parameters for testing.")
        param_res = [
            {str(i): 0.0 for i in range(32)}
            for _ in range(150)
        ]
        return param_res

    def handle(self, audio_file_path, result_dir, param_res=None):
        logger.info("Starting handle")
        s = time.time()
        if param_res is None:
            logger.info("No param_res provided, generating from hardcoded parameters")
            param_res = self.audio2param(audio_file_path)
        logger.info(f"Using {len(param_res)} parameters")
        for ii in range(len(param_res)):
            frame_id = ii
            if int(frame_id / self.bg_video_frame_count) % 2 == 0:
                frame_id = frame_id % self.bg_video_frame_count
            else:
                frame_id = self.bg_video_frame_count - 1 - frame_id % self.bg_video_frame_count
            self.input_queue.put((param_res[ii], frame_id, ii))
            if ii % 10 == 0:
                logger.info(f"Pushed {ii+1}/{len(param_res)} items to input_queue")
        self.input_queue.put(None)
        logger.info("Pushed None to input_queue to signal end")
        tmp_frame_dir = os.path.join(result_dir, 'tmp_frames')
        if os.path.exists(tmp_frame_dir):
            os.system(f'rm -rf {tmp_frame_dir}')
        os.mkdir(tmp_frame_dir)
        logger.info(f"Created tmp_frame_dir: {tmp_frame_dir}")
        frame_count = 0
        while True:
            try:
                res_data = self.output_queue.get(timeout=30)
                if res_data is None:
                    logger.info("Received None from output_queue, breaking loop")
                    break
                global_frame_index = res_data[0]
                target_path = f'{tmp_frame_dir}/{str(global_frame_index+1).zfill(5)}.jpg'
                cv2.imwrite(target_path, res_data[1])
                frame_count += 1
                if frame_count % 10 == 0:
                    logger.info(f"Wrote {frame_count} frames to {tmp_frame_dir}")
            except queue.Empty:
                logger.error("output_queue.get() timed out after 30s, breaking loop")
                break
        logger.info(f"Finished writing {frame_count} frames")
        for p in self.threads_prep:
            p.join()
        logger.info("All threads joined")
        logger.info(f"Result dir: {result_dir}, exists: {os.path.exists(result_dir)}")
        logger.info(f"tmp_frames dir: {tmp_frame_dir}, exists: {os.path.exists(tmp_frame_dir)}")
        tmp_frames_files = os.listdir(tmp_frame_dir) if os.path.exists(tmp_frame_dir) else []
        logger.info(f"tmp_frames contents: {', '.join(tmp_frames_files)}")
        ffmpeg_path = shutil.which('ffmpeg')
        logger.info(f"ffmpeg path: {ffmpeg_path}")
        if ffmpeg_path is None:
            logger.error("ffmpeg not found in PATH")
            raise FileNotFoundError("ffmpeg not found in PATH")
        output_video = os.path.join(result_dir, 'test_demo.mp4')
        cmd = f'"{ffmpeg_path}" -r 30 -i "{tmp_frame_dir}/%05d.jpg" -framerate 30 -c:v libx264 -preset veryslow -crf 5 -pix_fmt yuv420p -b:v 30000k -loglevel error "{output_video}" -y'
        logger.info(f"Running ffmpeg command: {cmd}")
        try:
            subprocess.run(cmd, shell=True, check=True)
            logger.info(f"Generated video: {output_video}, exists: {os.path.exists(output_video)}")
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg failed: {e}")
            raise
        logger.info(f"handle completed in {time.time() - s:.3f}s")

    @staticmethod
    def read_wav_to_bytes(file_path):
        import wave
        try:
            with wave.open(file_path, 'rb') as wav_file:
                params = wav_file.getparams()
                print(f"Channels: {params.nchannels}, Sample Width: {params.sampwidth}, Frame Rate: {params.framerate}, Number of Frames: {params.nframes}")
                frames = wav_file.readframes(params.nframes)
                return frames
        except wave.Error as e:
            print(f"Error reading WAV file: {e}")
            return None

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--audio_file', type=str)
    parser.add_argument('--result_dir', type=str)
    parser.add_argument('--param_file', type=str, help="Path to precomputed param_res JSON file", default=None)
    args = parser.parse_args()
    audio_file = args.audio_file
    tmp_frame_dir = args.result_dir
    lite_avatar = liteAvatar(data_dir=args.data_dir, num_threads=2, generate_offline=True, use_gpu=False)
    lite_avatar.handle(audio_file, tmp_frame_dir)