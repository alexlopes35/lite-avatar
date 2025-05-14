import os
import numpy as np
import cv2
import json
import time
import librosa
import threading
import queue
from loguru import logger
import base64
import soundfile as sf
from io import BytesIO
from pydub import AudioSegment
from pydub.silence import detect_silence
from torchvision import transforms
from tqdm import tqdm
import torch
from scipy.interpolate import interp1d
import wave
import shutil
import subprocess

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
                    use_gpu=True):
           
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
           self.load_data_thread: threading.Thread = None

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
           
           y1,y2,x1,x2 = open(f'{data_dir}/face_box.txt', 'r').readlines()[0].split()
           self.y1,self.y2,self.x1,self.x2 = int(y1),int(y2),int(x1),int(x2)
           
           self.merge_mask = (np.ones((self.y2-self.y1,self.x2-self.x1,3)) * 255).astype(np.uint8)
           self.merge_mask[20:-20,20:-20,:] *= 0
           self.merge_mask = cv2.GaussianBlur(self.merge_mask, (31,31), 20)
           self.merge_mask = self.merge_mask / 255
           
           self.frame_vid_list = []
           
           self.image_transforms = transforms.Compose(
           [   
               transforms.ToTensor(),
               transforms.Normalize([0.5], [0.5]),
           ])
           logger.info("load data sync in {:.3f}s", time.time() - t)
       
       def load_data(self, data_dir, bg_frame_cnt=None):
           logger.info(f'loading data from {data_dir}')
           s = time.time()

           self.ref_img_list = []
           user_image_path = os.path.join(data_dir, 'image.png')
           if os.path.exists(user_image_path):
               logger.info(f'Using user-uploaded image: {user_image_path}')
               image = cv2.cvtColor(cv2.imread(user_image_path)[:,:,0:3], cv2.COLOR_BGR2RGB)
               image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LANCZOS4)
               ref_img = self.image_transforms(np.uint8(image))
               encoder_input = ref_img.unsqueeze(0).float().to(self.device)
               x = self.encoder(encoder_input)
               # Ensure x is a tensor before cloning
               if isinstance(x, torch.Tensor):
                   self.ref_img_list = [x.clone() for _ in range(bg_frame_cnt or 150)]
               else:
                   logger.error(f"Encoder output x is not a tensor: {type(x)}")
           else:
               logger.warning(f'No user image found at {user_image_path}, using ref_frames')
               for ii in tqdm(range(bg_frame_cnt or 150)):
                   img_file_path = os.path.join(data_dir, 'ref_frames', f'ref_{ii:05d}.jpg')
                   if not os.path.exists(img_file_path):
                       logger.error(f'Reference frame not found: {img_file_path}')
                       continue
                   image = cv2.cvtColor(cv2.imread(img_file_path)[:,:,0:3], cv2.COLOR_BGR2RGB)
                   image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LANCZOS4)
                   ref_img = self.image_transforms(np.uint8(image))
                   encoder_input = ref_img.unsqueeze(0).float().to(self.device)
                   x = self.encoder(encoder_input)
                   self.ref_img_list.append(x)
           
           logger.info(f'load data over in {time.time() - s}s')
       
       def face_gen_loop(self, thread_id, barrier, in_queue, out_queue):
           while True:
               try:
                   data = in_queue.get()
               except queue.Empty:
                   break
               
               if data is None:
                   in_queue.put(None)
                   break
               
               s = time.time()
               
               param_res = data[0]
               bg_frame_id = data[1]
               global_frame_id = data[2]
               
               mouth_img = self.param2img(param_res, bg_frame_id)
               full_img, mouth_img = self.merge_mouth_to_bg(mouth_img, bg_frame_id, use_photo=True)
               
               logger.info('global_frame_id: {} in {}s'.format(global_frame_id, round(time.time() - s, 3)))
               
               out_queue.put((global_frame_id, full_img, mouth_img))
           
           barrier.wait()
           if thread_id == 0:
               out_queue.put(None)
               
       def param2img(self, param_res, bg_frame_id, global_frame_id=0, is_idle=False):
           param_val = []
           for key in param_res:
               val = param_res[key]
               param_val.append(val)
           param_val = np.asarray(param_val)
           
           source_img = self.generator(self.ref_img_list[bg_frame_id], torch.from_numpy(param_val).unsqueeze(0).float().to(self.device))
           source_img = source_img.detach().to("cpu")
           
           return source_img
       
       def get_idle_param(self):
           bg_param = self.neutral_pose
           tmp_json = {}
           for ii in range(len(self.p_list)):
               tmp_json[str(ii)] = float(bg_param[ii])
           return tmp_json
       
       def merge_mouth_to_bg(self, mouth_image, bg_frame_id, use_photo=False):
           mouth_image = (mouth_image / 2 + 0.5).clamp(0, 1)
           mouth_image = mouth_image[0].permute(1,2,0)*255
           
           mouth_image = mouth_image.numpy().astype(np.uint8)
           mouth_image = cv2.resize(mouth_image, (self.x2-self.x1, self.y2-self.y1), interpolation=cv2.INTER_LANCZOS4)
           mouth_image = mouth_image[:,:,::-1]
           
           if use_photo:
               user_image_path = os.path.join(self.data_dir, 'image.png')
               if os.path.exists(user_image_path):
                   full_img = cv2.cvtColor(cv2.imread(user_image_path)[:,:,0:3], cv2.COLOR_BGR2RGB)
                   full_img = cv2.resize(full_img, (512, 512), interpolation=cv2.INTER_LANCZOS4)
                   full_img[self.y1:self.y2, self.x1:self.x2, :] = mouth_image * (1 - self.merge_mask) + full_img[self.y1:self.y2, self.x1:self.x2, :] * self.merge_mask
               else:
                   full_img = self.bg_data_list[bg_frame_id].copy()
           else:
               full_img = self.bg_data_list[bg_frame_id].copy()
           
           if not use_photo:
               full_img[self.y1:self.y2,self.x1:self.x2,:] = mouth_image * (1 - self.merge_mask) + full_img[self.y1:self.y2,self.x1:self.x2,:] * self.merge_mask
           full_img = full_img.astype(np.uint8)
           return full_img, mouth_image.astype(np.uint8)
       
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
           input_audio, sr = sf.read(audio_file_path)
           if sr != 22050:
               logger.warning(f"Input audio sample rate {sr} Hz, expected 22050 Hz, resampling")
               input_audio = librosa.resample(input_audio, orig_sr=sr, target_sr=22050)
               sr = 22050
           
           input_audio = input_audio / np.max(np.abs(input_audio))
           input_audio = librosa.effects.preemphasis(input_audio, coef=0.97)  # Enhance audio
           
           param_res, _, _ = self.audio2mouth.inference(subtitles=None, input_audio=input_audio)
           
           sil_scale = np.zeros(len(param_res))
           sound = AudioSegment.from_file(audio_file_path, format="wav")
           start_end_list = detect_silence(sound, min_silence_len=200, silence_thresh=-35)
           if len(start_end_list) > 0:
               for start, end in start_end_list:
                   start_frame = int(start / 1000 * 30)
                   end_frame = int(end / 1000 * 30)
                   logger.info(f'silence part: {start_frame}-{end_frame} frames')
                   sil_scale[start_frame:end_frame] = 1
           sil_scale = np.pad(sil_scale, 2, mode='edge')
           kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
           sil_scale = np.convolve(sil_scale, kernel, 'same')
           sil_scale = sil_scale[2:-2]
           self.make_silence(param_res, sil_scale)
           if self.fps != 30:
               param_res = self.interp_param(param_res, fps=self.fps)
           
           if is_complete:
               param_res = self.padding_last(param_res)
               
           return param_res
       
       def make_silence(self, param_res, sil_scale):
           bg_param = self.neutral_pose
           
           for ii in range(len(param_res)):
               for key in param_res[ii]:
                   neu_value = bg_param[int(key)]
                   param_res[ii][key] = param_res[ii][key] * (1 - sil_scale[ii]) + neu_value * sil_scale[ii]
           return param_res
       
       def handle(self, audio_file_path, result_dir, param_res=None):
           if param_res is None:
               param_res = self.audio2param(audio_file_path)
           
           for ii in range(len(param_res)):
               s = time.time()
               frame_id = ii
               if int(frame_id / self.bg_video_frame_count) % 2 == 0:
                   frame_id = frame_id % self.bg_video_frame_count
               else:
                   frame_id = self.bg_video_frame_count - 1 - frame_id % self.bg_video_frame_count
               self.input_queue.put((param_res[ii], frame_id, ii))
           
           self.input_queue.put(None)
           
           tmp_frame_dir = os.path.join(result_dir, 'tmp_frames')
           if os.path.exists(tmp_frame_dir):
               os.system(f'rm -rf {tmp_frame_dir}')
           os.mkdir(tmp_frame_dir)
           
           while True:
               res_data = self.output_queue.get()
               if res_data is None:
                   break
               global_frame_index = res_data[0]
               target_path = f'{tmp_frame_dir}/{str(global_frame_index+1).zfill(5)}.jpg'
               cv2.imwrite(target_path, res_data[1])
           
           for p in self.threads_prep:
               p.join()
           
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
           cmd = f'"{ffmpeg_path}" -r 30 -i "{tmp_frame_dir}/%05d.jpg" -i "{audio_file_path}" -framerate 30 -c:v libx264 -preset veryslow -crf 16 -pix_fmt yuv420p -b:v 12000k -c:a aac -b:a 256k -strict experimental -loglevel error "{output_video}" -y'
           logger.info(f"Running ffmpeg command: {cmd}")
           try:
               subprocess.run(cmd, shell=True, check=True)
               logger.info(f"Generated video: {output_video}, exists: {os.path.exists(output_video)}")
           except subprocess.CalledProcessError as e:
               logger.error(f"ffmpeg failed: {e}")
               raise
       
       @staticmethod
       def read_wav_to_bytes(file_path):
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
       args = parser.parse_args()
       
       audio_file = args.audio_file
       tmp_frame_dir = args.result_dir
       
       lite_avatar = liteAvatar(data_dir=args.data_dir, num_threads=1, generate_offline=True)
       
       lite_avatar.handle(audio_file, tmp_frame_dir)