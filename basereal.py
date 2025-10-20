import math
import torch
import numpy as np

import subprocess
import os
import time
import cv2
import glob
import resampy

import queue
from queue import Queue
from threading import Thread, Event
from io import BytesIO
import soundfile as sf

import asyncio
from av import AudioFrame, VideoFrame

import av
from fractions import Fraction

from ttsreal import EdgeTTS,SovitsTTS,XTTS,CosyVoiceTTS,FishTTS,TencentTTS
from logger import logger

from tqdm import tqdm
def read_imgs(img_list):
    frames = []
    logger.info('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def play_audio(quit_event, queue):
    import pyaudio
    from logger import logger 
    
    p = pyaudio.PyAudio()
    
    virtual_device_index = None
    logger.info("检测可用音频设备:")
    for i in range(p.get_device_count()):
        device = p.get_device_info_by_index(i)
        device_name = device['name'].lower()
        logger.info(f"设备 {i}: {device_name}, 最大输出通道: {device['maxOutputChannels']}")
        
        if 'virtual' in device_name or 'virtual audio' in device_name:
            virtual_device_index = i
            logger.info(f"找到虚拟声卡: 设备ID={i}, 名称={device_name}")
    
    if virtual_device_index is None:
        try:
            virtual_device_index = p.get_default_output_device_info()['index']
            device_info = p.get_device_info_by_index(virtual_device_index)
            logger.info(f"使用默认输出设备: 设备ID={virtual_device_index}, 名称={device_info['name']}")
        except Exception as e:
            logger.error(f"获取默认设备失败: {e}，使用设备ID=0")
            virtual_device_index = 0
    
    try:
        device_info = p.get_device_info_by_index(virtual_device_index)
        max_channels = device_info['maxOutputChannels']
        channels = 1  # 修改为单声道
        logger.info(f"使用通道数: {channels}, 设备最大支持: {max_channels}")
    except Exception as e:
        logger.error(f"获取设备通道数失败: {e}，使用默认通道数1")
        channels = 1
    
    try:
        stream = p.open(
            rate=16000,
            channels=channels,
            format=pyaudio.paInt16,  # 使用16位整数格式
            output=True,
            output_device_index=virtual_device_index
        )
        stream.start_stream()
        logger.info(f"音频流已启动，通道数: {channels}, 设备: {device_info['name']}")
        
        while not quit_event.is_set():
            stream.write(queue.get(block=True))
        
        stream.stop_stream()
        stream.close()
    except OSError as e:
        logger.error(f"音频流打开失败: {e}")
        logger.error(f"设备 {virtual_device_index} 通道数: {channels}, 最大支持: {max_channels}")
    finally:
        p.terminate()
        
class BaseReal:
    def __init__(self, opt):
        self.opt = opt
        self.sample_rate = 16000
        self.chunk = self.sample_rate // opt.fps # 320 samples per chunk (20ms * 16000 / 1000)
        self.sessionid = self.opt.sessionid

        if opt.tts == "edgetts":
            self.tts = EdgeTTS(opt,self)
        elif opt.tts == "gpt-sovits":
            self.tts = SovitsTTS(opt,self)
        elif opt.tts == "xtts":
            self.tts = XTTS(opt,self)
        elif opt.tts == "cosyvoice":
            self.tts = CosyVoiceTTS(opt,self)
        elif opt.tts == "fishtts":
            self.tts = FishTTS(opt,self)
        elif opt.tts == "tencent":
            self.tts = TencentTTS(opt,self)
        
        self.speaking = False

        self.recording = False
        self._record_video_pipe = None
        self._record_audio_pipe = None
        self.width = self.height = 0

        self.curr_state=0
        self.custom_img_cycle = {}
        self.custom_audio_cycle = {}
        self.custom_audio_index = {}
        self.custom_index = {}
        self.custom_opt = {}
        self.__loadcustom()
        
        # 全局统一时间线索引，确保所有状态沿同一时间线循环
        self.timeline_index = 0

    def put_msg_txt(self,msg,eventpoint=None):
        self.tts.put_msg_txt(msg,eventpoint)
    
    def put_audio_frame(self,audio_chunk,eventpoint=None): #16khz 20ms pcm
        self.asr.put_audio_frame(audio_chunk,eventpoint)

    def put_audio_file(self,filebyte): 
        input_stream = BytesIO(filebyte)
        stream = self.__create_bytes_stream(input_stream)
        streamlen = stream.shape[0]
        idx=0
        while streamlen >= self.chunk:  #and self.state==State.RUNNING
            self.put_audio_frame(stream[idx:idx+self.chunk])
            streamlen -= self.chunk
            idx += self.chunk
    
    def __create_bytes_stream(self,byte_stream):
        #byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]put audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    def flush_talk(self):
        self.tts.flush_talk()
        self.asr.flush_talk()

    def is_speaking(self)->bool:
        return self.speaking
    
    def __loadcustom(self):
        for item in self.opt.customopt:
            logger.info(item)
            input_img_list = glob.glob(os.path.join(item['imgpath'], '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.custom_img_cycle[item['audiotype']] = read_imgs(input_img_list)
            self.custom_audio_cycle[item['audiotype']], sample_rate = sf.read(item['audiopath'], dtype='float32')
            self.custom_audio_index[item['audiotype']] = 0
            self.custom_index[item['audiotype']] = 0
            self.custom_opt[item['audiotype']] = item

    def init_customindex(self):
        self.curr_state=0
        for key in self.custom_audio_index:
            self.custom_audio_index[key]=0
        for key in self.custom_index:
            self.custom_index[key]=0

    def notify(self,eventpoint):
        logger.info("notify:%s",eventpoint)

    def start_recording(self):
        """开始录制视频"""
        if self.recording:
            return

        command = ['ffmpeg',
                    '-y', '-an',
                    '-f', 'rawvideo',
                    '-vcodec','rawvideo',
                    '-pix_fmt', 'bgr24', #像素格式
                    '-s', "{}x{}".format(self.width, self.height),
                    '-r', str(25),
                    '-i', '-',
                    '-pix_fmt', 'yuv420p', 
                    '-vcodec', "h264",
                    #'-f' , 'flv',                  
                    f'temp{self.opt.sessionid}.mp4']
        self._record_video_pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)

        acommand = ['ffmpeg',
                    '-y', '-vn',
                    '-f', 's16le',
                    #'-acodec','pcm_s16le',
                    '-ac', '1',
                    '-ar', '16000',
                    '-i', '-',
                    '-acodec', 'aac',
                    #'-f' , 'wav',                  
                    f'temp{self.opt.sessionid}.aac']
        self._record_audio_pipe = subprocess.Popen(acommand, shell=False, stdin=subprocess.PIPE)

        self.recording = True
        # self.recordq_video.queue.clear()
        # self.recordq_audio.queue.clear()
        # self.container = av.open(path, mode="w")
    
        # process_thread = Thread(target=self.record_frame, args=())
        # process_thread.start()
    
    def record_video_data(self,image):
        if self.width == 0:
            print("image.shape:",image.shape)
            self.height,self.width,_ = image.shape
        if self.recording:
            self._record_video_pipe.stdin.write(image.tostring())

    def record_audio_data(self,frame):
        if self.recording:
            self._record_audio_pipe.stdin.write(frame.tostring())
    
		
    def stop_recording(self):
        """停止录制视频"""
        if not self.recording:
            return
        self.recording = False 
        self._record_video_pipe.stdin.close()  #wait() 
        self._record_video_pipe.wait()
        self._record_audio_pipe.stdin.close()
        self._record_audio_pipe.wait()
        cmd_combine_audio = f"ffmpeg -y -i temp{self.opt.sessionid}.aac -i temp{self.opt.sessionid}.mp4 -c:v copy -c:a copy data/record.mp4"
        os.system(cmd_combine_audio) 
        #os.remove(output_path)

    def mirror_index(self,size, index):
        #size = len(self.coord_list_cycle)
        turn = index // size
        res = index % size
        if turn % 2 == 0:
            return res
        else:
            return size - res - 1 
    
    def get_audio_stream(self,audiotype):
        idx = self.custom_audio_index[audiotype]
        stream = self.custom_audio_cycle[audiotype][idx:idx+self.chunk]
        self.custom_audio_index[audiotype] += self.chunk
        if self.custom_audio_index[audiotype]>=self.custom_audio_cycle[audiotype].shape[0]:
            self.curr_state = 1  #当前视频不循环播放，切换到静音状态
        return stream
    
    def set_custom_state(self,audiotype, reinit=True):
        print('set_custom_state:',audiotype)
        self.curr_state = audiotype
        if reinit:
            self.custom_audio_index[audiotype] = 0
            self.custom_index[audiotype] = 0

    def process_frames(self,quit_event,loop=None,audio_track=None,video_track=None):
        enable_transition = True  # 设置为False禁用过渡效果，True启用
        
        # 状态切换控制变量
        _current_state = "silent"  # 当前状态：silent(静音) 或 speaking(说话)
        _last_audio_time = 0  # 最后一次收到音频的时间
        _silent_delay = 0.1  # 说话到静音的延迟时间（秒）
        
        # 使用全局统一时间线索引，确保所有状态沿同一时间线循环
        # self.timeline_index 在 __init__ 中初始化并在每帧末尾自增
        
        if enable_transition:
            _last_speaking = True
            _transition_start = time.time()
            _transition_duration = 0.2  # 过渡时间
            _last_silent_frame = None  # 静音帧缓存
            _last_speaking_frame = None  # 说话帧缓存
        
        if self.opt.transport=='virtualcam':
            import pyvirtualcam
            vircam = None

            audio_tmp = queue.Queue(maxsize=3000)
            audio_thread = Thread(target=play_audio, args=(quit_event,audio_tmp,), daemon=True, name="pyaudio_stream")
            audio_thread.start()
        
        while not quit_event.is_set():
            try:
                res_frame,idx,audio_frames = self.res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            
            if enable_transition:
                # 检测是否有音频输入
                has_audio = not (audio_frames[0][1]!=0 and audio_frames[1][1]!=0)
                current_time = time.time()
                
                if has_audio:
                    # 收到音频，更新最后音频时间
                    _last_audio_time = current_time
                    # 静音到说话：立即切换
                    if _current_state == "silent":
                        logger.info("状态切换：静音 → 说话（立即）")
                        _current_state = "speaking"
                        _transition_start = current_time
                    # 在延迟期间收到新音频，继续保持说话状态
                    elif _current_state == "speaking":
                        # 无需操作，继续保持说话状态
                        pass
                else:
                    # 没有音频，检查是否需要延迟切换到静音
                    if _current_state == "speaking" and (current_time - _last_audio_time) > _silent_delay:
                        logger.info(f"状态切换：说话 → 静音（延迟{_silent_delay}秒）")
                        _current_state = "silent"
                        _transition_start = current_time
                
                # 更新过渡状态用于视觉效果
                _last_speaking = (_current_state == "speaking")

            # 根据当前状态渲染帧（关键修正：只基于状态判断，不再检查音频数据）
            if _current_state == "speaking":
                # 说话状态：优先使用推理结果，失败时使用循环帧
                self.speaking = True
                
                # 检查是否有音频数据可供推理
                has_audio_data = not (audio_frames[0][1]!=0 and audio_frames[1][1]!=0)
                
                if has_audio_data and res_frame is not None:
                    # 有音频数据且有推理结果，使用推理帧
                    try:
                        current_frame = self.paste_back_frame(res_frame, idx)
                    except Exception as e:
                        logger.warning(f"paste_back_frame error: {e}")
                        # 推理失败，使用循环帧（按统一时间线索引）
                        mirror_idx = self.mirror_index(len(self.frame_list_cycle), self.timeline_index)
                        current_frame = self.frame_list_cycle[mirror_idx]
                else:
                    # 无音频数据或无推理结果（延迟期间），使用循环帧继续说话状态（按统一时间线索引）
                    mirror_idx = self.mirror_index(len(self.frame_list_cycle), self.timeline_index)
                    current_frame = self.frame_list_cycle[mirror_idx]
                    
                if enable_transition:
                    # 静音→说话过渡
                    if time.time() - _transition_start < _transition_duration and _last_silent_frame is not None:
                        alpha = min(1.0, (time.time() - _transition_start) / _transition_duration)
                        combine_frame = cv2.addWeighted(_last_silent_frame, 1-alpha, current_frame, alpha, 0)
                    else:
                        combine_frame = current_frame
                    # 缓存说话帧
                    _last_speaking_frame = combine_frame.copy()
                else:
                    combine_frame = current_frame
            else:
                # 静音状态：显示静音帧
                self.speaking = False
                audiotype = audio_frames[0][1] if audio_frames[0][1] != 0 else 1  # 默认类型
                if self.custom_index.get(audiotype) is not None: #有自定义视频
                    # 使用统一时间线索引选择自定义静音帧
                    mirindex = self.mirror_index(len(self.custom_img_cycle[audiotype]), self.timeline_index)
                    target_frame = self.custom_img_cycle[audiotype][mirindex]
                else:
                    # 静音状态也使用循环帧（按统一时间线索引）
                    mirror_idx = self.mirror_index(len(self.frame_list_cycle), self.timeline_index)
                    target_frame = self.frame_list_cycle[mirror_idx]
                
                if enable_transition:
                    # 说话→静音过渡
                    if time.time() - _transition_start < _transition_duration and _last_speaking_frame is not None:
                        alpha = min(1.0, (time.time() - _transition_start) / _transition_duration)
                        combine_frame = cv2.addWeighted(_last_speaking_frame, 1-alpha, target_frame, alpha, 0)
                    else:
                        combine_frame = target_frame
                    # 缓存静音帧
                    _last_silent_frame = combine_frame.copy()
                else:
                    combine_frame = target_frame

            if self.opt.transport=='virtualcam':
                if vircam==None:
                    height, width,_= combine_frame.shape
                    vircam = pyvirtualcam.Camera(width=width, height=height, fps=25, fmt=pyvirtualcam.PixelFormat.BGR,print_fps=True)
                vircam.send(combine_frame)
            else: #webrtc
                image = combine_frame
                image[0,:] &= 0xFE
                new_frame = VideoFrame.from_ndarray(image, format="bgr24")
                asyncio.run_coroutine_threadsafe(video_track._queue.put((new_frame,None)), loop)
            self.record_video_data(combine_frame)

            for audio_frame in audio_frames:
                frame,type,eventpoint = audio_frame
                frame = (frame * 32767).astype(np.int16)

                if self.opt.transport=='virtualcam':
                    audio_tmp.put(frame.tobytes()) #TODO
                else: #webrtc
                    new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                    new_frame.planes[0].update(frame.tobytes())
                    new_frame.sample_rate=16000
                    asyncio.run_coroutine_threadsafe(audio_track._queue.put((new_frame,eventpoint)), loop)
                self.record_audio_data(frame)
            
            # 统一时间线自增：每输出一帧画面后推进时间线索引
            self.timeline_index += 1
            
            if self.opt.transport=='virtualcam':
                vircam.sleep_until_next_frame()
        if self.opt.transport=='virtualcam':
            audio_thread.join()
            vircam.close()
        logger.info('basereal process_frames thread stop') 
    
    # def process_custom(self,audiotype:int,idx:int):
    #     if self.curr_state!=audiotype: #从推理切到口播
    #         if idx in self.switch_pos:  #在卡点位置可以切换
    #             self.curr_state=audiotype
    #             self.custom_index=0
    #     else:
    #         self.custom_index+=1