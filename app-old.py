from flask import Flask, render_template,send_from_directory,request, jsonify
from flask_sockets import Sockets
import base64
import json
import re
import numpy as np
from threading import Thread,Event
import torch.multiprocessing as mp

from aiohttp import web
import aiohttp
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription,RTCIceServer,RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender
from webrtc import HumanPlayer
from basereal import BaseReal
from llm import llm_response

import argparse
import random
import shutil
import asyncio
import torch
import gc
import time
import threading

from typing import Dict
from logger import logger


app = Flask(__name__)
#sockets = Sockets(app)
nerfreals = []
opt = None
model = None
avatar = None
        

pcs = set()

def clean_cache():

    gc.collect()

    torch.cuda.empty_cache()

    print("Cache cleaned!")

def periodic_cache_clean():

    while True:
        time.sleep(3600)  # 每小时清理一次缓存（可以根据需求调整时间）
        clean_cache()

def start_cache_cleaner():
    cache_thread = threading.Thread(target=periodic_cache_clean, daemon=True)
    cache_thread.start()
    print("Cache cleaner thread started.")



def randN(N)->int:
    '''生成长度为 N的随机数 '''
    min = pow(10, N - 1)
    max = pow(10, N)
    return random.randint(min, max - 1)

def build_nerfreal(sessionid:int)->BaseReal:
    opt.sessionid=sessionid
    if opt.model == 'wav2lip':
        from lipreal import LipReal
        nerfreal = LipReal(opt,model,avatar)
    elif opt.model == 'musetalk':
        from musereal import MuseReal
        nerfreal = MuseReal(opt,model,avatar)
    # elif opt.model == 'ernerf':
    #     from nerfreal import NeRFReal
    #     nerfreal = NeRFReal(opt,model,avatar)
    elif opt.model == 'ultralight':
        from lightreal import LightReal
        nerfreal = LightReal(opt,model,avatar)
    return nerfreal

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    # 使用递增 sessionid（从 0 开始）
    sessionid = len(nerfreals)
    # 确保 nerfreals 列表足够大以容纳新 sessionid
    if sessionid >= len(nerfreals):
        nerfreals.append(None)  # 预分配空间
    logger.info(f"New session created with sessionid={sessionid}")
    # 初始化 nerfreals 实例
    try:
        nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal, sessionid)
        nerfreals[sessionid] = nerfreal
    except Exception as e:
        logger.error(f"Failed to build nerfreal for sessionid={sessionid}: {e}")
        return web.Response(status=500)

    # 创建 WebRTC 连接并返回 SDP 和 sessionid
    pc = RTCPeerConnection()
    pcs.add(pc)

    async def on_connectionstatechange():
        logger.info(f"Connection state is {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            await pc.close()
            pcs.discard(pc)
            if sessionid < len(nerfreals):
                del nerfreals[sessionid]
    pc.on("connectionstatechange", on_connectionstatechange)

    try:
        # 初始化 HumanPlayer
        player = HumanPlayer(nerfreals[sessionid])
        # 记录日志（确保 player 已定义）
        logger.info(f"Added audio track: {player.audio}")
        logger.info(f"Added video track: {player.video}")
        # 添加媒体轨道
        audio_sender = pc.addTrack(player.audio)
        video_sender = pc.addTrack(player.video)
        # 设置视频编码偏好
        capabilities = RTCRtpSender.getCapabilities("video")
        preferences = list(filter(lambda x: x.name in ["H264", "VP8", "rtx"], capabilities.codecs))
        transceiver = pc.getTransceivers()[1]
        transceiver.setCodecPreferences(preferences)
    except Exception as e:
        logger.error(f"Failed to create HumanPlayer or add tracks for sessionid={sessionid}: {e}")
        await pc.close()
        pcs.discard(pc)
        return web.Response(status=500)

    # 处理 WebRTC 握手
    try:
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
                "sessionid": sessionid
            }),
        )
    except Exception as e:
        logger.error(f"Failed to set remote description or create answer for sessionid={sessionid}: {e}")
        await pc.close()
        pcs.discard(pc)
        return web.Response(status=500)

async def human(request):
    params = await request.json()
    sessionid = params.get("sessionid", 0)

    # 7. 检查 sessionid 是否有效
    if sessionid >= len(nerfreals) or sessionid < 0:
        logger.warning(f"Session {sessionid} not found in nerfreals")
        return web.json_response({"code": 1, "message": "Session not found"})

    if params.get("interrupt", False):
        try:
            nerfreals[sessionid].flush_talk()
        except Exception as e:
            logger.error(f"Error flushing talk for sessionid={sessionid}: {e}")
            return web.json_response({"code": 1, "message": "Flush failed"})

    if params["type"] == "echo":
        try:
            nerfreals[sessionid].put_msg_txt(params["text"])
            return web.json_response({"code": 0, "data": "Text sent"})
        except Exception as e:
            logger.error(f"Error sending text to sessionid={sessionid}: {e}")
            return web.json_response({"code": 1, "message": "Text sending failed"})

    elif params["type"] == "chat":
        try:
            res = await asyncio.get_event_loop().run_in_executor(
                None, llm_response, params["text"], nerfreals[sessionid]
            )
            return web.json_response({"code": 0, "data": "Response generated"})
        except Exception as e:
            logger.error(f"Error generating response for sessionid={sessionid}: {e}")
            return web.json_response({"code": 1, "message": "Response generation failed"})

    else:
        logger.warning(f"Unknown type: {params['type']}")
        return web.json_response({"code": 1, "message": "Unknown request type"})

async def humanaudio(request):
    try:
        form= await request.post()
        sessionid = int(form.get('sessionid',0))
        fileobj = form["file"]
        filename=fileobj.filename
        filebytes=fileobj.file.read()
        nerfreals[sessionid].put_audio_file(filebytes)

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg":"err","data": ""+e.args[0]+""}
            ),
        )

async def set_audiotype(request):
    params = await request.json()

    sessionid = params.get('sessionid',0)    
    nerfreals[sessionid].set_custom_state(params['audiotype'],params['reinit'])

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data":"ok"}
        ),
    )

async def record(request):
    params = await request.json()

    sessionid = params.get('sessionid',0)
    if params['type']=='start_record':
        # nerfreals[sessionid].put_msg_txt(params['text'])
        nerfreals[sessionid].start_recording()
    elif params['type']=='end_record':
        nerfreals[sessionid].stop_recording()
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data":"ok"}
        ),
    )


async def is_speaking(request):
    params = await request.json()
    sessionid = params.get("sessionid", 0)

    # 8. 检查 sessionid 是否有效
    if sessionid >= len(nerfreals) or sessionid < 0:
        logger.warning(f"Session {sessionid} not found in nerfreals")
        return web.json_response({"code": 1, "message": "Session not found"})

    try:
        # 先判断是否是 coroutine function
        method = nerfreals[sessionid].is_speaking
        if asyncio.iscoroutinefunction(method):
            data = await method()
        else:
            data = method()
        return web.json_response({"code": 0, "data": data})
    except Exception as e:
        logger.error(f"Error in is_speaking for sessionid={sessionid}: {e}")
        return web.json_response({"code": 1, "message": "Internal error"})

async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

async def post(url,data):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url,data=data) as response:
                return await response.text()
    except aiohttp.ClientError as e:
        logger.info(f'Error: {e}')

async def run(push_url,sessionid):
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal,sessionid)
    nerfreals[sessionid] = nerfreal

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    player = HumanPlayer(nerfreals[sessionid])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)

    await pc.setLocalDescription(await pc.createOffer())
    answer = await post(push_url,pc.localDescription.sdp)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer,type='answer'))
                                            
if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    
    # audio FPS
    parser.add_argument('--fps', type=int, default=50, help="audio fps,must be 50")
    # sliding window left-middle-right length (unit: 20ms)
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=8)
    parser.add_argument('-r', type=int, default=10)

    parser.add_argument('--W', type=int, default=450, help="GUI width")
    parser.add_argument('--H', type=int, default=450, help="GUI height")

    #musetalk opt
    parser.add_argument('--avatar_id', type=str, default='avator_1', help="define which avatar in data/avatars")
    #parser.add_argument('--bbox_shift', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16, help="infer batch")

    parser.add_argument('--customvideo_config', type=str, default='', help="custom action json")

    parser.add_argument('--tts', type=str, default='edgetts', help="tts service type") #xtts gpt-sovits cosyvoice
    parser.add_argument('--REF_FILE', type=str, default="zh-CN-YunxiaNeural")
    parser.add_argument('--REF_TEXT', type=str, default=None)
    parser.add_argument('--TTS_SERVER', type=str, default='http://127.0.0.1:9880') # http://localhost:9000
    # parser.add_argument('--CHARACTER', type=str, default='test')
    # parser.add_argument('--EMOTION', type=str, default='default')

    parser.add_argument('--model', type=str, default='musetalk') #musetalk wav2lip ultralight

    parser.add_argument('--transport', type=str, default='rtcpush') #webrtc rtcpush virtualcam
    parser.add_argument('--push_url', type=str, default='http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream') #rtmp://localhost/live/livestream

    parser.add_argument('--max_session', type=int, default=1)  #multi session count
    parser.add_argument('--listenport', type=int, default=8010, help="web listen port")

    opt = parser.parse_args()
    #app.config.from_object(opt)
    #print(app.config)
    opt.customopt = []
    if opt.customvideo_config!='':
        with open(opt.customvideo_config,'r') as file:
            opt.customopt = json.load(file)

    # if opt.model == 'ernerf':       
    #     from nerfreal import NeRFReal,load_model,load_avatar
    #     model = load_model(opt)
    #     avatar = load_avatar(opt) 
    if opt.model == 'musetalk':
        from musereal import MuseReal,load_model,load_avatar,warm_up
        logger.info(opt)
        model = load_model()
        avatar = load_avatar(opt.avatar_id) 
        warm_up(opt.batch_size,model)      
    elif opt.model == 'wav2lip':
        from lipreal import LipReal,load_model,load_avatar,warm_up
        logger.info(opt)
        model = load_model("./models/wav2lip.pth")
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,model,256)
    elif opt.model == 'ultralight':
        from lightreal import LightReal,load_model,load_avatar,warm_up
        logger.info(opt)
        model = load_model(opt)
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,avatar,160)

    # if opt.transport=='rtmp':
    #     thread_quit = Event()
    #     nerfreals[0] = build_nerfreal(0)
    #     rendthrd = Thread(target=nerfreals[0].render,args=(thread_quit,))
    #     rendthrd.start()
    if opt.transport=='virtualcam':
        thread_quit = Event()
        # 确保 nerfreals 列表至少有一个元素
        if len(nerfreals) == 0:
            nerfreals.append(None)
        nerfreals[0] = build_nerfreal(0)
        rendthrd = Thread(target=nerfreals[0].render,args=(thread_quit,))
        rendthrd.start()

    #############################################################################
    appasync = web.Application(client_max_size=1024**2*100)
    appasync.on_shutdown.append(on_shutdown)
    appasync.router.add_post("/offer", offer)
    appasync.router.add_post("/human", human)
    appasync.router.add_post("/humanaudio", humanaudio)
    appasync.router.add_post("/set_audiotype", set_audiotype)
    appasync.router.add_post("/record", record)
    appasync.router.add_post("/is_speaking", is_speaking)
    appasync.router.add_static('/',path='web')

    # Configure default CORS settings.
    cors = aiohttp_cors.setup(appasync, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })
    # Configure CORS on all routes.
    for route in list(appasync.router.routes()):
        cors.add(route)

    pagename='webrtcapi.html'
    if opt.transport=='rtmp':
        pagename='echoapi.html'
    elif opt.transport=='rtcpush':
        pagename='rtcpushapi.html'
    logger.info('start http server; http://<serverip>:'+str(opt.listenport)+'/'+pagename)
    logger.info('如果使用webrtc，推荐访问webrtc集成前端: http://<serverip>:'+str(opt.listenport)+'/dashboard.html')
    def run_server(runner):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '0.0.0.0', opt.listenport)
        loop.run_until_complete(site.start())
        if opt.transport=='rtcpush':
            for k in range(opt.max_session):
                push_url = opt.push_url
                if k!=0:
                    push_url = opt.push_url+str(k)
                loop.run_until_complete(run(push_url,k))
        loop.run_forever()    
    #Thread(target=run_server, args=(web.AppRunner(appasync),)).start()
    run_server(web.AppRunner(appasync))