from __future__ import print_function
import time
import os
import glob2 as gb
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable
import face_recognition
import transforms as transforms
from skimage import io
from skimage.transform import resize
from models import *
import PIL
import threading
from facenet_pytorch import MTCNN, InceptionResnetV1
import queue



parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset_folder', default='./data/demo', type=str, help='dataset path')
parser.add_argument('--confidence_threshold', default=0.2, type=float, help='confidence_threshold')# 0.02
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
args = parser.parse_args(args=[])

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # print('Missing keys:{}'.format(len(missing_keys)))
    # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    # print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    # print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    # print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

device= torch.device("cpu" if args.cpu else "cuda")
torch.set_grad_enabled(False)


# 检测模型
cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50
# net and model
detection_model = RetinaFace(cfg=cfg, phase = 'test')
detection_model = load_model(detection_model, args.trained_model, args.cpu)     # cpu:defalt = False, trained_model:default='./weights/Resnet50_Final.pth'
detection_model.to(device)
detection_model.eval()
print('Detection model finished')

# 情绪模型
emotion_model = VGG('VGG19')
checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'))
emotion_model.load_state_dict(checkpoint['net'])
emotion_model.to(device)
emotion_model.eval()
print('Emotion model finished')

# 身份识别模型
face_model = InceptionResnetV1(pretrained='vggface2',device = 'cuda').eval()
face_model.to(device)
face_model.eval()
print('Identity model finished')


cut_size = 44
transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),])



# 选择输入的视频（摄像头）
# capture = cv2.VideoCapture(0)
# capture.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
# capture = cv2.VideoCapture("C:/Users/Administrator/Desktop/面部识别/二十八所测试/demo02.mp4")
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fps = capture.get(cv2.CAP_PROP_FPS)/2
# size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# output_movie = cv2.VideoWriter('1.mp4', fourcc, fps, size)

# 编码人脸库
img_path = gb.glob(r'./photo/*.jpg')
known_face_names = []
known_face_encodings = []
for i in img_path:
    picture_name = i.replace('./photo/*.jpg', '')
    picture_name = picture_name.replace('./photo\\', '')
    picture_newname = picture_name.replace('.jpg', '')
    frame = cv2.imread(i)
    img = np.float32(frame)
    resize_num = 0.3
    if resize_num != 1:
        img = cv2.resize(img, None, None, fx=resize_num, fy=resize_num, interpolation=cv2.INTER_LINEAR)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    loc, conf, landms = detection_model(img)

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize_num
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize_num
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1]
    # order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    # dets = dets[:args.keep_top_k, :]
    # landms = landms[:args.keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)

    bboxs = dets

    if bboxs.size != 0:
        for box in bboxs:
            x = int(box[0])
            y = int(box[1])
            w = int(box[2]) - int(box[0])
            h = int(box[3]) - int(box[1])
            confidence = str(box[4])
            if x < 0 or y < 0:
                break
            raw_img = frame[y:y + h, x:x + w]
            raw_img = raw_img.transpose(2, 0, 1)
            raw_img = torch.from_numpy(raw_img).unsqueeze(0)
            raw_img = F.pad(raw_img, (50, 50, 50, 50))
            raw_img = raw_img.float()
            raw_img = raw_img.cuda()
            face_encodings = face_model(raw_img)
            someone_face_encoding = face_encodings
        if len(someone_face_encoding) == 0:
            break
        else:
            someone_face_encoding = someone_face_encoding[0].cpu().numpy()
            known_face_names.append(picture_newname)
            known_face_encodings.append(someone_face_encoding)
print(known_face_names)
someone_img = []
someone_face_encoding = []
face_locations = []
face_encodings = []
face_names = []




def get_box(frame_queue1, box_queue1, frame_queue2, box_queue2):
    # capture = cv2.VideoCapture(0)
    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
    capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    while True:
        # time1 = time.time()
        ret, frame = capture.read()
        if not ret:
            break
        time1 = time.time()
        img = np.float32(frame)
        resize_num = 0.3
        img = cv2.resize(img, None, None, fx=resize_num, fy=resize_num, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        loc, conf, landms = detection_model(img)
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize_num
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize_num
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        # dets = dets[:args.keep_top_k, :]
        # landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        bboxs = dets
        # if len(bboxs) == 0:
        #     bboxs=[[239.73961,   109.76238,   399.9596,    311.41574,0,
        #             0,0,0,0,0,
        #             0,0,0,0,0]]
        time_used = time.time() - time1
        print('detection::::', 1 / time_used)
        # print(bboxs)
        box_queue1.put(bboxs)
        box_queue2.put(bboxs)
        frame_queue1.put(frame)
        frame_queue2.put(frame)


def get_emotion_thread(frame_queue1, box_queue1, emotion_queue):
    torch.set_grad_enabled(False)
    # emotion_frame_counter = 1
    while True:
        frame = frame_queue1.get()
        bboxs = box_queue1.get()
        emotions = []
        if len(bboxs) != 0:
            time1 = time.time()
            for box in bboxs:
                if box[0] > 0:
                    x = int(box[0])
                else:
                    x = 1
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])
                if x < 0 or y < 0:
                    break
                raw_img = frame[y:y + h, x:x + w]
                gray = rgb2gray(raw_img)
                gray = resize(gray, (48, 48)).astype(np.uint8)
                img = gray[:, :, np.newaxis]

                img = np.concatenate((img, img, img), axis=2)
                img = Image.fromarray(img)
                inputs = transform_test(img)


                ncrops, c, h_resize, w_resize = np.shape(inputs)

                inputs = inputs.view(-1, c, h_resize, w_resize)
                inputs = inputs.cuda()
                inputs = Variable(inputs, volatile=True)
                outputs = emotion_model(inputs)

                outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

                score = F.softmax(outputs_avg)
                _, predicted = torch.max(outputs_avg.data, 0)
                time_used = time.time() - time1
                print('emotion::::', 1/time_used)
                emotions.append(predicted)
            emotion_queue.put(emotions)
        else:
            emotion_queue.put(emotions)



def get_face_thread(frame_queue2, box_queue2, emotion_queue):
    torch.set_grad_enabled(False)
    face_frame_counter = 1
    while True:
        frame = frame_queue2.get()
        bboxs = box_queue2.get()
        emotions = emotion_queue.get()
        time1 = time.time()
        face_counter = 0

        for box in bboxs:
            if box[0] > 0:
                x = int(box[0])
            else:
                x = 1
            y = int(box[1])
            w = int(box[2]) - int(box[0])
            h = int(box[3]) - int(box[1])
            if x < 0 or y < 0:
                break
            if face_frame_counter == 1:
                raw_img = frame[y:y + h, x:x + w]
                # resize_num = 0.3
                # raw_img = cv2.resize(raw_img, None, None, fx=resize_num, fy=resize_num, interpolation=cv2.INTER_LINEAR)
                raw_img = raw_img.transpose(2, 0, 1)
                raw_img = torch.from_numpy(raw_img).unsqueeze(0)
                raw_img = F.pad(raw_img, (50, 50, 50, 50))

                raw_img = raw_img.float()
                raw_img = raw_img.cuda()
                face_encodings = []
                face_encodings = face_model(raw_img)
                # for i in range(len(face_encodings1)):
                #     face = face_encodings1[i]
                #     face_encodings.append(face)
                face_names = []

                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    face_encoding = face_encoding.cpu().numpy()
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                    name = "Unknown"

                    # # If a match was found in known_face_encodings, just use the first one.
                    # if True in matches:
                    #     first_match_index = matches.index(True)
                    #     name = known_face_names[first_match_index]

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)
                for name in face_names:
                    # cv2.rectangle(frame, (left, bottom), (right, bottom + 25), (0, 0, 255), cv2.FILLED)
                    cv2.putText(frame, name, (x, y + h + 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

                class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                predicted = emotions[face_counter]
                face_counter += 1
                # for index, emotion in enumerate(class_names):
                #     cv2.putText(frame, emotion, (10, index * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
                #     cv2.rectangle(frame, (130, index * 20 + 10),
                #                   (130 + int(score.data.cpu().numpy()[index] * 100), (index + 1) * 20 + 4),
                #                   (255, 0, 0), -1)

                cv2.putText(frame, class_names[int(predicted.cpu().numpy())], (x - 20, y - 20),
                            cv2.FONT_HERSHEY_PLAIN, 2,
                            (0, 255, 0), 1)
                time_used = time.time() - time1
                print('face::::', 1/time_used)

            else:
                for name in face_names:
                    # cv2.rectangle(frame, (left, bottom), (right, bottom + 25), (0, 0, 255), cv2.FILLED)
                    cv2.putText(frame, name, (x, y + h + 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

                class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                predicted = emotions[face_counter]
                face_counter += 1
                # for index, emotion in enumerate(class_names):
                #     cv2.putText(frame, emotion, (10, index * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
                #     cv2.rectangle(frame, (130, index * 20 + 10),
                #                   (130 + int(score.data.cpu().numpy()[index] * 100), (index + 1) * 20 + 4),
                #                   (255, 0, 0), -1)

                cv2.putText(frame, class_names[int(predicted.cpu().numpy())], (x - 20, y - 20), cv2.FONT_HERSHEY_PLAIN, 2,
                            (0, 255, 0), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if len(bboxs) != 0:
            cv2.putText(frame, str(1 / (time.time()-time1)), (20, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1)
        cv2.imshow('face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # face_frame_counter += 1


if __name__ == '__main__':
    frame_queue1 = queue.Queue(10)
    box_queue1 = queue.Queue(10)
    frame_queue2 = queue.Queue(10)
    box_queue2 = queue.Queue(10)
    emotion_queue = queue.Queue(10)

    box_thread = threading.Thread(target=get_box , args=(frame_queue1, box_queue1, frame_queue2, box_queue2) )
    emotion_thread = threading.Thread(target=get_emotion_thread , args=(frame_queue1, box_queue1, emotion_queue) )
    identity_thread = threading.Thread(target=get_face_thread , args=(frame_queue2, box_queue2, emotion_queue) )


    box_thread.start()
    # box_thread.join()
    emotion_thread.start()
    # emotion_thread.join()
    identity_thread.start()

    box_thread.join()
    emotion_thread.join()