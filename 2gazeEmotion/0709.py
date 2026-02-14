import argparse
import logging
import os
import re
import pathlib
import warnings
import json
from timeit import timeit
from demo import output_video_file_path

import torch
from omegaconf import DictConfig, OmegaConf

from demo import Demo
from utils01 import (check_path_all, download_dlib_pretrained_model,
                    download_ethxgaze_model, download_mpiifacegaze_model,
                    download_mpiigaze_model, expanduser_all,
                    generate_dummy_camera_params)

logger = logging.getLogger(__name__)
output_dir = r'.\results\camera'  # 设置输出目录 \output_results.txt


def globalize():
    global faceLandmarks
    faceLandmarks = []

# 参数解析
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        help='Config file. When using a config file, all the other '
        'commandline arguments are ignored. '
    )
    parser.add_argument(
        '--mode',
        default='mpiigaze',
        type=str,
        choices=['mpiigaze', 'mpiifacegaze', 'eth-xgaze'],
        help='With \'mpiigaze\', MPIIGaze model will be used. '
        'With \'mpiifacegaze\', MPIIFaceGaze model will be used. '
        'With \'eth-xgaze\', ETH-XGaze model will be used.')
    parser.add_argument(
        '--face-detector',
        type=str,
        default='mediapipe',
        choices=[
            'dlib', 'face_alignment_dlib', 'face_alignment_sfd', 'mediapipe'
        ],
        help='The method used to detect faces and find face landmarks '
        '(default: \'mediapipe\')')
    parser.add_argument('--device',
                        type=str,
                        choices=['cpu', 'cuda'],
                        help='Device used for model inference.')
    parser.add_argument('--image',
                        type=str,
                        help='Path to an input image file.')
    parser.add_argument('--video',
                        type=str,
                        help='Path to an input video file.')
    parser.add_argument(
        '--camera',
        type=str,
        help='Camera calibration file. '
    )
    parser.add_argument(
        '--output-dir',
        '-o',
        default=output_video_file_path,
        type=str,
        help='If specified, the overlaid video will be saved to this directory.'
    )
    parser.add_argument(
        '--exampleoutput-dir',
        '-p',
        default='./exampleoutput',
        type=str,
        help='If specified, one slide of the video will be saved to this directory.'
    )
    parser.add_argument('--ext',
                        '-e',
                        default='mp4',
                        type=str,
                        choices=['avi', 'mp4'],
                        help='Output video file extension。')
    parser.add_argument(
        '--no-screen',
        action='store_true',
        help='If specified, the video is not displayed on screen, and saved '
        'to the output directory.')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


# 加载模型参数
def load_mode_config(args: argparse.Namespace) -> DictConfig:
    package_root = pathlib.Path(__file__).parent.resolve()
    if args.mode == 'mpiigaze':
        path = package_root / 'data/configs/mpiigaze.yaml'
    elif args.mode == 'mpiifacegaze':
        path = package_root / 'data/configs/mpiifacegaze.yaml'
    elif args.mode == 'eth-xgaze':
        path = package_root / 'data/configs/eth-xgaze.yaml'
    else:
        raise ValueError
    config = OmegaConf.load(path)
    config.PACKAGE_ROOT = package_root.as_posix()

    if args.face_detector:
        config.face_detector.mode = args.face_detector
    if args.device:
        config.device = args.device
    if config.device == 'cuda' and not torch.cuda.is_available():
        config.device = 'cpu'
        warnings.warn('Run on CPU because CUDA is not available.')
    if args.image and args.video:
        raise ValueError('Only one of --image or --video can be specified.')
    if args.image:
        config.demo.image_path = args.image
        config.demo.use_camera = False
    if args.video:
        config.demo.use_camera = False
    if args.camera:
        config.gaze_estimator.camera_params = args.camera
    elif args.image or args.video:
        config.gaze_estimator.use_dummy_camera_params = True
    if args.output_dir:
        config.demo.output_dir = args.output_dir
    if args.ext:
        config.demo.output_file_extension = args.ext
    if args.no_screen:
        config.demo.display_on_screen = False
        if not config.demo.output_dir:
            config.demo.output_dir = 'outputs'

    return config


def is_video_file(file_path):
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.flv']
    return os.path.splitext(file_path)[1].lower() in video_extensions


def get_video_files_from_dir(dir_path):
    video_files = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if is_video_file(os.path.join(root, file)):
                video_files.append(os.path.join(root, file))
    return video_files


def process_video_input(config, video_input):
    if os.path.isdir(video_input):
        video_files = get_video_files_from_dir(video_input)
        if video_files:
            print(f"Found {len(video_files)} video files.")
            for video_file in video_files:
                video_file = re.sub(r"'", "", video_file)
                # 临时解除只读状态
                OmegaConf.set_readonly(config, False)
                config.demo.video_path = video_file
                if not config.demo.output_dir:
                    config.demo.output_dir = output_video_file_path

                if config.gaze_estimator.use_dummy_camera_params:
                    generate_dummy_camera_params(config)
                OmegaConf.set_readonly(config, True)
                demo = Demo(config)
                demo.run()
        else:
            print("No video files found in the directory.")
    elif os.path.isfile(video_input):
        if is_video_file(video_input):
            # 临时解除只读状态
            OmegaConf.set_readonly(config, False)
            config.demo.video_path = video_input
            if not config.demo.output_dir:
                config.demo.output_dir = output_dir  # 设置输出目录
            if config.gaze_estimator.use_dummy_camera_params:
                generate_dummy_camera_params(config)
            OmegaConf.set_readonly(config, True)
            demo = Demo(config)
            demo.run()
        else:
            print("Error: The file is not a supported video file.")
    else:
        print("Error: The path is neither a file nor a directory.")


def main():
    args = parse_args()
    if args.debug:
        logging.getLogger('ptgaze').setLevel(logging.DEBUG)

    try:
        if args.config:
            config = OmegaConf.load(args.config)
        elif args.mode:
            config = load_mode_config(args)
        else:
            raise ValueError("You need to specify one of '--mode' or '--config'.")

        expanduser_all(config)

        OmegaConf.set_readonly(config, True)
        logger.info(OmegaConf.to_yaml(config))

        if config.face_detector.mode == 'dlib':
            download_dlib_pretrained_model()
        if args.mode:
            if config.mode == 'MPIIGaze':
                download_mpiigaze_model()
            elif config.mode == 'MPIIFaceGaze':
                download_mpiifacegaze_model()
            elif config.mode == 'ETH-XGaze':
                download_ethxgaze_model()

        check_path_all(config)
        if config.demo.use_camera:
            if config.gaze_estimator.use_dummy_camera_params:
                generate_dummy_camera_params(config)
            if not config.demo.output_dir:
                config.demo.output_dir = output_dir
            demo = Demo(config)
            demo.run()
        else:
            process_video_input(config, args.video)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


def process_video_input(config, video_input):
    if os.path.isdir(video_input):
        video_files = get_video_files_from_dir(video_input)
        if video_files:
            print(f"Found {len(video_files)} video files.")
            for video_file in video_files:
                video_file = re.sub(r"'", "", video_file)
                # 临时解除只读状态
                OmegaConf.set_readonly(config, False)
                config.demo.video_path = video_file
                if config.gaze_estimator.use_dummy_camera_params:
                    generate_dummy_camera_params(config)
                OmegaConf.set_readonly(config, True)
                demo = Demo(config)
                demo.run()
        else:
            print("No video files found in the directory.")
    elif os.path.isfile(video_input):
        if is_video_file(video_input):
            # 临时解除只读状态
            OmegaConf.set_readonly(config, False)
            config.demo.video_path = video_input
            if config.gaze_estimator.use_dummy_camera_params:
                generate_dummy_camera_params(config)
            OmegaConf.set_readonly(config, True)
            demo = Demo(config)
            demo.run()
        else:
            print("Error: The file is not a supported video file.")
    else:
        print("Error: The path is neither a file nor a directory.")

if __name__ == '__main__':
    main()
    globalize()
