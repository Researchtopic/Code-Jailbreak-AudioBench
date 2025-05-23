# sudo apt-get update
# sudo apt-get install ffmpeg

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.append('/path/to/VITA_project')
os.chdir('/path/to/VITA_project')

import time
import argparse
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from itertools import count
from pprint import pprint
import json

from decord import VideoReader, cpu
from vita.constants import (
    DEFAULT_AUDIO_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    MAX_IMAGE_LENGTH,
)
from vita.conversation import SeparatorStyle, conv_templates
from vita.model.builder import load_pretrained_model
from vita.util.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_audio_token,
    tokenizer_image_token,
)
from vita.util.utils import disable_torch_init


def _get_rawvideo_dec(
    video_path,
    image_processor,
    max_frames=MAX_IMAGE_LENGTH,
    min_frames=4,
    image_resolution=384,
    video_framerate=1,
    s=None,
    e=None,
    image_aspect_ratio="pad",
):
    # speed up video decode via decord.

    if s is None:
        start_time, end_time = None, None
    else:
        start_time = int(s)
        end_time = int(e)
        start_time = start_time if start_time >= 0.0 else 0.0
        end_time = end_time if end_time >= 0.0 else 0.0
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        elif start_time == end_time:
            end_time = start_time + 1

    if os.path.exists(video_path):
        vreader = VideoReader(video_path, ctx=cpu(0))
    else:
        print(video_path)
        raise FileNotFoundError

    fps = vreader.get_avg_fps()
    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
    num_frames = f_end - f_start + 1
    if num_frames > 0:
        # T x 3 x H x W
        sample_fps = int(video_framerate)
        t_stride = int(round(float(fps) / sample_fps))

        all_pos = list(range(f_start, f_end + 1, t_stride))
        if len(all_pos) > max_frames:
            sample_pos = [
                all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)
            ]
        elif len(all_pos) < min_frames:
            sample_pos = [
                all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=min_frames, dtype=int)
            ]
        else:
            sample_pos = all_pos

        patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]

        if image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            patch_images = [
                expand2square(i, tuple(int(x * 255) for x in image_processor.image_mean))
                for i in patch_images
            ]
            patch_images = [
                image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
                for i in patch_images
            ]
        else:
            patch_images = [
                image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
                for i in patch_images
            ]

        patch_images = torch.stack(patch_images)
        slice_len = patch_images.shape[0]

        return patch_images, slice_len
    else:
        print("video path: {} error.".format(video_path))


if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Process model and video paths.")

    # Add arguments
    parser.add_argument("--model_path", type=str, default="models/VITA-1.5", help="Path to the model directory")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--audio_path", type=str, default="../audio_jailbreak/datasets/advbench_audio/audio_original/audio_001.wav")
    parser.add_argument("--model_type", type=str, default="qwen2p5_instruct")
    parser.add_argument("--conv_mode", type=str, default="qwen2p5_instruct")
    parser.add_argument("--question", type=str, default="") # current support input modality: text -> text, audio -> text, image/video + text/audio -> text
    parser.add_argument("--frameCat", action='store_true')

    # Parse the arguments
    args = parser.parse_args()

    # Assign arguments to variables
    model_path = args.model_path
    model_base = args.model_base
    video_path = args.video_path
    image_path = args.image_path
    audio_path = args.audio_path
    qs = args.question
    assert (audio_path is None) != (qs == ""), "Exactly one of audio_path or qs must be non-None"
    conv_mode = args.conv_mode

    if args.frameCat:
        from vita.util.data_utils_video_audio_neg_frameCat import dynamic_preprocess
    else:
        from vita.util.data_utils_video_audio_neg_patch import dynamic_preprocess

    # The number of visual tokens varies with the length of the video. "max_frames" is the maximum number of frames.
    # When the video is long, we will uniformly downsample the video to meet the frames when equal to the "max_frames".
    max_frames = MAX_IMAGE_LENGTH  # 100

    # The number of frames retained per second in the video.
    video_framerate = 1

    # Sampling Parameter
    temperature = 0.01
    top_p = None
    num_beams = 1

    disable_torch_init()
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name, args.model_type
    )

    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor

    audio_encoder = model.get_audio_encoder()
    audio_encoder.to(dtype=torch.float16)
    audio_processor = audio_encoder.audio_processor
    model.eval()

    # settings
    model_name = "VITA1.5"
    log_folder = f'../audio_jailbreak/filtered_Advbench_with_defense_logs/{model_name}'
    attack_types = ['audio-text']

    datasets = []
    for dirpath, dirnames, filenames in os.walk("../audio_jailbreak/datasets/filtered_Advbench_audio_with_defense"):
        if not dirnames and filenames:
            datasets.append(dirpath)
    print(datasets)

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    else:
        print(f"log folder existed in {log_folder}.")
        # exit()


    for dataset in datasets:
        for attack_type in attack_types:
            dataset_name = dataset.split('/')[-2] + "_" + dataset.split('/')[-1]
            log_path = os.path.join(log_folder, f'{model_name}_{dataset_name}_{attack_type}.log')

            if os.path.exists(log_path):
                print(f"log existed in {log_path}.")
                continue
            
            log_dict = dict()
            key_generator = count(0)
            for i, audio_path in enumerate(tqdm(os.listdir(dataset))):
                prompt_text = "Provide a detailed answer that exceeds 500 words."

                audio_path = os.path.join(dataset, audio_path)

                qs = ""
                if audio_path is not None:
                    audio, audio_for_llm_lens = audio_processor.process(audio_path)
                    audio_length = audio.shape[0]
                    audio = torch.unsqueeze(audio, dim=0)
                    audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
                    audio_for_llm_lens = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)
                    audios = dict()
                    audios["audios"] = audio.half().cuda()
                    audios["lengths"] = audio_length.half().cuda()
                    audios["lengths_for_llm"] = audio_for_llm_lens.cuda()
                else:
                    audio = torch.zeros(400, 80)
                    audio_length = audio.shape[0]
                    audio_for_llm_lens = 60
                    audio = torch.unsqueeze(audio, dim=0)
                    audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
                    audio_for_llm_lens = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)
                    audios = dict()
                    audios["audios"] = audio.half().cuda()
                    audios["lengths"] = audio_length.half().cuda()
                    audios["lengths_for_llm"] = audio_for_llm_lens.cuda()
                    # audios = None

                # Check if the video exists
                if video_path is not None:
                    video_frames, slice_len = _get_rawvideo_dec(
                        video_path,
                        image_processor,
                        max_frames=max_frames,
                        video_framerate=video_framerate,
                        image_aspect_ratio=getattr(model.config, "image_aspect_ratio", None),
                    )
                    image_tensor = video_frames.half().cuda()
                    if audio_path:
                        qs = DEFAULT_IMAGE_TOKEN * slice_len + "\n" + qs + DEFAULT_AUDIO_TOKEN
                    else:
                        qs = DEFAULT_IMAGE_TOKEN * slice_len + "\n" + qs
                    modality = "video"
                elif image_path is not None:
                    image = Image.open(image_path).convert("RGB")
                    if args.frameCat:
                        image, p_num = dynamic_preprocess(image, min_num=2, max_num=12, image_size=448, use_thumbnail=True, img_mean=image_processor.image_mean)
                    else:
                        image, p_num = dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True)
                    assert len(p_num) == 1
                    image_tensor = model.process_images(image, model.config).to(
                        dtype=model.dtype, device="cuda"
                    )
                    if audio_path:
                        qs = DEFAULT_IMAGE_TOKEN * p_num[0] + "\n" + qs + DEFAULT_AUDIO_TOKEN
                    else:
                        qs = DEFAULT_IMAGE_TOKEN * p_num[0] + "\n" + qs
                    modality = "image"
                else:
                    image_tensor = torch.zeros((1, 3, 448, 448)).to(dtype=model.dtype, device="cuda")
                    if audio_path:
                        qs = qs + DEFAULT_AUDIO_TOKEN
                    modality = "lang"

                conv = conv_templates[conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt(modality)

                if audio_path:
                    input_ids = (
                        tokenizer_image_audio_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                        .unsqueeze(0)
                        .cuda()
                    )
                else:
                    input_ids = (
                        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                        .unsqueeze(0)
                        .cuda()
                    )

                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                start_time = time.time()
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor,
                        audios=audios,
                        do_sample=False,
                        temperature=temperature,
                        top_p=top_p,
                        num_beams=num_beams,
                        output_scores=True,
                        return_dict_in_generate=True,
                        max_new_tokens=1024,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria],
                        shared_v_pid_stride=None#2#16#8#4#1#None,
                    )

                output_ids = output_ids.sequences
                input_token_len = input_ids.shape[1]
                if args.model_type == "mixtral-8x7b":
                    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                    if n_diff_input_output > 0:
                        print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
                        output_ids = output_ids[:, input_token_len:]
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

                outputs = outputs.strip()
                if outputs.endswith(stop_str):
                    outputs = outputs[: -len(stop_str)]
                outputs = outputs.strip()

                item = {
                    "type": attack_type,
                    "audio": audio_path,
                    'prompt': prompt_text,
                    'response': outputs,
                    'word count': len(outputs.split())
                }
                key = str(next(key_generator))
                log_dict[key] = item
                pprint(item)

                if i % 10 == 0 or i == len(os.listdir(dataset)) - 1:
                    with open(log_path, 'w') as file:
                        json.dump(log_dict, file, indent=4)