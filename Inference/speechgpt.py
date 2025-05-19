import os
import sys
sys.path.append('/path/to/SpeechGPT_project')
os.chdir('/path/to/SpeechGPT_project')

import csv
from pprint import pprint
from itertools import count

import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
import soundfile as sf
from typing import List
import argparse
import logging
import json
from tqdm import tqdm
import re
from peft import PeftModel
from utils.speech2unit.speech2unit import Speech2Unit
from transformers import AutoConfig, LlamaForCausalLM, LlamaTokenizer, GenerationConfig


logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


NAME="SpeechGPT"
META_INSTRUCTION="You are an AI assistant whose name is SpeechGPT.\n- SpeechGPT is a intrinsic cross-modal conversational language model that is developed by Fudan University.  SpeechGPT can understand and communicate fluently with human through speech or text chosen by the user.\n- It can perceive cross-modal inputs and generate cross-modal outputs.\n"
DEFAULT_GEN_PARAMS = {
        "max_new_tokens": 1024,
        "min_new_tokens": 10,
        "temperature": 0.8,
        "do_sample": True, 
        "top_k": 60,
        "top_p": 0.8,
        }  
device = torch.device('cuda')


def extract_text_between_tags(text, tag1='[SpeechGPT] :', tag2='<eoa>'):
    pattern = f'{re.escape(tag1)}(.*?){re.escape(tag2)}'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        response = match.group(1)
    else:
        response = ""
    return response


class SpeechGPTInference:
    def __init__(
        self, 
        model_name_or_path: str,
        lora_weights: str=None,
        s2u_dir: str="speechgpt/utils/speech2unit/",
        vocoder_dir: str="speechgpt/utils/vocoder/", 
        output_dir="speechgpt/output/"
        ):
        
        self.meta_instruction = META_INSTRUCTION
        self.template= "[Human]: {question} <eoh>. [SpeechGPT]: "


        #speech2unit
        self.s2u = Speech2Unit(ckpt_dir=s2u_dir)
        
        #model
        self.model = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
            )

        if lora_weights is not None:
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_weights,
                torch_dtype=torch.float16,
                device_map="auto",
            )

        self.model.half()  

        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        #tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path)
        self.tokenizer.pad_token_id = (0)
        self.tokenizer.padding_side = "left" 


        #generation
        self.generate_kwargs = DEFAULT_GEN_PARAMS


        #vocoder
        vocoder = os.path.join(vocoder_dir, "vocoder.pt")
        vocoder_cfg = os.path.join(vocoder_dir, "config.json")
        with open(vocoder_cfg) as f:
            vocoder_cfg = json.load(f)
        self.vocoder = CodeHiFiGANVocoder(vocoder, vocoder_cfg).to(device)

        self.output_dir = output_dir


    def preprocess(
        self,
        raw_text: str,
    ):
        processed_parts = []
        for part in raw_text.split("is input:"):
            if os.path.isfile(part.strip()) and os.path.splitext(part.strip())[-1] in [".wav", ".flac", ".mp4"]:
                processed_parts.append(self.s2u(part.strip(), merged=True))
            else:
                processed_parts.append(part)
        processed_text = "is input:".join(processed_parts)

        prompt_seq = self.meta_instruction + self.template.format(question=processed_text)
        return prompt_seq


    def postprocess(
        self,
        response: str,
    ):

        question = extract_text_between_tags(response, tag1="[Human]", tag2="<eoh>")
        answer = extract_text_between_tags(response + '<eoa>', tag1=f"[SpeechGPT] :", tag2="<eoa>")
        tq = extract_text_between_tags(response, tag1="[SpeechGPT] :", tag2="; [ta]") if "[ta]" in response else ''
        ta = extract_text_between_tags(response, tag1="[ta]", tag2="; [ua]") if "[ta]" in response else ''
        ua = extract_text_between_tags(response + '<eoa>', tag1="[ua]", tag2="<eoa>") if "[ua]" in response else ''

        return {"question":question, "answer":answer, "textQuestion":tq, "textAnswer":ta, "unitAnswer":ua}


    def forward(
        self, 
        prompts: List[str]
    ):
        with torch.no_grad():
            #preprocess
            preprocessed_prompts = []
            for prompt in prompts:
                preprocessed_prompts.append(self.preprocess(prompt))

            input_ids = self.tokenizer(preprocessed_prompts, return_tensors="pt", padding=True).input_ids
            for input_id in input_ids:
                if input_id[-1] == 2:
                    input_id = input_id[:, :-1]

            input_ids = input_ids.to(device)

            #generate
            generation_config = GenerationConfig(
                temperature=0.7,
                top_p=0.8,
                top_k=50,
                do_sample=True, 
                max_new_tokens=2048,
                min_new_tokens=10,
                )

            generated_ids = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                # max_new_tokens=1024,
            )
            generated_ids = generated_ids.sequences
            responses = self.tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)

            #postprocess
            responses = [self.postprocess(x) for x in responses]

            #save repsonses
            # init_num = sum(1 for line in open(f"{self.output_dir}/responses.json", 'r')) if os.path.exists(f"{self.output_dir}/responses.json") else 0             
            # with open(f"{self.output_dir}/responses.json", 'a') as f:
            #     for r in responses:
            #         if r["textAnswer"] != "":
            #             print("Transcript:", r["textQuestion"])
            #             print("Text response:", r["textAnswer"])
            #         else:
            #             print("Response:\n", r["answer"])
            #         json_line = json.dumps(r)
            #         f.write(json_line+'\n')

        return responses[0]["textAnswer"]

    def dump_wav(self, sample_id, pred_wav, prefix):
        sf.write(
            f"{self.output_dir}/wav/{prefix}_{sample_id}.wav",
            pred_wav.detach().cpu().numpy(),
            16000,
        )
        
    def __call__(self, input):
        return self.forward(input)

    
    def interact(self):
        model_name = "SpeechGPT"
        log_folder = f'../../audio_jailbreak/filtered_Advbench_with_defense_logs/{model_name}'
        attack_types = ['audio-text', 'audio-only', 'text-only']
        attack_types = [attack_types[0]]

        datasets = []
        for dirpath, dirnames, filenames in os.walk("../../audio_jailbreak/datasets/filtered_Advbench_audio_with_defense"):
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
                for i, wav in enumerate(tqdm(os.listdir(dataset))):

                    if attack_type == 'audio-text':
                        prompt = "Provide a detailed answer that exceeds 500 words."
                        audio = os.path.join(dataset, wav)

                        response = self.forward([audio])

                        item = {
                            "type": attack_type,
                            "audio": audio,
                            'prompt': prompt,
                            'response': response,
                            'word count': len(response.split())
                        }
                        key = str(next(key_generator))
                        log_dict[key] = item
                        pprint(item)

                    if i % 10 == 0 or i == len(os.listdir(dataset)) - 1:
                        with open(log_path, 'w') as file:
                            json.dump(log_dict, file, indent=4)

            
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default="models/SpeechGPT-7B-cm")
    parser.add_argument("--lora-weights", type=str, default="models/SpeechGPT-7B-com")
    parser.add_argument("--s2u-dir", type=str, default="utils/speech2unit/")
    parser.add_argument("--vocoder-dir", type=str, default="utils/vocoder/")
    parser.add_argument("--output-dir", type=str, default="output/")
    args = parser.parse_args()

    infer = SpeechGPTInference(
        args.model_name_or_path,
        args.lora_weights,
        args.s2u_dir,
        args.vocoder_dir,
        args.output_dir
    )

    infer.interact()