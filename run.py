from datasets import Dataset
import gc
import huggingface_hub
import pandas as pd
import time
import tqdm
# from vllm import LLM, SamplingParams
from google.genai import types, Client
from transformers import AutoTokenizer
import torch
from Logger import Logger
import os
from dotenv import load_dotenv


def vllm_run(model_id, df):
    log = Logger()
    # dataset = Dataset.from_pandas(df)
    # model_id = "google/gemma-3-12b-it"
    #
    # # vllm 실행
    # # vllm 3.12 버전 지원 안하니까 유의하시길.
    # llm = LLM(
    #     model=model_id,
    #     trust_remote_code=True,
    #     quantization="bitsandbytes",
    #     load_format="bitsandbytes",
    #     gpu_memory_utilization=0.7,
    #     max_model_len=4096
    # )
    #
    # torch.cuda.empty_cache()
    # gc.collect()
    #
    # # 토크나이저 로드
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # # EOS 토큰 ID 가져오기
    # eos_token_id = tokenizer.eos_token_id
    #
    # logging.info(f'배치 사이즈 : {llm.llm_engine.scheduler_config.max_num_seqs}')
    #
    # result_text = []
    #
    # start_time = time.time()
    # logging.info(f'시작 시간 : {start_time}')
    #
    # for i, test in enumerate(tqdm.tqdm(dataset)):
    #
    #     # 출력 형태를 반드시 확인하고 진행해주세요.
    #     # if i>= 5 :
    #     #     break
    #
    #     text = eval(test['related_documents'])
    #     sampling_params = SamplingParams(temperature=0.2,
    #                                      seed=42,
    #                                      max_tokens=512,
    #                                      top_p=0.8,
    #                                      stop_token_ids=[eos_token_id])
    #     outputs = llm.chat(text,
    #                        sampling_params)
    #
    #     for output in outputs:
    #         prompt = output.prompt
    #         generated_text = output.outputs[0].text
    #         logging.info(generated_text)
    #         result_text.append(generated_text)
    #
    # end_time = time.time()
    # logging.info(f'종료 시간 : {end_time}')
    # logging.info(f'총 걸린 시간 : {end_time-start_time}')

def run_with_api(df):
    # 로그 출력
    log = Logger()

    load_dotenv()

    # 키 불러오기
    load_dotenv()
    api_key = os.getenv('API_KEY')
    dataset = Dataset.from_pandas(df)

    client = Client(api_key=api_key)

    start_time = time.time()
    log.info(f'시작 시간 : {start_time}')
    #
    for i, test in enumerate(tqdm.tqdm(dataset)):

        # 출력 형태를 반드시 확인하고 진행해주세요.
        if i>= 5 :
            break

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=test['related_documents'],
            config=types.GenerateContentConfig(
                max_output_tokens=512,
                temperature=0.1,
                topP = 0.98,
            )
        )

        generated_text = response.text

        print(f'{generated_text}')

    end_time = time.time()
    log.info(f'종료 시간 : {end_time}')
    log.info(print(f'총 걸린 시간 : {end_time-start_time}'))



if __name__ == '__main__':

    test_data = pd.read_csv('./data/combined_test_data.csv', encoding='utf-8-sig')
    run_with_api(test_data)
