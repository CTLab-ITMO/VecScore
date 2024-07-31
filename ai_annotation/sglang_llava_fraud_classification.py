import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
from PIL import Image


import pandas as pd
from tqdm import tqdm
import glob
import sglang as sgl
from sglang.lang.chat_template import get_chat_template


system_prompt = """
I have a picture and its ratings from the marketer. The marketer's score can be from 1 to 6. Markers can sometimes make mistakes or intentionally put down scores at random in order to finish the markup faster. You can notice this because a beautiful picture has bad marks on the criterion of beauty. Your task is to classify erroneous markup.

After your thorough analysis, provide your output as a single word, which must be one of the following categories:

fraud
data_markup_error
not_sure
safe

Where:
- fraud: intentional data corruption
- data_markup_error: it's just a data markup error
- not_sure: you can't tell if the picture is fairly appreciated or not
- safe: the picture has fair ratings

Here are some examples:

1) image of fallen star 
1. The context is appropriate: 1
2. General impression: 1
3. Aesthetic beauty: 1
4. Precisely drowned: 1
Your output should be: fraud

2) image of fallen star
1. The context is appropriate: 3
2. General impression: 5
3. Aesthetic beauty: 6
4. Precisely drowned: 4
Your output should be: safe

Remember to respond with only one of the four category words, without any additional text or explanation.
"""


@sgl.function
def image_qa(s, image, question):
    s += sgl.user(sgl.image(image) + question)
    s += sgl.assistant(sgl.gen("answer"))


def process_batch(img_paths, answers: list[list[int]], batch_size=2):

    img_prompts = [f"""Classify this:\n
        1. The context is appropriate: {item[0]}\n
        2. General impression: {item[1]}\n
        3. Aesthetic beauty {item[2]}\n
        4. Exactly drown {item[3]}\n""" for item in answers]
    
    states = image_qa.run_batch(
        [
            {"image": path, "question": f"{system_prompt}\n\n{img_prompt}"}
            for path, img_prompt in zip(img_paths, img_prompts)],
        max_new_tokens=100,
    )
    data = [s["answer"] for s in states]
    return data


def get_latest_checkpoint(temp_data_dir):
    checkpoints = glob.glob(os.path.join(temp_data_dir, "*_save.csv"))
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    return latest_checkpoint


if __name__ == "__main__":
    desired_cpu_cores = "60-120"
    pid = os.getpid()
    os.system(f"taskset -p -c {desired_cpu_cores} {pid}")

    runtime = sgl.Runtime(
        model_path="liuhaotian/llava-v1.6-34b",
        tokenizer_path="liuhaotian/llava-v1.6-34b-tokenizer",
    )
    runtime.endpoint.chat_template = get_chat_template("chatml-llava")
    sgl.set_default_backend(runtime)
    
    df = pd.read_csv('/home/jupyter-kazancev.danil7@wb-2ede4/projects/anti_spam/work/VecScore/dataframes/loaded_parsed_toloka_dataset_llava.csv')
    
    img_paths = df.img_path.tolist()
    img_answers = df.result_list.tolist()
    
    temp_data_dir = "dataframes/llava_harm"
    os.makedirs(temp_data_dir, exist_ok=True)
    
    # Check for the latest checkpoint
    latest_checkpoint = get_latest_checkpoint(temp_data_dir)
    
    if latest_checkpoint:
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        checkpoint_df = pd.read_csv(latest_checkpoint)
        start_idx = len(checkpoint_df)
        answers = checkpoint_df['answer_llava'].tolist()
    else:
        print("Starting from the beginning")
        start_idx = 0
        answers = []

    batch_size = 10  # Adjust based on your GPU memory

    for idx in tqdm(range(start_idx, len(img_paths), batch_size)):
        batch_paths = img_paths[idx:idx+batch_size]
        batch_labels = df.iloc[idx:idx+batch_size].result_list.tolist()
        
        batch_answers = process_batch(batch_paths, batch_labels,  batch_size)
   
        log_data = list(zip(batch_answers, batch_labels))

        for data in log_data:
            print(data, '\n')

        answers.extend(batch_answers)

        if (idx + batch_size) % 1000 == 0 or (idx + batch_size >= len(img_paths)):
            print(f"Processed up to {idx + batch_size} images")
            min_df = df.iloc[:idx + batch_size].copy()
            min_df["answer_llava"] = answers
            checkpoint_path = os.path.join(temp_data_dir, f"{idx + batch_size}_save.csv")
            min_df.to_csv(checkpoint_path, index=False)
            print(f"Checkpoint saved: {checkpoint_path}")

    df["answer_llava"] = answers
    df.to_csv("dataframes/llava_harm_answers.csv", index=False)
    print("Processing completed and final results saved.")

    runtime.shutdown()