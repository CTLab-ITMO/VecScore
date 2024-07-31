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
You will receive an image, and your task is to evaluate it based on the following criteria, using a scale from 1 to 6, where 1 is the lowest score and 6 is the highest:

1. Context appropriateness
2. General impression
3. Aesthetic beauty
4. Precision of execution

After evaluating, provide your scores in the following format:

output_format: [1..6, 1..6, 1..6, 1..6]

Example: 
Input: image of a fallen star.
output: [1, 3, 5, 6]

Please provide only the scores, without any additional text or explanations.
"""

img_prompt = """Classify this image using my 4 criterias. Don't forget about output format."""

@sgl.function
def image_qa(s, image, question):
    s += sgl.user(sgl.image(image) + question)
    s += sgl.assistant(sgl.gen("answer"))


def process_batch(img_paths, answers: list[list[int]], batch_size=2):

    img_prompts = [img_prompt for item in answers]
    
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
    df = df.drop_duplicates('img_path')
    
    img_paths = df.img_path.tolist()
    img_answers = df.result_list.tolist()
    
    temp_data_dir = "dataframes/llava_ranking"
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
    df.to_csv("dataframes/llava_ranking_images.csv", index=False)
    print("Processing completed and final results saved.")

    runtime.shutdown()