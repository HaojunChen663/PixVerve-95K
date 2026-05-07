import os
import re
import json
import asyncio
import argparse
import logging
import pandas as pd
from pathlib import Path
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI  # Use AsyncOpenAI for concurrent processing
from util.image_process import MSFI_Preprocessor
from util.build_prompts import build_global_eval_prompt, build_local_eval_prompt

# Logging configurations
def setup_logging(output_dir, dir_name=None):
    if dir_name:
        log_path = os.path.join(output_dir, f"{dir_name}_msfi_eval.log")
    else:
        log_path = os.path.join(output_dir, "msfi_eval.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

# Progress tracker to show task completion status
class ProgressTracker:
    def __init__(self, total):
        self.total = total
        self.current = 0
        self.success = 0
        self.failed = 0
        self.lock = asyncio.Lock()

    async def update(self, is_success=True):
        async with self.lock:
            self.current += 1
            if is_success: self.success += 1
            else: self.failed += 1
            return self.current

# Weights for S_global and S_local calculation
WEIGHTS = {
    "global": {
        "SC-global": 5, 
        "PI": 3, 
        "LC": 2, 
        "CH": 2
    },
    "local": {
        "NGE": 2, 
        "GA": 3, 
        "TF": 5, 
        "MGC": 2, 
        "SC-local": 2
    }
}

# Scorer class to calculate weighted scores
class MSFI_Scorer:
    @staticmethod
    def calculate_weighted_score(scores_dict, weight_map):
        try:
            # Handle potential None or non-dict input
            if not isinstance(scores_dict, dict):
                return 3.0
            s = [float(scores_dict.get(k, 3)) for k in weight_map.keys()]   # Get the score of each sub-dimension, default to 3
            w = list(weight_map.values())
            return sum(si * wi for si, wi in zip(s, w)) / sum(w)
        except Exception:
            return 3.0

# Extract JSON content from the model's response using <json> tags or fallback methods
def extract_json_from_response(text):
    try:
        tag_match = re.search(r'<json>(.*?)</json>', text, re.DOTALL)
        content = tag_match.group(1).strip() if tag_match else text
        
        content = content.replace("```json", "").replace("```", "").strip()
        
        return json.loads(content)
    except Exception:
        try:
            brace_match = re.search(r'(\{.*\})', text, re.DOTALL)
            if brace_match:
                return json.loads(brace_match.group(1))
        except:
            return None
    return None

# Async function to wrap an MLLM request for global-scale or local-scale evaluation
async def mllm_inference(client, model_path, prompt, image_b64_list, semaphore):
    async with semaphore:
        try:
            content = [{"type": "text", "text": prompt}]
            for b64 in image_b64_list:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            
            response = await client.chat.completions.create(
                model=model_path,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=8192,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
            raw_text = response.choices[0].message.content
            return extract_json_from_response(raw_text)
        except Exception as e:
            return None

# Process a single generated image including pre-processing, MLLM inference (global-scale and local-scale evaluations), and results saving
async def process_single_image(img_path, client, model_path, preprocessor, semaphore, tracker, paths):
    file_name = os.path.basename(img_path)
    try:
        # Use util.image_process.MSFI_Preprocessor to preprocess the image and get the global-scale and local-scale evaluation inputs
        data = preprocessor.process_image(img_path)
        if data is None:
            raise ValueError("Image preprocessing failed")
        
        # Construct the global-scale evaluation task
        g_prompt = build_global_eval_prompt()
        g_res = await mllm_inference(client, model_path, g_prompt, [data['global_scale']['image_b64']], semaphore)
        
        if g_res is None:
            logging.error(f"❌ Global-scale fidelity evaluation failed for {file_name}")
        
        # Construct and run local-scale evaluation tasks for all 10 patches
        l_tasks = []
        for patch in data['local_scale']:
            l_prompt = build_local_eval_prompt(patch['location_info']['relative_coords'])
            l_tasks.append(mllm_inference(client, model_path, l_prompt, [patch['patch_b64'], data['global_scale']['image_b64']], semaphore))    # Image 1: local patch; Image 2: global image
        
        l_raw_results = await asyncio.gather(*l_tasks)
        
        # Filter out None results (Failed patches)
        l_res_list = []
        for i, res in enumerate(l_raw_results):
            if res is None:
                logging.warning(f"⚠️ Patch {i+1} evaluation failed for {file_name}")
            else:
                l_res_list.append(res)

        # Save to the corresponding JSONL immediately
        if g_res:
            with open(paths['global_jsonl'], 'a', encoding='utf-8') as f:
                f.write(json.dumps({"file": file_name, "results": g_res}, ensure_ascii=False) + '\n')
        
        if l_res_list:
            with open(paths['local_jsonl'], 'a', encoding='utf-8') as f:
                f.write(json.dumps({"file": file_name, "patches": l_res_list}, ensure_ascii=False) + '\n')

        success = (g_res is not None) and (len(l_res_list) == 10)
        await tracker.update(success)
        return success

    except Exception as e:
        logging.error(f"❌ Critical error processing {file_name}: {str(e)}")
        await tracker.update(False)
        return False

# Main
async def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_dir = Path(args.gen_dir)

    logger = setup_logging(args.output_dir, dir_name=img_dir.name)
    logger.info(f"Starting MSFI Evaluation with model: {args.model_path}")

    # Initialize OpenAI client, preprocessor, and concurrency semaphore
    client = AsyncOpenAI(api_key="EMPTY", base_url=args.api_url)
    preprocessor = MSFI_Preprocessor()
    semaphore = asyncio.Semaphore(args.concurrency)
    
    img_paths = [str(p) for p in img_dir.glob("*") if p.suffix.lower() in ['.png', '.jpg', '.jpeg']]    # Images to evaluate (should be 200)
    tracker = ProgressTracker(len(img_paths))
    
    paths = {
        "global_jsonl": output_dir / f"{img_dir.name}_global_eval.jsonl",
        "local_jsonl": output_dir / f"{img_dir.name}_local_eval.jsonl",
        "csv": output_dir / f"{img_dir.name}_MSFI.csv"
    }

    # Begin evaluation
    tasks = [process_single_image(p, client, args.model_path, preprocessor, semaphore, tracker, paths) for p in img_paths]
    await tqdm.gather(*tasks, desc="MSFI Evaluating")
    
    # Scoring Phase: Load from JSONL for final CSV generation
    logger.info("All evaluations completed. Starting to generate final scores...")
    
    # Load global evaluation results into a dict {filename: data}
    global_data_map = {}
    if paths['global_jsonl'].exists():
        with open(paths['global_jsonl'], 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                global_data_map[item['file']] = item['results']

    # Load local evaluation results into a dict {filename: patch_list}
    local_data_map = {}
    if paths['local_jsonl'].exists():
        with open(paths['local_jsonl'], 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                local_data_map[item['file']] = item['patches']

    final_rows = []
    scorer = MSFI_Scorer()

    # Iterate through images that have both global and local evaluation results
    for file_name in global_data_map.keys():
        if file_name not in local_data_map:
            logger.warning(f"⚠️ No local evaluation results for {file_name}.")
            continue
            
        g_scores = global_data_map[file_name]
        l_patches_scores = local_data_map[file_name] # A list of successful patches' scores
        
        # Calculate S_global
        s_g = scorer.calculate_weighted_score(g_scores, WEIGHTS['global'])
        
        # Calculate S_local
        num_patches = len(l_patches_scores)
        patch_weighted_scores = []
        l_sub_avg = {k: 0.0 for k in WEIGHTS['local'].keys()} 
        
        for p_res in l_patches_scores:
            patch_weighted_scores.append(scorer.calculate_weighted_score(p_res, WEIGHTS['local']))
            for k in l_sub_avg:
                l_sub_avg[k] += float(p_res.get(k, 3)) / num_patches
            
        s_l = sum(patch_weighted_scores) / num_patches
        msfi = s_g + (s_g / 5) * s_l
        
        # Construct the row content
        row = {
            "File_Name": file_name,
            "MSFI": round(msfi, 4),
            "S_global": round(s_g, 4),
            "S_local": round(s_l, 4),
            **{f"{k}": float(g_scores.get(k, 3)) for k in WEIGHTS['global'].keys()},
            **{f"{k}_avg": round(l_sub_avg[k], 4) for k in WEIGHTS['local'].keys()},
            "Valid_Patches": num_patches
        }
        final_rows.append(row)

    # Final summary and display
    if final_rows:
        df = pd.DataFrame(final_rows)
        df.to_csv(paths['csv'], index=False)
        summary_means = df.mean(numeric_only=True)
        
        logger.info("-" * 50)
        logger.info(f"📊 The evaluation results of all images are saved to {paths['csv']}")
        
        print(f"\nMSFI Summary:")
        print(f"MSFI: {summary_means['MSFI']:.4f}")
        print(f"S_global: {summary_means['S_global']:.4f} | S_local: {summary_means['S_local']:.4f}")
        print("-" * 30)
        print("🔍 Global-scale Fidelity Metrics:")
        for k in WEIGHTS['global'].keys():
            print(f"   - {k}: {summary_means[f'{k}']:.4f}")
        print("\n🔍 Local-scale Fidelity Metrics (Avg across patches):")
        for k in WEIGHTS['local'].keys():
            print(f"   - {k}: {summary_means[f'{k}_avg']:.4f}")
    else:
        logger.error("❌ No valid evaluation results found to generate CSV.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UHR MSFI Evaluation Pipeline")

    parser.add_argument("--gen_dir", type=str, required=True, help="Path to the directory containing your generated images")
    parser.add_argument("--output_dir", type=str, default="./MSFI_evaluation", help="Directory for logs and evaluation results")
    parser.add_argument("--model_path", type=str, required=True, help="Path to your local Qwen3.5-35B-A3B model for evaluation")
    parser.add_argument("--api_url", type=str, default="http://127.0.0.1:8000/v1", help="vLLM API URL")
    parser.add_argument("--concurrency", type=int, default=11, help="Parallel API requests (a single image requires 11 API calls: 1 global-scale evaluation + 10 local-scale evaluations)")
    
    args = parser.parse_args()
    asyncio.run(main(args))