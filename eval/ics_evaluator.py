import os
import re
import json
import math
import asyncio
import argparse
import logging
import base64
import io
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI  # Use AsyncOpenAI for concurrent processing
from util.build_prompts import build_ICS_eval_prompt

# Logging configurations
def setup_logging(output_dir, dir_name=None):
    if dir_name:
        log_path = os.path.join(output_dir, f"{dir_name}_ics_eval.log")
    else:
        log_path = os.path.join(output_dir, "ics_eval.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

# Resize images and convert to base64 data URL for API input
def preprocess_image(image_path, target_long_edge=3072):
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            w, h = img.size
            if max(w, h) > target_long_edge:
                scale = target_long_edge / max(w, h)
                new_size = (int(w * scale), int(h * scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=98)   # High-quality saving to memory
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return img_str
    except Exception as e:
        logging.error(f"Error preprocessing image {image_path}: {e}")
        return None

# Extract JSON content from the model's response using <json> tags or fallback methods
def extract_json_from_response(text):
    try:
        tag_match = re.search(r'<json>(.*?)</json>', text, re.DOTALL)
        content = tag_match.group(1).strip() if tag_match else text
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except:
        try:
            brace_match = re.search(r'(\{.*\})', text, re.DOTALL)
            if brace_match:
                return json.loads(brace_match.group(1))
        except:
            return None
    return None

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

# Process a single generated image including pre-processing, MLLM inference, and results saving
async def process_single_image(img_info, client, args, semaphore, tracker, output_file):
    img_path = img_info['path']
    long_caption = img_info['long_caption']
    file_name = os.path.basename(img_path)

    async with semaphore:
        try:
            b64_data = preprocess_image(img_path)
            if not b64_data:
                raise ValueError(f"Preprocessing failed for {file_name}")

            # Build prompt for MLLM evaluation
            prompt = build_ICS_eval_prompt(long_caption)

            # MLLM request
            response = await client.chat.completions.create(
                model=args.model_path,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}}
                    ]
                }],
                temperature=0.1,
                max_tokens=8192,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )

            raw_text = response.choices[0].message.content
            res_json = extract_json_from_response(raw_text)

            if res_json and all(k in res_json for k in ["IEV", "AAA", "SRA"]):
                iev, aaa, sra = float(res_json["IEV"]), float(res_json["AAA"]), float(res_json["SRA"])
                ics_score = math.sqrt(iev / 10.0) * (0.6 * aaa + 0.4 * sra)  # Final ICS calculation based on the scoring formula
                
                result = {
                    "file_name": file_name,
                    "ics_score": round(ics_score, 4),
                    "IEV": iev,
                    "AAA": aaa,
                    "SRA": sra,
                    "reasoning": res_json.get("reasoning", "")
                }

                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
                await tracker.update(True)
                return {"ics": ics_score, "iev": iev, "aaa": aaa, "sra": sra}
            else:
                raise ValueError("Incomplete JSON response")

        except Exception as e:
            logging.error(f"❌ Failed to process {file_name}: {str(e)}")
            await tracker.update(False)
            return None

# Main Pipeline
async def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gen_dir = Path(args.gen_dir)

    logger = setup_logging(args.output_dir, dir_name=gen_dir.name)

    # Initialize OpenAI client and concurrency semaphore
    client = AsyncOpenAI(api_key="EMPTY", base_url=args.api_url)
    semaphore = asyncio.Semaphore(args.concurrency)

    # Load metadata and find matching files
    logger.info(f"Loading benchmark jsonl from: {args.benchmark_jsonl}")
    metadata = {}
    with open(args.benchmark_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            # Use basename without extension as key
            basename = os.path.splitext(item['file_name'])[0]
            metadata[basename] = item['long_caption']

    tasks_info = []
    for img_p in gen_dir.glob("*"):
        if img_p.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            basename = img_p.stem
            if basename in metadata:
                tasks_info.append({
                    "path": str(img_p),
                    "long_caption": metadata[basename]
                })

    if not tasks_info:
        logger.error("No matching images found between benchmark.jsonl and gen_dir.")
        return

    logger.info(f"Total images to evaluate: {len(tasks_info)}")
    tracker = ProgressTracker(len(tasks_info))
    output_jsonl = os.path.join(args.output_dir, f"{gen_dir.name}_ics_eval.jsonl")

    # Begin evaluation with concurrent processing
    tasks = [process_single_image(info, client, args, semaphore, tracker, output_jsonl) for info in tasks_info]
    all_results = await tqdm.gather(*tasks, desc="ICS Evaluating")

    # Score aggregation and final reporting
    logger.info("-" * 50)
    logger.info(f"✅ Evaluation completed. Success: {tracker.success}/{tracker.total}")
    logger.info(f"Results saved to {output_jsonl}")

    valid_results = [r for r in all_results if r is not None]
    
    if valid_results:
        print(f"\n Valid ICS Scores: {len(valid_results)} / {tracker.total}")
        avg_ics = sum(r['ics'] for r in valid_results) / len(valid_results)
        avg_iev = sum(r['iev'] for r in valid_results) / len(valid_results)
        avg_aaa = sum(r['aaa'] for r in valid_results) / len(valid_results)
        avg_sra = sum(r['sra'] for r in valid_results) / len(valid_results)
        print(f"\n{'='*30}")
        print(f"ICS for {gen_dir.name}: {avg_ics:.4f}")
        print(f"\nIEV for {gen_dir.name}: {avg_iev:.4f}")
        print(f"AAA for {gen_dir.name}: {avg_aaa:.4f}")
        print(f"SRA for {gen_dir.name}: {avg_sra:.4f}")
        print(f"{'='*30}\n")
    else:
        logger.error("No valid scores were obtained.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Instance-centric Compliance Score (ICS) Evaluator")
    parser.add_argument("--benchmark_jsonl", type=str, default="/z_datasets/PixVerve-95K/PixVerve-95K-metadata/benchmark.jsonl")
    parser.add_argument("--gen_dir", type=str, required=True, help="Path to the directory containing your generated images")
    parser.add_argument("--output_dir", type=str, default="./ICS_evaluation", help="Directory to save evaluation results and logs")
    parser.add_argument("--model_path", type=str, required=True, help="Path to your local Qwen3.5-35B-A3B model for evaluation")
    parser.add_argument("--api_url", type=str, default="http://127.0.0.1:8000/v1", help="vLLM API URL")
    parser.add_argument("--concurrency", type=int, default=8, help="Parallel API requests")

    args = parser.parse_args()
    asyncio.run(main(args))