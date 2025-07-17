import torch
import time
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration
import warnings
warnings.filterwarnings("ignore")

class FastImageCaptioner:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        """
        Fast image captioning model optimized for RTX 4080
        
        Models to try (ordered by speed vs quality):
        - "Salesforce/blip-image-captioning-base" (fastest, good quality)
        - "Salesforce/blip-image-captioning-large" (slower, better quality)
        - "microsoft/git-base" (very fast, decent quality)
        - "nlpconnect/vit-gpt2-image-captioning" (fastest, basic quality)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Loading Image Captioner on {self.device}")
        
        # Setup GPU optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            gpu_name = torch.cuda.get_device_name()
            print(f"🎮 GPU: {gpu_name}")
        
        # Load model and processor
        print(f"📥 Loading {model_name}...")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use FP16 for speed
            low_cpu_mem_usage=True,
        ).to(self.device)
        
        self.model.eval()
        
        # Warmup
        print("🔥 Warming up...")
        self.warmup()
        print("✅ Ready for captioning!")
    
    def warmup(self, num_runs=3):
        """Warm up the model"""
        dummy_image = Image.new('RGB', (224, 224), color='red')
        
        with torch.no_grad():
            for _ in range(num_runs):
                inputs = self.processor(dummy_image, return_tensors="pt").to(self.device, torch.float16)
                _ = self.model.generate(**inputs, max_length=50, num_beams=1, do_sample=False)
        
        torch.cuda.empty_cache()
    
    def caption_single_image(self, image_path, max_length=50, num_beams=1):
        """
        Generate caption for single image
        
        Args:
            image_path: Path to image or PIL Image or URL
            max_length: Maximum caption length (shorter = faster)
            num_beams: Beam search (1 = fastest, 5 = better quality)
        """
        # Load image
        if isinstance(image_path, str):
            if image_path.startswith('http'):
                response = requests.get(image_path)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        # Process image
        inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)
        
        # Generate caption with timing
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=False,
                early_stopping=True
            )
        
        torch.cuda.synchronize()
        inference_time = (time.perf_counter() - start_time) * 1000
        
        # Decode caption
        caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        
        return {
            'caption': caption,
            'inference_time_ms': inference_time,
            'image': image
        }
    
    def caption_batch(self, image_list, batch_size=16, max_length=50, num_beams=1):
        """
        Batch caption generation for maximum throughput
        
        Note: Smaller batch_size for captioning due to memory usage
        """
        print(f"🔄 Captioning {len(image_list)} images in batches of {batch_size}")
        
        all_results = []
        total_time = 0
        
        for i in range(0, len(image_list), batch_size):
            batch_paths = image_list[i:i + batch_size]
            
            # Load batch images
            images = []
            for img_path in batch_paths:
                if isinstance(img_path, str):
                    if img_path.startswith('http'):
                        response = requests.get(img_path)
                        img = Image.open(BytesIO(response.content)).convert('RGB')
                    else:
                        img = Image.open(img_path).convert('RGB')
                else:
                    img = img_path
                images.append(img)
            
            # Process batch
            inputs = self.processor(images, return_tensors="pt", padding=True).to(self.device, torch.float16)
            
            # Batch inference
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    do_sample=False,
                    early_stopping=True
                )
            
            torch.cuda.synchronize()
            batch_time = (time.perf_counter() - start_time) * 1000
            total_time += batch_time
            
            # Decode captions
            captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            for j, caption in enumerate(captions):
                all_results.append({
                    'image_path': batch_paths[j],
                    'caption': caption,
                    'batch_time_ms': batch_time / len(batch_paths)
                })
            
            print(f"   Batch {i//batch_size + 1}: {batch_time/len(batch_paths):.2f}ms per image")
        
        avg_time = total_time / len(image_list)
        
        # Print results summary
        print(f"\n📝 Captioning Results:")
        for i, result in enumerate(all_results):
            image_name = result['image_path'].split('\\')[-1] if isinstance(result['image_path'], str) else f"Image {i+1}"
            print(f"   {i+1}. {image_name}: '{result['caption']}'")
        
        print(f"\n📊 Performance Summary:")
        print(f"   Average per image: {avg_time:.2f}ms")
        print(f"   Throughput: {1000/avg_time:.1f} captions/second")
        
        return {
            'results': all_results,
            'avg_time_per_image_ms': avg_time,
            'throughput_fps': 1000 / avg_time
        }
    
    def benchmark_captioning(self, num_runs=20):
        """Benchmark captioning speed"""
        print(f"🏁 Benchmarking captioning speed ({num_runs} runs)")
        
        test_image = Image.new('RGB', (224, 224), color='blue')
        times = []
        
        for i in range(num_runs):
            result = self.caption_single_image(test_image, max_length=30, num_beams=1)
            times.append(result['inference_time_ms'])
            
            if i % 5 == 0:
                print(f"   Run {i}: {result['inference_time_ms']:.2f}ms - '{result['caption']}'")
        
        times = np.array(times)
        print(f"\n📊 Captioning Performance:")
        print(f"   Mean: {np.mean(times):.2f}ms ± {np.std(times):.2f}ms")
        print(f"   Min: {np.min(times):.2f}ms | Max: {np.max(times):.2f}ms")
        print(f"   Throughput: {1000/np.mean(times):.1f} captions/second")
        
        return times

    def show_batch_results(self, results):
        """Helper function to display batch results nicely"""
        print(f"\n📝 Batch Captioning Results:")
        print("=" * 60)
        
        for i, result in enumerate(results['results'], 1):
            image_name = result['image_path'].split('\\')[-1] if isinstance(result['image_path'], str) else f"Image {i}"
            print(f"{i:2d}. {image_name}")
            print(f"    Caption: '{result['caption']}'")
            print(f"    Time: {result['batch_time_ms']:.1f}ms")
            print()
        
        print(f"📊 Summary: {len(results['results'])} images")
        print(f"   Average: {results['avg_time_per_image_ms']:.1f}ms per image")
        print(f"   Throughput: {results['throughput_fps']:.1f} captions/second")