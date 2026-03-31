import torch
import torch.nn.functional as F
import numpy as np
import cv2
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
import copy

# ==========================================
# 1. SDXL Attention Hook
# ==========================================

class WinWinLayAttnProcessorSDXL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_maps = None 

    # SDXL 的 Attention 呼叫簽名與 SD 1.5 略有不同
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        
        # SDXL 的 Cross Attention
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        # 儲存 Attention Maps
        # SDXL 的 Cross Attention 通常發生在 encoder_hidden_states.shape[1] == 77 (CLIP 1) 或 1280 (OpenCLIP 投影後) 
        # 但通常我們關注 Prompt token 長度，SDXL 預設 prompt length 也是 77
        if encoder_hidden_states.shape[1] == 77: 
            self.attention_maps = attention_probs
            
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

class WinWinLayPipelineSDXL:
    def __init__(self, device="cuda"):
        self.device = device
        print("正在載入 SDXL Base 1.0 模型...")
        
        # 為了省顯存，載入時用 FP16，但梯度計算時我們會小心處理
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16, 
            variant="fp16",
            use_safetensors=True
        ).to(device)
        
        self.pipe.vae.to(dtype=torch.float32) # Fix black image issue
        # 使用 DDIM Scheduler
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        # Hook Attention Processors
        # SDXL 的結構比較深，我們只 Hook "attn2" (Cross-Attention)
        self.hooked_processors = {}
        for name, _ in self.pipe.unet.attn_processors.items():
            if "attn2" in name: 
                self.hooked_processors[name] = WinWinLayAttnProcessorSDXL()
            else:
                # 保持原本的高效 Attention (SDPA)
                self.hooked_processors[name] = AttnProcessor2_0()
                
        self.pipe.unet.set_attn_processor(self.hooked_processors)
        print("SDXL 模型載入完成。")

    # SDXL 需要特殊的 Prompt Encoding (雙 Encoder)
    def encode_prompts(self, prompt):
        # 取得 Text Embeddings (Prompt Embeds & Pooled Embeds)
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            do_classifier_free_guidance=True
        )
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    # SDXL 需要 Time IDs (Micro-conditioning)
    def get_add_time_ids(self, height, width):
        target_size = (height, width)
        original_size = (height, width) # 假設沒有裁切
        crops_coords_top_left = (0, 0)
        
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=self.pipe.unet.dtype).to(self.device)
        return torch.cat([add_time_ids, add_time_ids], dim=0)

    def decode_latents_to_pil(self, latents):
        latents = latents.to(dtype=torch.float32) # 關鍵
        image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
                
        return self.pipe.image_processor.postprocess(image, output_type="pil")[0]

    # 取得 Token Indices (針對 SDXL 的第二個 Tokenizer，因為它主導語義)
    def get_token_indices(self, prompt, subject_words):
        # SDXL 使用 tokenizer_2 (OpenCLIP)
        tokenizer = self.pipe.tokenizer_2
        tokens = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()

        def find_subseq_positions(seq, subseq):
            positions = []
            if len(subseq) == 0 or len(subseq) > len(seq): return positions
            for i in range(len(seq) - len(subseq) + 1):
                if seq[i:i+len(subseq)] == subseq: positions.append(i)
            return positions

        indices = []
        for word in subject_words:
            word_tokens = tokenizer(word, add_special_tokens=False)["input_ids"]
            if len(word_tokens) == 0: 
                indices.append([])
                continue
            
            starts = find_subseq_positions(tokens, word_tokens)
            if not starts: 
                indices.append([])
                continue
                
            all_idxs = []
            for s in starts: all_idxs.extend(list(range(s, s + len(word_tokens))))
            indices.append(all_idxs)
        return indices

    def compute_gaussian_prior(self, width, height, bbox, sigma_scale=0.35):
        x_min, y_min, w_box, h_box = bbox
        cx, cy = x_min + w_box / 2, y_min + h_box / 2
        
        x = torch.arange(width, device=self.device).float()
        y = torch.arange(height, device=self.device).float()
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        sigma_x = max(w_box * sigma_scale, 1e-6)
        sigma_y = max(h_box * sigma_scale, 1e-6)
        
        gaussian = torch.exp(-((xx - cx)**2 / (2 * sigma_x**2) + (yy - cy)**2 / (2 * sigma_y**2)))
        return gaussian

    def generate(self, prompt, bboxes, subjects, rho_list, method="WinWinLay", steps=50, guidance_scale=7.5, seed=42):
        generator = torch.Generator(self.device).manual_seed(seed)
        # SDXL 標準解析度
        height, width = 1024, 1024 

        # 1. Encode Prompts (SDXL 特有)
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompts(prompt)
        
        # 為了 Classifier-Free Guidance (CFG)，我們 concat embedding
        # 注意：這裡我們先保持 FP16，等進 UNet 再看情況
        prompt_embeds_all = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        
        # Prepare Time IDs
        add_time_ids = self.get_add_time_ids(height, width)
        
        # 整理成 SDXL UNet 需要的 added_cond_kwargs
        added_cond_kwargs = {
            "text_embeds": add_text_embeds,
            "time_ids": add_time_ids
        }

        # 取得 Token Indices
        obj_token_indices = self.get_token_indices(prompt, subjects)

        # 2. Init Latents (SDXL VAE downscale factor is 8 -> 1024/8 = 128)
        # 全程使用 FP32 初始化
        latents = torch.randn((1, 4, 128, 128), generator=generator, device=self.device, dtype=torch.float32)
        self.pipe.scheduler.set_timesteps(steps)
        latents = latents * self.pipe.scheduler.init_noise_sigma

        final_attn_maps = {} 
        step_images = []

        print(f"開始 SDXL 生成... 模式: {method}")

        # === 論文參數設定 ===
        early_steps = 10 
        prior_decay_steps = 10 
        langevin_steps_O = 4 
        signal_to_noise_r = 0.06 

        for i, t in enumerate(self.pipe.scheduler.timesteps):
            
            # --- WinWinLay / Backward Guidance Phase ---
            if i < early_steps and len(bboxes) > 0 and "WinWinLay" in method:
                
                decay_factor = max(0.0, 1.0 - (float(i) / float(prior_decay_steps)))
                
                for k in range(langevin_steps_O):
                    with torch.enable_grad():
                        latents = latents.detach().requires_grad_(True)
                        
                        # Clear maps
                        for name, processor in self.pipe.unet.attn_processors.items():
                            if hasattr(processor, "attention_maps"): processor.attention_maps = None

                        # Forward pass
                        # 注意：這裡輸入的 latents 是 FP32
                        # 但 SDXL UNet 權重是 FP16。PyTorch 會自動處理大部分，但為了安全，我們讓 Input 符合 Model
                        # 或是我們暫時 cast UNet 為 FP32 (這會吃爆顯存)，或者我們在計算時小心轉型
                        
                        # 策略：輸入轉 FP16 跑 Forward，得到 FP16 output，然後轉 FP32 算 Loss
                        # 這樣可以省很多顯存，但反向傳播的精度可能會有一點點損失
                        
                        unet_input = latents.to(dtype=self.pipe.unet.dtype) # 轉 FP16
                        
                        # 只傳入 Positive Prompt Embeds 來算 Attention Maps 比較準
                        # 這裡我們只用 batch=1 (Positive) 來做 Guidance，因為我們只在乎 Subject 在哪
                        unet_output = self.pipe.unet(
                            unet_input, t, 
                            encoder_hidden_states=prompt_embeds, # 這裡只傳 positive
                            added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids[:1]}
                        )
                        
                        current_score = unet_output.sample
                        
                        # 轉回 FP32 進行數值計算
                        score_fp32 = current_score.float()
                        score_norm = torch.linalg.norm(score_fp32.flatten())

                        total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)

                        for box_idx, (bbox, token_idxs, box_rho) in enumerate(zip(bboxes, obj_token_indices, rho_list)):
                            if not token_idxs: continue
                            
                            # SDXL 解析度較高，我們將 target_res 設為 32 (對應 1024 的 32x32 feature map)
                            target_res = 32 
                            extracted_maps = []
                            
                            for name, processor in self.pipe.unet.attn_processors.items():
                                if hasattr(processor, "attention_maps") and processor.attention_maps is not None:
                                    try:
                                        # 取出 map，轉 FP32
                                        attn_map = processor.attention_maps[:, :, token_idxs].mean(dim=2).mean(dim=0).float()
                                        res = int(np.sqrt(attn_map.shape[0]))
                                        
                                        # SDXL 的 feature map 尺寸很多，我們選取較有語義意義的中層
                                        if res in [32, 64]: 
                                            m = attn_map.view(res, res)
                                            if res != target_res:
                                                m = F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(target_res, target_res), mode="bilinear", align_corners=False).squeeze()
                                            extracted_maps.append(m)
                                    except: continue
                            
                            if not extracted_maps: continue
                            
                            avg_map = torch.stack(extracted_maps).mean(dim=0)
                            
                            # Normalize
                            avg_map = (avg_map - avg_map.min()) / (avg_map.max() - avg_map.min() + 1e-6)
                            avg_map_prob = avg_map / (avg_map.sum() + 1e-6)

                            # Bounding Box indices
                            x, y, w, h = bbox
                            x_idx, y_idx = int(x*target_res), int(y*target_res)
                            w_idx, h_idx = max(1, int(w*target_res)), max(1, int(h*target_res))
                            
                            # Energy Functions (Same as Paper)
                            mask = torch.zeros_like(avg_map)
                            mask[y_idx:y_idx+h_idx, x_idx:x_idx+w_idx] = 1.0
                            intersection = (avg_map * mask).sum()
                            union = avg_map.sum()
                            loss_eq2 = (1.0 - (intersection / (union + 1e-6))) ** 2

                            # gaussian_target = self.compute_gaussian_prior(target_res, target_res, (x_idx, y_idx, w_idx, h_idx))
                            # gaussian_target_prob = gaussian_target / (gaussian_target.sum() + 1e-6)
                            
                            # epsilon = 1e-6
                            # loss_prior = torch.sum(avg_map_prob * torch.log((avg_map_prob + epsilon) / (gaussian_target_prob + epsilon)))
                            # Prior Loss (Gaussian prior on average map)
                            gaussian_target = self.compute_gaussian_prior(target_res, target_res, (x_idx, y_idx, w_idx, h_idx))
                            loss_prior = F.mse_loss(avg_map.mean(dim=0), gaussian_target)
                            
                            
                            current_rho = box_rho * decay_factor
                            total_loss += loss_eq2 + (current_rho * loss_prior)

                        if total_loss != 0:
                            # 梯度計算 (會通過 FP16 的 UNet 回傳到 FP32 的 Latents)
                            grad_energy = torch.autograd.grad(total_loss, latents)[0]
                            
                            if torch.isnan(grad_energy).any():
                                print(f"[Warning] NaN in gradient at step {i}")
                                continue

                            grad_energy_norm = torch.linalg.norm(grad_energy.flatten())

                            nu = score_norm / (grad_energy_norm + 1e-6)
                            nu = torch.clamp(nu, max=100.0) 

                            total_grad_vec = score_fp32 - (nu * grad_energy)
                            total_grad_norm = torch.linalg.norm(total_grad_vec.flatten())

                            step_size_xi = 2 * (signal_to_noise_r * score_norm / (total_grad_norm + 1e-8)) ** 2
                            
                            update = step_size_xi * total_grad_vec
                            update = torch.clamp(update, -0.2, 0.2) 
                            
                            latents = latents + update 
                            
                            if i % 5 == 0 and k == 0:
                                print(f"[Step {i}] Nu: {nu.item():.2f}")

                            del total_loss, grad_energy, score_fp32
                            torch.cuda.empty_cache()
                    
                    latents = latents.detach()

            # Standard Denoising Step (SDXL)
            with torch.no_grad():
                # 準備輸入：需要處理 classifier-free guidance
                # 轉型成 FP16 餵給 Scheduler 和 Model
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = latent_model_input.to(dtype=self.pipe.unet.dtype)
                
                noise_pred = self.pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds_all,
                    added_cond_kwargs=added_cond_kwargs
                ).sample
                
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # Step 輸出通常與輸入同 dtype，如果 scheduler 支援 FP32 latents，我們希望它回傳 FP32
                # 為了安全，我們手動轉回 FP32 更新 latents
                latents = self.pipe.scheduler.step(noise_pred, t, latents.to(dtype=noise_pred.dtype)).prev_sample
                latents = latents.float() # 保持 FP32

            if i % 10 == 0:
                with torch.no_grad():
                    # Step Image Decoding
                    step_img = self.decode_latents_to_pil(latents)
                    step_images.append((step_img, f"step {i}"))

        latents = latents.detach()
        with torch.no_grad():
            image = self.decode_latents_to_pil(latents)
        
        # Heatmap (省略，邏輯同上，需注意 final_attn_maps 抓取時機)
        heatmap_images = []

        return image, heatmap_images, step_images

# ==========================================
# Utils & Gradio 部分保持不變 (除了 Generator 初始化)
# ==========================================
# 注意：記得修改 Utils 中的 image size 預設為 1024
generator_model = WinWinLayPipelineSDXL(device="cuda")


# ==========================================
# 2. Utils
# ==========================================
def draw_dashed_rect(draw, rect, color, width=3, dash_length=8):
    x0, y0, w, h = rect
    x1, y1 = x0 + w, y0 + h
    lines = [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
    for start, end in lines:
        x_start, y_start = start
        x_end, y_end = end
        total_len = int(np.hypot(x_end - x_start, y_end - y_start))
        if total_len == 0: continue
        dx, dy = (x_end - x_start) / total_len, (y_end - y_start) / total_len
        for i in range(0, total_len, dash_length * 2):
            s_x = x_start + dx * i
            s_y = y_start + dy * i
            e_x = x_start + dx * min(i + dash_length, total_len)
            e_y = y_start + dy * min(i + dash_length, total_len)
            draw.line([(s_x, s_y), (e_x, e_y)], fill=color, width=width)

def draw_layout_vis(image, bboxes, subjects):
    vis_img = image.copy()
    draw = ImageDraw.Draw(vis_img)
    W, H = image.size
    colors = ["#FF3333", "#33FF33", "#3333FF", "#FFFF33"]
    try: font = ImageFont.truetype("arial.ttf", 24)
    except: font = ImageFont.load_default()
    for i, (bbox, subject) in enumerate(zip(bboxes, subjects)):
        color = colors[i % len(colors)]
        bx, by, bw, bh = bbox
        x, y, w, h = int(bx * W), int(by * H), int(bw * W), int(bh * H)
        draw_dashed_rect(draw, (x, y, w, h), color, width=4)
        text_w, text_h = 0, 0
        if hasattr(draw, "textbbox"):
            tb = draw.textbbox((0, 0), subject, font=font)
            text_w, text_h = tb[2] - tb[0], tb[3] - tb[1]
        else: text_w, text_h = draw.textsize(subject, font=font)
        draw.rectangle([(x, y - text_h - 6), (x + text_w + 12, y)], fill=color)
        draw.text((x + 6, y - text_h - 4), subject, fill="white", font=font)
    return vis_img

# ==========================================
# 3. Gradio
# ==========================================

def add_object_row(objects_df):
    if objects_df is None: objects_df = []
    # 預設 Rho 設為 5.0
    return objects_df + [[False, "", 0.0, 0.0, 0.0, 0.0, 5.0]] 

def remove_marked_rows(objects_df):
    if objects_df is None: return []
    return [row for row in objects_df if len(row) >= 7 and not bool(row[0])]

def blank_canvas(size=1024):
    return np.ones((size, size, 3), dtype=np.uint8) * 255

def handle_image_select(image, start_point, evt: gr.SelectData):
    if evt is None or image is None: return start_point, None, "未選取"
    h, w = image.shape[:2]
    idx = getattr(evt, "index", None)
    if not isinstance(idx, (list, tuple)) or len(idx) < 2: return start_point, None, "Error"
    x2, y2 = float(idx[0]), float(idx[1])
    if start_point is None: return (x2, y2), None, f"起點 ({x2:.0f}, {y2:.0f})"
    x1, y1 = start_point
    x_min, x_max = sorted([x1, x2])
    y_min, y_max = sorted([y1, y2])
    box_w, box_h = max(1, x_max-x_min), max(1, y_max-y_min)
    return None, [x_min/w, y_min/h, box_w/w, box_h/h], "選取完成"

def add_selection_to_df(objects_df, subject, rho, norm_box):
    if norm_box is None: return objects_df or []
    if objects_df is None: objects_df = []
    x, y, w, h = [float(v) for v in norm_box]
    return objects_df + [[False, str(subject), x, y, w, h, float(rho)]]

def run_interface(prompt, objects_df, method, seed):
    subjects, bboxes, rhos = [], [], []
    if objects_df is None: objects_df = []
    for row in objects_df:
        try:
            if len(row) < 7: continue
            name = str(row[1]).strip()
            if name == "" or name.lower() == "nan": continue
            subjects.append(name)
            bboxes.append([float(row[i]) for i in range(2, 6)])
            rhos.append(float(row[6]))
        except: pass
    
    if subjects: print(f"Generating: {subjects}, Method: {method}")
    image, heatmaps, step_images = generator_model.generate(
        prompt=prompt, bboxes=bboxes, subjects=subjects, rho_list=rhos, method=method, seed=int(seed)
    )
    annotated_image = draw_layout_vis(image, bboxes, subjects)
    return annotated_image, heatmaps, step_images

css = "#col-container {max_width: 1200px; margin: 0 auto;}"
with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# 🦁 WinWinLay Comparison (SD 1.5 - Corrected FP32 Ver.)")
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Prompt", value="A teddy bear and a hello kitty sit in front of the Eiffel Tower.")
                seed = gr.Number(label="Seed", value=42)
                method_select = gr.Radio(
                    choices=["WinWinLay (Official Paper Setting)", "Baseline (Old Method)"],
                    value="WinWinLay (Official Paper Setting)",
                    label="生成模式"
                )
                select_canvas = gr.Image(value=blank_canvas(1024), label="畫布", type="numpy", interactive=True, height=300)
                selection_info = gr.Markdown("點擊畫布以選取")
                with gr.Row():
                    sel_subj = gr.Textbox(show_label=False, placeholder="Subject")
                    sel_rho = gr.Number(value=5.0, label="Rho (Prior Strength)")
                    add_sel_btn = gr.Button("⬇️ 加入")
                sel_state = gr.State(None)
                start_state = gr.State(None)
                with gr.Row():
                    add_btn = gr.Button("➕")
                    del_btn = gr.Button("🗑️")
                objects_df = gr.Dataframe(
                    headers=["del", "subject", "x", "y", "w", "h", "rho"],
                    datatype=["bool", "str", "number", "number", "number", "number", "number"],
                    value=[
                        [False, "teddy bear", 0.1, 0.4, 0.3, 0.4, 5.0],
                        [False, "hello kitty", 0.6, 0.4, 0.3, 0.4, 5.0],
                        [False, "Eiffel Tower", 0.35, 0.1, 0.3, 0.5, 5.0],
                    ],
                    type="array", col_count=7
                )
                run_btn = gr.Button("Generate", variant="primary")
            with gr.Column(scale=1):
                result_img = gr.Image(label="Result")
                hm_gallery = gr.Gallery(label="Heatmaps", columns=2)
                step_gallery = gr.Gallery(label="Steps", columns=4)

    run_btn.click(run_interface, [prompt, objects_df, method_select, seed], [result_img, hm_gallery, step_gallery])
    select_canvas.select(handle_image_select, [select_canvas, start_state], [start_state, sel_state, selection_info])
    add_sel_btn.click(add_selection_to_df, [objects_df, sel_subj, sel_rho, sel_state], [objects_df])
    add_btn.click(add_object_row, [objects_df], [objects_df])
    del_btn.click(remove_marked_rows, [objects_df], [objects_df])

if __name__ == "__main__":
    demo.launch(share=True)