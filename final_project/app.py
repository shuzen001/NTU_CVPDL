import torch
import torch.nn.functional as F
import numpy as np
import cv2
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.models.attention_processor import AttnProcessor
import copy

# ==========================================
# 1. 核心模型與 Attention Hook (SD 1.5 版)
# ==========================================

class WinWinLayAttnProcessor(AttnProcessor):
    def __init__(self):
        super().__init__()
        self.attention_maps = None 

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        # 儲存 Attention Maps (保持 FP16 即可，計算時再轉 FP32)
        if encoder_hidden_states.shape[1] == 77:
            self.attention_maps = attention_probs
            
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

class WinWinLayPipelineSD15:
    def __init__(self, device="cuda"):
        self.device = device
        print("正在載入 SD 1.5 模型...")
        # 載入 FP16 模型以節省顯存，但關鍵計算會轉 FP32
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            torch_dtype=torch.float32
        ).to(device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        self.hooked_processors = {}
        for name, _ in self.pipe.unet.attn_processors.items():
            if "attn2" in name: 
                self.hooked_processors[name] = WinWinLayAttnProcessor()
            else:
                self.hooked_processors[name] = AttnProcessor()
        self.pipe.unet.set_attn_processor(self.hooked_processors)
        print("SD 1.5 模型載入完成。")

    def decode_latents_to_pil(self, latents):
            # 1. 正常的 Rescale (這是對的)
            latents = 1 / self.pipe.vae.config.scaling_factor * latents
            
            # 2. 【關鍵修正】強制將 VAE 與 Latents 轉為 FP32 進行解碼
            # SD 1.5 的 VAE 在 FP16 下會炸掉 (Overflow)，導致圖片充滿雜訊
            needs_upcast = self.pipe.vae.dtype == torch.float16
            
            if needs_upcast:
                self.pipe.vae.to(dtype=torch.float32)
                
            # 使用 FP32 解碼
            image = self.pipe.vae.decode(latents.to(dtype=torch.float32), return_dict=False)[0]
            
            # (選擇性) 如果顯存非常吃緊，解碼後可以把 VAE 轉回 FP16，但通常沒必要
            if needs_upcast:
                self.pipe.vae.to(dtype=torch.float16)
                
            return self.pipe.image_processor.postprocess(image, output_type="pil")[0]

    def get_token_indices(self, prompt, subject_words):
        tokens = self.pipe.tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()

        def find_subseq_positions(seq, subseq):
            positions = []
            if len(subseq) == 0 or len(subseq) > len(seq): return positions
            for i in range(len(seq) - len(subseq) + 1):
                if seq[i:i+len(subseq)] == subseq: positions.append(i)
            return positions

        indices = []
        for word in subject_words:
            word_tokens = self.pipe.tokenizer(word, add_special_tokens=False)["input_ids"]
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
        # 【修正】這裡使用 FP32 確保高斯分佈計算精確
        x = torch.arange(width, device=self.device).float()
        y = torch.arange(height, device=self.device).float()
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        sigma_x = max(w_box * sigma_scale, 1e-6)
        sigma_y = max(h_box * sigma_scale, 1e-6)
        
        gaussian = torch.exp(-((xx - cx)**2 / (2 * sigma_x**2) + (yy - cy)**2 / (2 * sigma_y**2)))
        return gaussian

    def generate(self, prompt, bboxes, subjects, rho_list, method="WinWinLay", steps=50, guidance_scale=7.5, seed=42):
        generator = torch.Generator(self.device).manual_seed(seed)
        height, width = 512, 512

        # 1. Prepare Embeddings
        text_input = self.pipe.tokenizer(prompt, padding="max_length", max_length=self.pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embeddings = self.pipe.text_encoder(text_input.input_ids.to(self.device))[0]
        uncond_input = self.pipe.tokenizer([""], padding="max_length", max_length=text_input.input_ids.shape[-1], return_tensors="pt")
        uncond_embeddings = self.pipe.text_encoder(uncond_input.input_ids.to(self.device))[0]
        
        prompt_embeds = torch.cat([uncond_embeddings, text_embeddings])
        target_text_embeddings = text_embeddings

        obj_token_indices = self.get_token_indices(prompt, subjects)

        # 2. Init Latents (全程使用 FP32)
        latents = torch.randn((1, self.pipe.unet.config.in_channels, height // 8, width // 8), generator=generator, device=self.device, dtype=torch.float32)
        self.pipe.scheduler.set_timesteps(steps)
        latents = latents * self.pipe.scheduler.init_noise_sigma

        final_attn_maps = {} 
        step_images = []

        print(f"開始 SD 1.5 生成... 模式: {method} (FP32 Mode)")

        # === 論文參數設定 ===
        early_steps = 10 
        prior_decay_steps = 10 
        langevin_steps_O = 4 
        signal_to_noise_r = 0.06 

        for i, t in enumerate(self.pipe.scheduler.timesteps):
            
            # --- WinWinLay / Backward Guidance Phase ---
            if i < early_steps and len(bboxes) > 0 and "WinWinLay" in method:
                
                decay_factor = max(0.0, 1.0 - (float(i) / float(prior_decay_steps)))
                
                # Langevin Dynamics Inner Loop
                for k in range(langevin_steps_O):
                    with torch.enable_grad():
                        # Detach and require grad (保持 FP32)
                        latents = latents.detach().requires_grad_(True)
                        
                        # Clear maps
                        for name, processor in self.pipe.unet.attn_processors.items():
                            if hasattr(processor, "attention_maps"): processor.attention_maps = None

                        # Forward pass (Input: FP32, Model: FP32)
                        unet_output = self.pipe.unet(
                            latents, t, 
                            encoder_hidden_states=target_text_embeddings
                        )
                        
                        current_score = unet_output.sample if hasattr(unet_output, "sample") else unet_output[0]
                        
                        # Score 已經是 FP32，不需要轉型
                        score_fp32 = current_score
                        score_norm = torch.linalg.norm(score_fp32.flatten())

                        total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)

                        for box_idx, (bbox, token_idxs, box_rho) in enumerate(zip(bboxes, obj_token_indices, rho_list)):
                            if not token_idxs: continue
                            
                            target_res = 16 
                            extracted_maps = []
                            
                            for name, processor in self.pipe.unet.attn_processors.items():
                                if hasattr(processor, "attention_maps") and processor.attention_maps is not None:
                                    try:
                                        # 注意：這裡 map 取出來如果是 fp16 還是要轉 fp32，但因為模型是 fp32，這裡應該原生就是 fp32
                                        attn_map = processor.attention_maps[:, :, token_idxs].mean(dim=2).mean(dim=0).float()
                                        res = int(np.sqrt(attn_map.shape[0]))
                                        
                                        if res in [8, 16]: # Middle(8x8) & First Up(16x16)
                                            m = attn_map.view(res, res)
                                            if res != target_res:
                                                m = F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(target_res, target_res), mode="bilinear", align_corners=False).squeeze()
                                            extracted_maps.append(m)
                                    except: continue
                            
                            if not extracted_maps: continue
                            
                            avg_map = torch.stack(extracted_maps).mean(dim=0) # FP32
                            
                            # Normalize
                            avg_map = (avg_map - avg_map.min()) / (avg_map.max() - avg_map.min() + 1e-6)
                            avg_map_prob = avg_map / (avg_map.sum() + 1e-6)

                            # Bounding Box indices
                            x, y, w, h = bbox
                            x_idx, y_idx = int(x*target_res), int(y*target_res)
                            w_idx, h_idx = max(1, int(w*target_res)), max(1, int(h*target_res))
                            
                            # 1. Traditional Energy Function
                            mask = torch.zeros_like(avg_map)
                            mask[y_idx:y_idx+h_idx, x_idx:x_idx+w_idx] = 1.0
                            intersection = (avg_map * mask).sum()
                            union = avg_map.sum()
                            loss_eq2 = (1.0 - (intersection / (union + 1e-6))) ** 2

                            # 2. Non-local Attention Prior (KL Divergence)
                            gaussian_target = self.compute_gaussian_prior(target_res, target_res, (x_idx, y_idx, w_idx, h_idx))
                            gaussian_target_prob = gaussian_target / (gaussian_target.sum() + 1e-6)
                            
                            epsilon = 1e-6
                            loss_prior = torch.sum(avg_map_prob * torch.log((avg_map_prob + epsilon) / (gaussian_target_prob + epsilon)))

                            current_rho = box_rho * decay_factor
                            total_loss += loss_eq2 + (current_rho * loss_prior)

                        if total_loss != 0:
                            # 梯度計算 (FP32)
                            grad_energy = torch.autograd.grad(total_loss, latents)[0]
                            
                            # 檢查 NaN
                            if torch.isnan(grad_energy).any():
                                print(f"[Warning] NaN detected in gradient at step {i}, skipping update.")
                                continue

                            grad_energy_norm = torch.linalg.norm(grad_energy.flatten())

                            # Adaptive Weighting
                            nu = score_norm / (grad_energy_norm + 1e-6)
                            nu = torch.clamp(nu, max=100.0) 

                            total_grad_vec = score_fp32 - (nu * grad_energy)
                            total_grad_norm = torch.linalg.norm(total_grad_vec.flatten())

                            # Dynamic Step Size
                            step_size_xi = 2 * (signal_to_noise_r * score_norm / (total_grad_norm + 1e-8)) ** 2
                            
                            # Update
                            update = step_size_xi * total_grad_vec
                            update = torch.clamp(update, -0.2, 0.2) 
                            
                            # 【修正重點】不要再轉回 FP16 了，保持 Float
                            latents = latents + update 
                            # 舊的寫法：latents = latents.to(dtype=torch.float16)  <-- 刪除這行
                            
                            if i % 5 == 0 and k == 0:
                                print(f"[Step {i}] Nu: {nu.item():.2f} | Xi: {step_size_xi.item():.6f}")

                            del total_loss, grad_energy, score_fp32
                            torch.cuda.empty_cache()
                    
                    # Detach for next inner loop iteration
                    # 【修正重點】這裡也要保持 FP32
                    latents = latents.detach() # 舊的寫法有 .to(dtype=torch.float16)，也要刪除

            # Standard Denoising Step (Outer Loop)
            with torch.no_grad():
                # Scheduler 需要 FP32 還是 FP16 取決於 pipe 配置，但 FP32 比較安全
                latent_input = torch.cat([latents] * 2)
                latent_input = self.pipe.scheduler.scale_model_input(latent_input, t)
                
                noise_pred = self.pipe.unet(latent_input, t, encoder_hidden_states=prompt_embeds).sample
                
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # step 返回的 latents 類型會跟隨輸入，所以它會保持 FP32
                latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample

            if i % 10 == 0:
                with torch.no_grad():
                    step_img = self.decode_latents_to_pil(latents)
                    step_images.append((step_img, f"step {i}"))

        latents = latents.detach()
        with torch.no_grad():
            image = self.decode_latents_to_pil(latents)
        
        heatmap_images = []
        for subject, map_tensor in final_attn_maps.items():
            arr = map_tensor.float().cpu().numpy()
            heatmap = cv2.applyColorMap((arr * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmap_images.append((Image.fromarray(cv2.resize(heatmap, (512, 512), interpolation=cv2.INTER_NEAREST)), subject))

        return image, heatmap_images, step_images

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
generator_model = WinWinLayPipelineSD15(device="cuda")

def add_object_row(objects_df):
    if objects_df is None: objects_df = []
    # 預設 Rho 設為 5.0
    return objects_df + [[False, "", 0.0, 0.0, 0.0, 0.0, 5.0]] 

def remove_marked_rows(objects_df):
    if objects_df is None: return []
    return [row for row in objects_df if len(row) >= 7 and not bool(row[0])]

def blank_canvas(size=512):
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
                select_canvas = gr.Image(value=blank_canvas(512), label="畫布", type="numpy", interactive=True, height=300)
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