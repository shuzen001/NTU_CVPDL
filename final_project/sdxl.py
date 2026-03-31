import torch
import torch.nn.functional as F
import numpy as np
import cv2
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from diffusers.models.attention_processor import AttnProcessor
import copy

# ==========================================
# 1. 核心模型與 Attention Hook (SDXL)
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
        
        # SDXL Token Length check (77 for CLIP)
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
        print("正在載入 SDXL Base 模型...")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to(device)
        self.pipe.vae.to(dtype=torch.float32) # Fix black image issue
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        self.hooked_processors = {}
        for name, _ in self.pipe.unet.attn_processors.items():
            if "attn2" in name: 
                self.hooked_processors[name] = WinWinLayAttnProcessor()
            else:
                self.hooked_processors[name] = AttnProcessor()
        self.pipe.unet.set_attn_processor(self.hooked_processors)
        print("SDXL 模型載入完成。")

    def decode_latents_to_pil(self, latents):
        latents = latents.to(dtype=torch.float32)
        image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
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
            if len(word_tokens) == 0: indices.append([]); continue
            starts = find_subseq_positions(tokens, word_tokens)
            if not starts: indices.append([]); continue
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

    def generate(self, prompt, bboxes, subjects, rho_list, method="Ours (Per-Box)", steps=40, guidance_scale=7.5, seed=42):
        generator = torch.Generator(self.device).manual_seed(seed)
        height, width = 1024, 1024

        (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds) = self.pipe.encode_prompt(
            prompt=prompt, device=self.device, do_classifier_free_guidance=True
        )
        obj_token_indices = self.get_token_indices(prompt, subjects)

        latents = torch.randn((1, 4, height // 8, width // 8), generator=generator, device=self.device, dtype=torch.float16)
        self.pipe.scheduler.set_timesteps(steps)
        latents = latents * self.pipe.scheduler.init_noise_sigma

        final_attn_maps = {} 
        step_images = []

        print(f"開始 SDXL 生成... 模式: {method}")
        
        # Prepare added conditions
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self.pipe._get_add_time_ids((height, width), (0, 0), (height, width), dtype=torch.float16, text_encoder_projection_dim=self.pipe.text_encoder_2.config.projection_dim).to(self.device)
        
        if True: # CFG
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
        
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        # Parameters
        early_steps = 25
        prior_decay_steps = 20
        
        # Langevin Parameters (more conservative to reduce overfit)
        langevin_steps = 1
        step_size_xi = 0.05 # smaller ξ avoids over-updating

        for i, t in enumerate(self.pipe.scheduler.timesteps):
            
            # --- WinWinLay / Backward Guidance Phase ---
            if i < early_steps and len(bboxes) > 0: 
                # Start Langevin Loop
                for _ in range(langevin_steps):
                    with torch.enable_grad():
                        # Keep latents in the UNet's expected dtype to avoid dtype mismatches in embeddings
                        latents = latents.detach().to(dtype=self.pipe.unet.dtype).requires_grad_(True)
                        
                        # Clear Hooks
                        for name, processor in self.pipe.unet.attn_processors.items():
                            if hasattr(processor, "attention_maps"): processor.attention_maps = None

                        # Forward Pass (Positive Prompt Only for Guidance)
                        unet_output = self.pipe.unet(
                            latents, t, 
                            encoder_hidden_states=prompt_embeds[1].unsqueeze(0), # Positive prompt only
                            added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": self.pipe._get_add_time_ids((height, width), (0, 0), (height, width), dtype=torch.float16, text_encoder_projection_dim=self.pipe.text_encoder_2.config.projection_dim).to(self.device)}
                        )
                        
                        # Calculate ||Score|| (Eq. 20 Numerator)
                        current_score = unet_output.sample if hasattr(unet_output, "sample") else unet_output[0]
                        score_norm = torch.linalg.norm(current_score.flatten())

                        total_loss = 0.0
                        decay_factor = max(0.0, 1.0 - (float(i) / float(prior_decay_steps)))

                        for box_idx, (bbox, token_idxs, box_rho) in enumerate(zip(bboxes, obj_token_indices, rho_list)):
                            if not token_idxs: continue
                            
                            # Aggregate Maps: Check both 16x16 and 32x32 for robustness
                            extracted_maps = []
                            for target_res in [16, 32]:
                                for name, processor in self.pipe.unet.attn_processors.items():
                                    if hasattr(processor, "attention_maps") and processor.attention_maps is not None:
                                        try:
                                            # Average over heads and selected tokens
                                            attn_map = processor.attention_maps[:, :, token_idxs].mean(dim=2).mean(dim=0)
                                            res = int(np.sqrt(attn_map.shape[0]))
                                            if res == target_res:
                                                m = attn_map.view(res, res).unsqueeze(0).unsqueeze(0)
                                                if res != 32:
                                                    m = F.interpolate(m, size=(32, 32), mode='bilinear', align_corners=False)
                                                extracted_maps.append(m.squeeze())
                                        except: continue
                            
                            if not extracted_maps: continue
                            
                            # Final Aggregated Map (Normalized)
                            avg_map = torch.stack(extracted_maps).mean(dim=0)
                            if avg_map.dim() == 2:
                                avg_map = avg_map.unsqueeze(0)
                            
                            # Normalize map for stability (keep relative density)
                            avg_map = (avg_map - avg_map.min()) / (avg_map.max() - avg_map.min() + 1e-8)
                            final_attn_maps[subjects[box_idx]] = avg_map.detach().clone()

                            # BBox Coordinates
                            x, y, w, h = bbox
                            res = 32
                            x_idx, y_idx = int(x*res), int(y*res)
                            w_idx, h_idx = max(1, int(w*res)), max(1, int(h*res))
                            
                            # Energy Loss (IoU-style, stable scale)
                            mask = torch.zeros_like(avg_map)
                            mask[:, y_idx:y_idx+h_idx, x_idx:x_idx+w_idx] = 1.0
                            intersection = (avg_map * mask).sum()
                            union = avg_map.sum()
                            loss_eq2 = (1.0 - (intersection / (union + 1e-8))) ** 2

                            # Prior Loss (Gaussian prior on average map)
                            gaussian_target = self.compute_gaussian_prior(32, 32, (x_idx, y_idx, w_idx, h_idx)).to(avg_map.dtype)
                            
                            # 1. Normalize Attention Map to be a probability distribution
                            att_prob = avg_map.mean(dim=0)
                            att_prob = att_prob / (att_prob.sum() + 1e-6)

                            # 2. Normalize Gaussian Target
                            gaussian_prob = gaussian_target / (gaussian_target.sum() + 1e-6)

                            # 3. Compute KL Divergence
                            loss_prior = (att_prob * (torch.log(att_prob + 1e-6) - torch.log(gaussian_prob + 1e-6))).sum()

                            if "Baseline" in method:
                                total_loss += loss_eq2 
                            elif "WinWinLay" in method:
                                total_loss += loss_eq2 + (loss_prior * decay_factor)
                            else: # Ours
                                total_loss += loss_eq2 + (box_rho * decay_factor * loss_prior)

                        if total_loss != 0:
                            grad = torch.autograd.grad(total_loss, latents)[0]
                            grad_norm = torch.linalg.norm(grad.flatten())

                            if "WinWinLay" in method:
                                # Adaptive update with safer bounds (no added noise to keep box control strong)
                                safe_grad_norm = max(grad_norm.item(), 1e-3)
                                nu = score_norm / safe_grad_norm
                                nu = torch.clamp(nu, max=300.0)
                                
                                drift = step_size_xi * nu * grad
                                drift = torch.clamp(drift, -0.2, 0.2)
                                
                                latents = latents - drift
                                
                                if i % 10 == 0:
                                    print(f"[Step {i}] Nu: {nu.item():.2f} | Drift Mean: {drift.abs().mean().item():.4f}")

                            else:
                                # Baseline / Ours: Standard Gradient Descent
                                grad = torch.clamp(grad, -1.0, 1.0)
                                latents = latents - grad * 0.5 # Fixed step size

                            del total_loss, grad
                            torch.cuda.empty_cache()
                    
                    latents = latents.detach()

            # Standard Denoising Step
            with torch.no_grad():
                latent_input = torch.cat([latents] * 2)
                latent_input = self.pipe.scheduler.scale_model_input(latent_input, t)
                noise_pred_out = self.pipe.unet(latent_input, t, encoder_hidden_states=prompt_embeds, added_cond_kwargs=added_cond_kwargs)
                noise_pred = noise_pred_out.sample if hasattr(noise_pred_out, "sample") else noise_pred_out[0]
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
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
            if arr.ndim > 2:
                arr = arr.squeeze()
            if arr.ndim == 3:
                arr = arr[0]
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            heatmap = cv2.applyColorMap((arr * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmap_images.append((Image.fromarray(cv2.resize(heatmap, (1024, 1024), interpolation=cv2.INTER_NEAREST)), subject))

        return image, heatmap_images, step_images

# ==========================================
# 2. Utils & Gradio (保持不變)
# ==========================================
def draw_dashed_rect(draw, rect, color, width=5, dash_length=12):
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
    try: font = ImageFont.truetype("arial.ttf", 40)
    except: font = ImageFont.load_default()
    for i, (bbox, subject) in enumerate(zip(bboxes, subjects)):
        color = colors[i % len(colors)]
        bx, by, bw, bh = bbox
        x, y, w, h = int(bx * W), int(by * H), int(bw * W), int(bh * H)
        draw_dashed_rect(draw, (x, y, w, h), color, width=6)
        text_w, text_h = 0, 0
        if hasattr(draw, "textbbox"):
            tb = draw.textbbox((0, 0), subject, font=font)
            text_w, text_h = tb[2] - tb[0], tb[3] - tb[1]
        else: text_w, text_h = draw.textsize(subject, font=font)
        draw.rectangle([(x, y - text_h - 10), (x + text_w + 16, y)], fill=color)
        draw.text((x + 8, y - text_h - 8), subject, fill="white", font=font)
    return vis_img

generator_model = WinWinLayPipelineSDXL(device="cuda")

def add_object_row(objects_df):
    if objects_df is None: objects_df = []
    return objects_df + [[False, "", 0.0, 0.0, 0.0, 0.0, 20.0]]

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
        gr.Markdown("# 🦁 WinWinLay Comparison")
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Prompt", value="A teddy bear and a hello kitty sit in front of the Eiffel Tower.")
                seed = gr.Number(label="Seed", value=42)
                method_select = gr.Radio(
                    choices=["Baseline (Old Method)", "WinWinLay (Global Rho + Adaptive)", "Ours (Per-Box + Manual)"],
                    value="Ours (Per-Box + Manual)",
                    label="生成模式"
                )
                select_canvas = gr.Image(value=blank_canvas(), label="畫布", type="numpy", interactive=True, height=300)
                selection_info = gr.Markdown("點擊畫布以選取")
                with gr.Row():
                    sel_subj = gr.Textbox(show_label=False, placeholder="Subject")
                    sel_rho = gr.Number(value=20, show_label=False)
                    add_sel_btn = gr.Button("⬇️ 加入")
                sel_state = gr.State(None)
                start_state = gr.State(None)
                with gr.Row():
                    add_btn = gr.Button("➕")
                    del_btn = gr.Button("🗑️")
                objects_df = gr.Dataframe(
                    headers=["del", "subject", "x", "y", "w", "h", "rho"],
                    datatype=["bool", "str", "number", "number", "number", "number", "number"],
                    value=[[False, "teddy bear", 0.1, 0.4, 0.3, 0.4, 20], [False, "hello kitty", 0.6, 0.4, 0.3, 0.4, 20], [False, "Eiffel Tower", 0.35, 0.1, 0.3, 0.5, 10]],
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
