import os
from pathlib import Path
import shutil

import gradio as gr
from apphub.app import App, AppOption
from apphub.helper import wait_for_port
from os.path import join


class Sdwebui(App):

    @property
    def key(self):
        """key 是应用的唯一标识，用于在数据库中查找应用，所以这个值应该是唯一的"""
        return "sdwebui"

    @property
    def port(self):
        return 20000

    @property
    def op_port(self):
        return 30000

    class SdwebuiOption(AppOption):
        launch_option: str | None = ""

    cfg: SdwebuiOption

    def render_installation_page(self) -> "gr.Blocks":
        with gr.Blocks() as demo:
            gr.Markdown(
                """# 安装 Stable Diffusion Web UI

在 Featurize 中使用 Stable Diffusion Web UI 做图。

<div style="color: red">因为 SD 的安装过程比较长，因此建议将 SD 安装在云盘中，这样下次开机无需重新安装便可直接使用。</div>

<div style="color: red">安装大约需要 10GB 的存储空间，整个过程持续 10 分钟，请不要切换或刷新页面，耐心等待安装成功。</div>

如果将 SD 安装在云盘中，则 SD 所有的插件、模型、数据等都会被保存在云盘中，这对云盘容量的要求比较高，<strong>请随时注意云盘容量的配额<strong>，以免发生报错或产生计费。
"""
            )
            install_location = self.render_install_location(allow_work=True, default="work")
            version = gr.Dropdown(choices=["1.10.1", "1.9.0"], label="安装的版本", value="1.10.1")
            launch_option = gr.Textbox(
                label="默认启动项", info="在启动时还可以进一步调整"
            )
            install_extension = gr.Dropdown(
                label="安装常用插件", choices=["插件包（V1.0.0）", "不安装"], value="插件包（V1.0.0）", info="因为安装的插件数量比较多，安装过程会延长，请耐心等待"
            )
            self.render_installation_button(
                inputs=[install_location, version, launch_option, install_extension]
            )
            self.render_log()
        return demo

    @property
    def source_location(self):
        return os.path.join(self.cfg.install_location, "stable-diffusion-webui")

    @property
    def conda_env_name(self):
        return "sdwebui" if not self.in_work else "/cloud/app/sdwebui/env"

    @property
    def conda_mode(self):
        return "p" if self.in_work else "n"

    def render_setting_page(self):
        with gr.Blocks() as g:
            gr.Markdown("""# 设置 Stable Diffusion Web UI""")
            launch_option = gr.Textbox(
                label="默认启动项", info="在启动时还可以进一步调整"
            )
            self.render_setting_button([launch_option])
        return g

    def setting(self, launch_option):
        super().setting(launch_option)

    def installation(self, install_location, version, launch_option, install_extension):
        super().installation(install_location, version, launch_option)
        self.execute_command(
            f"git clone --depth 1 --branch v{version} https://github.com/AUTOMATIC1111/stable-diffusion-webui.git",
            self.cfg.install_location,
        )
        self.execute_command(
            f"echo httpx[socks] >> ./stable-diffusion-webui/requirements.txt",
            self.cfg.install_location,
        )
        if not self.in_work:
            self.execute_command(
                f"conda create -y -n {self.conda_env_name} python=3.10"
            )
        else:
            self.execute_command(
                f"conda create -y --prefix {self.conda_env_name} python=3.10"
            )

        if install_extension:
            self.execute_command("git clone --depth 1 https://github.com/OpenTalker/SadTalker.git", "stable-diffusion-webui/extensions")
            self.execute_command("git clone --depth 1 https://github.com/ahgsql/StyleSelectorXL.git", "stable-diffusion-webui/extensions")
            self.execute_command("git clone --depth 1 https://github.com/DominikDoom/a1111-sd-webui-tagcomplete.git", "stable-diffusion-webui/extensions")
            self.execute_command("git clone --depth 1 https://github.com/Bing-su/adetailer.git", "stable-diffusion-webui/extensions")
            self.execute_command("git clone --depth 1 https://github.com/deforum-art/deforum-for-automatic1111-webui.git", "stable-diffusion-webui/extensions")
            self.execute_command("git clone --depth 1 https://github.com/s9roll7/ebsynth_utility.git", "stable-diffusion-webui/extensions")
            self.execute_command("git clone --depth 1 https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111.git", "stable-diffusion-webui/extensions")
            self.execute_command("git clone --depth 1 https://github.com/fkunn1326/openpose-editor.git", "stable-diffusion-webui/extensions")
            self.execute_command("git clone --depth 1 https://github.com/adieyal/sd-dynamic-prompts.git", "stable-diffusion-webui/extensions")
            self.execute_command("git clone --depth 1 https://github.com/kohya-ss/sd-webui-additional-networks.git", "stable-diffusion-webui/extensions")
            self.execute_command("git clone --depth 1 https://github.com/thomasasfk/sd-webui-aspect-ratio-helper.git", "stable-diffusion-webui/extensions")
            self.execute_command("git clone --depth 1 https://github.com/Mikubill/sd-webui-controlnet.git", "stable-diffusion-webui/extensions")
            self.execute_command("git clone --depth 1 https://github.com/zanllp/sd-webui-infinite-image-browsing.git", "stable-diffusion-webui/extensions")
            self.execute_command("git clone --depth 1 https://github.com/Uminosachi/sd-webui-inpaint-anything.git", "stable-diffusion-webui/extensions")
            self.execute_command("git clone --depth 1 https://github.com/huchenlei/sd-webui-openpose-editor.git", "stable-diffusion-webui/extensions")
            self.execute_command("git clone --depth 1 https://github.com/Physton/sd-webui-prompt-all-in-one.git", "stable-diffusion-webui/extensions")
            self.execute_command("git clone --depth 1 https://github.com/Gourieff/sd-webui-reactor.git", "stable-diffusion-webui/extensions")
            self.execute_command("git clone --depth 1 https://github.com/hanamizuki-ai/stable-diffusion-webui-localization-zh_Hans", "stable-diffusion-webui/extensions")
            self.execute_command("git clone --depth 1 https://github.com/AUTOMATIC1111/stable-diffusion-webui-rembg.git", "stable-diffusion-webui/extensions")
            self.execute_command("git clone --depth 1 https://github.com/Coyote-A/ultimate-upscale-for-automatic1111.git", "stable-diffusion-webui/extensions")

        with self.conda_activate(self.conda_env_name, self.conda_mode):
            self.execute_command(f"pip install 'httpx[socks]' xformers")
            (
                Path(self.cfg.install_location)
                / "stable-diffusion-webui"
                / "webui-user.sh"
            ).write_text(
                """export venv_dir="-"
"""
            )
            # 启动一次
            self.execute_command(
                f"bash webui.sh --listen --port {self.port}",
                daemon=True,
                cwd=self.source_location,
            )
            wait_for_port(self.port)
            self.close()

        # 调用 app_installed，标准流程，该函数会通知前端安装已经完成，切换到应用的页面
        self.app_installed()

    def render_start_page(self):
        with gr.Blocks() as demo:
            gr.Markdown(
                f"""# {self.name} 尚未启动

请点击下方按钮启动 {self.name}。

当前 Stable Diffusion Web UI 被安装在 {os.path.join(self.cfg.install_location, "sdwebui")} 中，你可以使用下方的「文件」应用访问这个目录手动管理文件。

如果使用遇到问题，请及时关注公众号后向我们反馈：https://docs.featurize.cn 中可扫码联系我们。
"""
            )
            mount_models = gr.Dropdown(
                label="挂载常用模型（不占用云端硬盘空间）",
                info="我们会定期更新常用的挂载模型内容，如果你有其他建议，也欢迎向我们反馈。",
                value="v1",
                choices=[
                    ("挂载常用模型", "v1"),
                    ("不挂载", "bare"),
                ]
            )
            button = self.render_start_button(inputs=[mount_models])
            self.render_log()
        return demo

    def link_model_file(self, src):
        s = Path("/home/featurize/.public/sdwebui/models/") / src
        t = Path(self.cfg.install_location) / "stable-diffusion-webui" / "models" / src
        t.parent.mkdir(parents=True, exist_ok=True)
        if not t.exists():
            self.logger.info(f"link {s} to {t}")
            os.symlink(s, t)

    def start(self, mount_models):
        """安装完成后，应用并不会立即开始运行，而是调用这个 start 函数。"""

        if mount_models == "v1":
            Path("/home/featurize/.public/sdwebui").mkdir(parents=True, exist_ok=True)
            self.execute_command(
                f"sudo mount -t nfs -o ro,acregmin=600,acregmax=3600,rsize=1048576,wsize=1048576,noatime,tcp 172.16.0.227:/featurize-public/sdwebui/assets_{mount_models} /home/featurize/.public/sdwebui"
            )
            self.link_model_file("BLIP/model_base_caption_capfilt_large.pth")
            self.link_model_file("Codeformer/codeformer-v0.1.0.pth")
            self.link_model_file("ControlNet/Instantid/config.json")
            self.link_model_file("ControlNet/Instantid/control_instant_id_sdxl.safetensors")
            self.link_model_file("ControlNet/Instantid/ip-adapter_instant_id_sdxl.bin")
            self.link_model_file("ControlNet/IP-Adapter/ip-adapter-faceid-plusv2_sd15.bin")
            self.link_model_file("ControlNet/IP-Adapter/ip-adapter-faceid-plusv2_sdxl.bin")
            self.link_model_file("ControlNet/IP-Adapter/ip-adapter-full-face_sd15.pth")
            self.link_model_file("ControlNet/IP-Adapter/ip-adapter_sd15_plus.pth")
            self.link_model_file("ControlNet/SD1.5/control_sd15_random_color.pth")
            self.link_model_file("ControlNet/SD1.5/control_v11e_sd15_ip2p.pth")
            self.link_model_file("ControlNet/SD1.5/control_v11e_sd15_shuffle.pth")
            self.link_model_file("ControlNet/SD1.5/control_v11f1e_sd15_tile.pth")
            self.link_model_file("ControlNet/SD1.5/control_v11f1p_sd15_depth.pth")
            self.link_model_file("ControlNet/SD1.5/control_v11p_sd15_canny.pth")
            self.link_model_file("ControlNet/SD1.5/control_v11p_sd15_inpaint.pth")
            self.link_model_file("ControlNet/SD1.5/control_v11p_sd15_lineart.pth")
            self.link_model_file("ControlNet/SD1.5/control_v11p_sd15_mlsd.pth")
            self.link_model_file("ControlNet/SD1.5/control_v11p_sd15_normalbae.pth")
            self.link_model_file("ControlNet/SD1.5/control_v11p_sd15_openpose.pth")
            self.link_model_file("ControlNet/SD1.5/control_v11p_sd15s2_lineart_anime.pth")
            self.link_model_file("ControlNet/SD1.5/control_v11p_sd15_scribble.pth")
            self.link_model_file("ControlNet/SD1.5/control_v11p_sd15_seg.pth")
            self.link_model_file("ControlNet/SD1.5/control_v11p_sd15_softedge.pth")
            self.link_model_file("ControlNet/SD1.5/control_v1p_sd15_qrcode_monster_v2.safetensors")
            self.link_model_file("ControlNet/SD1.5/ioclab_sd15_recolor.safetensors")
            self.link_model_file("ControlNet/SDXL/diffusers_xl_canny_full.safetensors")
            self.link_model_file("ControlNet/SDXL/diffusers_xl_depth_full.safetensors")
            self.link_model_file("ControlNet/SDXL/kohya_controllllite_xl_blur_anime_beta.safetensors")
            self.link_model_file("ControlNet/SDXL/kohya_controllllite_xl_blur.safetensors")
            self.link_model_file("ControlNet/SDXL/kohya_controllllite_xl_canny_anime.safetensors")
            self.link_model_file("ControlNet/SDXL/kohya_controllllite_xl_canny.safetensors")
            self.link_model_file("ControlNet/SDXL/kohya_controllllite_xl_depth_anime.safetensors")
            self.link_model_file("ControlNet/SDXL/kohya_controllllite_xl_depth.safetensors")
            self.link_model_file("ControlNet/SDXL/kohya_controllllite_xl_openpose_anime.safetensors")
            self.link_model_file("ControlNet/SDXL/kohya_controllllite_xl_openpose_anime_v2.safetensors")
            self.link_model_file("ControlNet/SDXL/kohya_controllllite_xl_scribble_anime.safetensors")
            self.link_model_file("ControlNet/SDXL/OpenPoseXL2.safetensors")
            self.link_model_file("ControlNet/SDXL/sai_xl_canny_256lora.safetensors")
            self.link_model_file("ControlNet/SDXL/sai_xl_depth_256lora.safetensors")
            self.link_model_file("ControlNet/SDXL/sai_xl_recolor_256lora.safetensors")
            self.link_model_file("ControlNet/SDXL/sai_xl_sketch_256lora.safetensors")
            self.link_model_file("ControlNet/SDXL/sargezt_xl_depth_faid_vidit.safetensors")
            self.link_model_file("ControlNet/SDXL/sargezt_xl_depth.safetensors")
            self.link_model_file("ControlNet/SDXL/sargezt_xl_depth_zeed.safetensors")
            self.link_model_file("ControlNet/SDXL/sargezt_xl_softedge.safetensors")
            self.link_model_file("ControlNet/SDXL/t2i-adapter_diffusers_xl_canny.safetensors")
            self.link_model_file("ControlNet/SDXL/t2i-adapter_diffusers_xl_depth_midas.safetensors")
            self.link_model_file("ControlNet/SDXL/t2i-adapter_diffusers_xl_depth_zoe.safetensors")
            self.link_model_file("ControlNet/SDXL/t2i-adapter_diffusers_xl_lineart.safetensors")
            self.link_model_file("ControlNet/SDXL/t2i-adapter_diffusers_xl_openpose.safetensors")
            self.link_model_file("ControlNet/SDXL/t2i-adapter_diffusers_xl_sketch.safetensors")
            self.link_model_file("ControlNet/SDXL/t2i-adapter_xl_canny.safetensors")
            self.link_model_file("ControlNet/SDXL/t2i-adapter_xl_openpose.safetensors")
            self.link_model_file("ControlNet/SDXL/t2i-adapter_xl_sketch.safetensors")
            self.link_model_file("ESRGAN/4x_NMKD-Siax_200k.pth")
            self.link_model_file("ESRGAN/4x_NMKD-Superscale-SP_178000_G.pth")
            self.link_model_file("ESRGAN/4x-UltraSharp.pth")
            self.link_model_file("ESRGAN/8x_NMKD-Faces_160000_G.pth")
            self.link_model_file("ESRGAN/8x_NMKD-Superscale_150000_G.pth")
            self.link_model_file("ESRGAN/lollypop.pth")
            self.link_model_file("GFPGAN/detection_Resnet50_Final.pth")
            self.link_model_file("GFPGAN/parsing_parsenet.pth")
            self.link_model_file("insightface/inswapper_128.onnx")
            self.link_model_file("insightface/models/buffalo_l/1k3d68.onnx")
            self.link_model_file("insightface/models/buffalo_l/2d106det.onnx")
            self.link_model_file("insightface/models/buffalo_l/det_10g.onnx")
            self.link_model_file("insightface/models/buffalo_l/genderage.onnx")
            self.link_model_file("insightface/models/buffalo_l/w600k_r50.onnx")
            self.link_model_file("karlo/ViT-L-14_stats.th")
            self.link_model_file("Lora/FaceID/ip-adapter-faceid-plusv2_sd15_lora.safetensors")
            self.link_model_file("Lora/FaceID/ip-adapter-faceid-plusv2_sdxl_lora.safetensors")
            self.link_model_file("Lora/Fast_model/lcm-lora-sdv1-5.safetensors")
            self.link_model_file("Lora/Fast_model/lcm-lora-sdxl.safetensors")
            self.link_model_file("Lora/SD1.5/anxiang-暗香-anxiang.json")
            self.link_model_file("Lora/SD1.5/anxiang-暗香-anxiang.png")
            self.link_model_file("Lora/SD1.5/anxiang-暗香-anxiang.safetensors")
            self.link_model_file("Lora/SD1.5/Dark Majic 中国刺绣繁花似锦 v1.0.json")
            self.link_model_file("Lora/SD1.5/Dark Majic 中国刺绣繁花似锦 v1.0.png")
            self.link_model_file("Lora/SD1.5/Dark Majic 中国刺绣繁花似锦 v1.0.safetensors")
            self.link_model_file("Lora/SD1.5/焱落纱_v1.0.json")
            self.link_model_file("Lora/SD1.5/焱落纱_v1.0.png")
            self.link_model_file("Lora/SD1.5/焱落纱_v1.0.safetensors")
            self.link_model_file("Lora/SD1.5/『醉枫染墨』国风主题-影楼专用模型 v2.0.json")
            self.link_model_file("Lora/SD1.5/『醉枫染墨』国风主题-影楼专用模型 v2.0.png")
            self.link_model_file("Lora/SD1.5/『醉枫染墨』国风主题-影楼专用模型 v2.0.safetensors")
            self.link_model_file("Lora/SDXL/gmic iocn_guofeng2D-000014 v1.0.png")
            self.link_model_file("Lora/SDXL/gmic iocn_guofeng2D-000014 v1.0.safetensors")
            self.link_model_file("Lora/SDXL/HandDrawing l 卡通手绘-SDXL_v1.0.png")
            self.link_model_file("Lora/SDXL/HandDrawing l 卡通手绘-SDXL_v1.0.safetensors")
            self.link_model_file("Lora/SDXL/电商-超现实主义v2.png")
            self.link_model_file("Lora/SDXL/电商-超现实主义v2.safetensors")
            self.link_model_file("prompt_expansion/config.json")
            self.link_model_file("prompt_expansion/merges.txt")
            self.link_model_file("prompt_expansion/positive.txt")
            self.link_model_file("prompt_expansion/pytorch_model.bin")
            self.link_model_file("prompt_expansion/special_tokens_map.json")
            self.link_model_file("prompt_expansion/tokenizer_config.json")
            self.link_model_file("prompt_expansion/tokenizer.json")
            self.link_model_file("prompt_expansion/vocab.json")
            self.link_model_file("reactor/faces/Mr.ma.safetensors")
            self.link_model_file("RealESRGAN/RealESRGAN_x4plus_anime_6B.pth")
            self.link_model_file("RealESRGAN/RealESRGAN_x4plus.pth")
            self.link_model_file("Stable-diffusion/SD1.5/DarkSushi 大颗寿司Mix 2.25D.png")
            self.link_model_file("Stable-diffusion/SD1.5/DarkSushi 大颗寿司Mix 2.25D.safetensors")
            self.link_model_file("Stable-diffusion/SD1.5/GhostMix鬼混_V2.0.png")
            self.link_model_file("Stable-diffusion/SD1.5/GhostMix鬼混_V2.0.safetensors")
            self.link_model_file("Stable-diffusion/SD1.5/majicMIX-realistic-麦橘写实-v7.png")
            self.link_model_file("Stable-diffusion/SD1.5/majicMIX-realistic-麦橘写实-v7.safetensors")
            self.link_model_file("Stable-diffusion/SD1.5/niji-动漫二次元_4.0.png")
            self.link_model_file("Stable-diffusion/SD1.5/niji-动漫二次元_4.0.safetensors")
            self.link_model_file("Stable-diffusion/SD1.5/QteaMix-通用Q版模型-gamma.png")
            self.link_model_file("Stable-diffusion/SD1.5/QteaMix-通用Q版模型-gamma.safetensors")
            self.link_model_file("Stable-diffusion/SD1.5/revAnimated_v2Pruned.png")
            self.link_model_file("Stable-diffusion/SD1.5/revAnimated_v2Pruned.safetensors")
            self.link_model_file("Stable-diffusion/SD3/sd3_medium_incl_clips_t5xxlfp16.jpg")
            self.link_model_file("Stable-diffusion/SD3/sd3_medium_incl_clips_t5xxlfp16.safetensors")
            self.link_model_file("Stable-diffusion/SDXL/copaxTimelessxlSDXL1_v12.png")
            self.link_model_file("Stable-diffusion/SDXL/copaxTimelessxlSDXL1_v12.safetensors")
            self.link_model_file("Stable-diffusion/SDXL/Juggernaut_X_RunDiffusion_Hyper.png")
            self.link_model_file("Stable-diffusion/SDXL/Juggernaut_X_RunDiffusion_Hyper.safetensors")
            self.link_model_file("Stable-diffusion/SDXL/sd_xl_refiner_1.0.safetensors")
            self.link_model_file("Stable-diffusion/SDXL/比鲁斯大型建筑大模型 XL0.35 PRO.png")
            self.link_model_file("Stable-diffusion/SDXL/比鲁斯大型建筑大模型 XL0.35 PRO.safetensors")
            self.link_model_file("torch_deepdanbooru/model-resnet_custom_v3.pt")
            self.link_model_file("VAE/sdxl-vae-fp16-fix.safetensors")
            self.link_model_file("VAE/vae-ft-mse-840000-ema-pruned.ckpt")
            self.link_model_file("VAE-approx/model.pt")
            self.link_model_file("VAE-approx/vaeapprox-sd3.pt")
            self.link_model_file("VAE-approx/vaeapprox-sdxl.pt")
            self.link_model_file("SadTalker/hub/checkpoints/dinov2_vitl14_pretrain.pth")
            self.link_model_file("SadTalker/hub/checkpoints/tf_efficientnet_b5_ap-9e82fae8.pth")
            self.link_model_file("SadTalker/mapping_00109-model.pth.tar")
            self.link_model_file("SadTalker/mapping_00229-model.pth.tar")
            self.link_model_file("SadTalker/SadTalker_V0.0.2_256.safetensors")
            self.link_model_file("SadTalker/SadTalker_V0.0.2_512.safetensors")

            # 这里不能在启动的时候去复制拷贝... 因为用户可能自己会修改配置，除了模型，其他的还是交给用户，或者看是否可以在 install 阶段去处理
            # target = (Path(self.cfg.install_location) / "stable-diffusion-webui/models/SadTalker/hub/facebookresearch_dinov2_main").as_posix()
            # self.execute_command(f"/home/featurize/.public/sdwebui/models/SadTalker/hub/facebookresearch_dinov2_main {target}")

            # target = (Path(self.cfg.install_location) / "stable-diffusion-webui" / "configs").as_posix()
            # self.execute_command(f"cp -r /home/featurize/.public/sdwebui/c/configs {target}")

            # target_folder = (Path(self.cfg.install_location) / "stable-diffusion-webui").as_posix()
            # self.execute_command(f"cp /home/featurize/.public/sdwebui/c/config.json {target_folder}")
            # self.execute_command(f"cp /home/featurize/.public/sdwebui/c/ui-config.json {target_folder}")

        with self.conda_activate(self.conda_env_name, self.conda_mode):
            self.execute_command(
                f"bash webui.sh --listen --port {self.port} {self.cfg.launch_option}",
                daemon=True,
                cwd=self.source_location,
            )
        wait_for_port(self.port)
        self.app_started()


def main():
    return Sdwebui()
